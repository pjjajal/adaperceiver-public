import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import random
from itertools import product
import pandas as pd
from tqdm import tqdm
from pprint import pprint


import hydra
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from nutils.benchmark import benchmark_model, measure_flops
import evaluate


from dataset import create_dataset
from modules.layers.rope import precompute_freqs_cis
from modules.segmentation.segmenter import AdaptiveSeg
from modules.networks.dense_pred import DensePredictionAdaPerceiver
from modules.networks.factory import create_dense_prediction_model

torch.set_float32_matmul_precision("high")


@torch.inference_mode()
def eval(cfg, model: AdaptiveSeg, loader, device):
    @torch.inference_mode()
    def _eval_config(num_tokens, mat_dim, depth):
        metric = evaluate.load("mean_iou")

        # Progress bar for batch processing
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            for batch in tqdm(
                loader,
                desc=f"Evaluating num_tokens={num_tokens}, mat_dim={mat_dim}, depth={depth}",
                leave=False,
            ):
                img, label = batch
                img = img.to(device)
                label = label.to(device)

                freq_cis = precompute_freqs_cis(
                    dim=model.backbone.embed_dim // model.backbone.num_heads,
                    end=num_tokens,
                    train_len=model.backbone.max_latent_tokens,
                    theta=model.backbone.rope_theta,
                )
                output, _ = model.forward(
                    img,
                    num_tokens=num_tokens,
                    mat_dim=mat_dim,
                    depth=depth,  # +1 because depth is zero-indexeds
                    freq_cis=freq_cis,
                )
                metric.add_batch(predictions=output[-1].argmax(dim=1), references=label)

        metrics = metric.compute(
            num_labels=cfg.model.encoder_config.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        input_shape = img.shape
        # input_shape = (1, *input_shape[1:])  # for benchmarking, use batch size 1
        torch.cuda.reset_peak_memory_stats()
        freq_cis = precompute_freqs_cis(
            dim=model.backbone.embed_dim // model.backbone.num_heads,
            end=num_tokens,
            train_len=model.backbone.max_latent_tokens,
            theta=model.backbone.rope_theta,
        )
        benchmark_results = benchmark_model(
            model,
            input_shape,
            device,
            num_tokens=num_tokens,
            mat_dim=mat_dim,
            depth=depth,
            freq_cis=freq_cis,
        )
        memory = torch.cuda.max_memory_allocated()

        # We measure the backbone FLOPs only if using full attention
        flops = {"forward_total": 0.0, "backward_total": 0.0}
        if cfg.model.encoder_config.attn_layer == "full":
            flops_input_shape = (
                1,
                *input_shape[1:],
            )  # Use batch size of 1 for FLOPs measurement
            with torch.inference_mode(False):
                flops = measure_flops(
                    model.backbone,
                    flops_input_shape,
                    device,
                    num_tokens=num_tokens,
                    mat_dim=mat_dim,
                    depth=depth,
                    freq_cis=freq_cis,
                )

        return {
            "parameters": sum(p.numel() for p in model.parameters()),
            "batch_size": cfg.dataset.val_batch_size,
            "num_tokens": num_tokens,
            "mat_dim": mat_dim,
            "depth": depth,
            "mean_iou": float(metrics["mean_iou"]),
            "mean_accuracy": float(metrics["mean_accuracy"]),
            "overall_accuracy": float(metrics["overall_accuracy"]),
            "median_ms": float(benchmark_results.median * 1e3),
            "mean_ms": float(benchmark_results.mean * 1e3),
            "memory (GB)": (memory / (1024**3)),
            "flops forward [GFLOPs]": flops["forward_total"] / 1e9,
            "flops backward [GFLOPs]": flops["backward_total"] / 1e9,
        }

    eval_cfg = cfg.evaluation
    eval_depth = eval_cfg.eval_depth
    eval_depth = (
        range(cfg.model.encoder_config.depth)
        if eval_depth is None
        else [eval_depth - 1]  # zero-indexed.
    )

    results = []
    # Progress bar for configurations
    for mat_dim, num_tokens, depth in tqdm(
        product(
            cfg.evaluation.mat_dims,
            cfg.evaluation.token_grans,
            eval_depth,
        ),
        desc="Evaluating configurations",
        total=len(cfg.evaluation.mat_dims)
        * len(cfg.evaluation.token_grans)
        * len(eval_depth),
    ):
        eval_results = _eval_config(num_tokens, mat_dim, depth + 1)
        # Pretty print each evaluation result
        print("\nEvaluation Result:")
        pprint(eval_results)
        print("-" * 40)  # Separator for better readability
        results.append(eval_results)

    return pd.DataFrame(results)


@hydra.main(version_base=None, config_path="config/evaluation/segmentation")
def main(cfg: DictConfig):
    # Set seed
    L.seed_everything(cfg.seed)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Model and Data Setup
    backbone = create_dense_prediction_model(cfg.model)
    model = AdaptiveSeg(
        backbone,
        num_classes=cfg.model.encoder_config.num_classes,
        head_drop=cfg.model.encoder_config.seg_head_drop,
        cat_cls=cfg.model.encoder_config.cat_cls,
    )
    if cfg.model.model_checkpoint:
        print(f"Loading from checkpoint: {cfg.model.model_checkpoint}")
        state_dict = torch.load(cfg.model.model_checkpoint)
        state_dict = {
            key.replace("module.", ""): val for key, val in state_dict.items()
        }  # this handles the cause
        state_dict = {
            key.replace("_orig_mod.", ""): val for key, val in state_dict.items()
        }
        state_dict.pop("n_averaged", None)
        state_dict.pop("ema_n_averaged", None)
        model.load_state_dict(state_dict, strict=True)

    if cfg.model.compile:
        model.compile()

    # create dataset and dataloaders
    train_dataset, val_dataset, collate_fn = create_dataset(
        cfg.dataset, cfg.dataset.num_proc, eval_only=True
    )

    del train_dataset  # we don't need training dataset

    assert val_dataset is not None, "Validation dataset is required."
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.val_batch_size,
        num_workers=cfg.dataset.val_num_workers,
        pin_memory=cfg.dataset.pin_memory,
        collate_fn=collate_fn,
    )

    model = model.to(device)
    model = model.eval()

    eval_results = eval(cfg, model, val_loader, device)
    eval_results.to_csv(cfg.evaluation.output_csv)


if __name__ == "__main__":
    main()

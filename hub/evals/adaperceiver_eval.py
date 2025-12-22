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
from torchmetrics import Accuracy
from nutils.benchmark import benchmark_model, measure_flops


from dataset import create_dataset
from hub.networks.adaperceiver_classification import ClassificationAdaPerceiver


torch.set_float32_matmul_precision("high")


@torch.inference_mode()
def eval(cfg, model: ClassificationAdaPerceiver, loader, device):
    @torch.inference_mode()
    def _eval_config(num_tokens, mat_dim, depth):
        top_1_accuracy = Accuracy(
            task="multiclass", num_classes=model.num_classes, top_k=1
        ).to(device)
        top_5_accuracy = Accuracy(
            task="multiclass", num_classes=model.num_classes, top_k=5
        ).to(device)

        # # Progress bar for batch processing
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            for batch in tqdm(
                loader,
                desc=f"Evaluating num_tokens={num_tokens}, mat_dim={mat_dim}, depth={depth}",
                leave=False,
            ):
                img, label = batch
                img = img.to(device)
                label = label.to(device)
                with torch.no_grad():
                    output = model(
                        img,
                        num_tokens=num_tokens,
                        mat_dim=mat_dim,
                        depth=depth,  # +1 because depth is zero-indexeds
                    )
                    top_1_accuracy(output.logits.squeeze(), label)
                    top_5_accuracy(output.logits.squeeze(), label)

        input_shape = img.shape
        torch.cuda.reset_peak_memory_stats()
        benchmark_results = benchmark_model(
            model,
            input_shape,
            device,
            num_tokens=num_tokens,
            mat_dim=mat_dim,
            depth=depth,
        )
        memory = torch.cuda.max_memory_allocated()

        flops = {"forward_total": 0.0, "backward_total": 0.0}
        if model.attn_layer == "full":
            flops_input_shape = (
                1,
                *input_shape[1:],
            )  # Use batch size of 1 for FLOPs measurement
            with torch.inference_mode(False):
                flops = measure_flops(
                    model,
                    flops_input_shape,
                    device,
                    num_tokens=num_tokens,
                    mat_dim=mat_dim,
                    depth=depth,
                )

        return {
            "parameters": sum(p.numel() for p in model.parameters()),
            "batch_size": cfg.dataset.val_batch_size,
            "num_tokens": num_tokens,
            "mat_dim": mat_dim,
            "depth": depth,
            "top1": float(top_1_accuracy.compute()),  # Convert tensor to float
            "top5": float(top_5_accuracy.compute()),  # Convert tensor to float
            "median_ms": float(benchmark_results.median * 1e3),
            "mean_ms": float(benchmark_results.mean * 1e3),
            "memory (GB)": (memory / (1024**3)),
            "flops forward [GFLOPs]": flops["forward_total"] / 1e9,
            "flops backward [GFLOPs]": flops["backward_total"] / 1e9,
        }

    eval_cfg = cfg.evaluation
    eval_depth = eval_cfg.eval_depth
    eval_depth = (
        range(model.depth)
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


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    # Set seed
    L.seed_everything(cfg.seed)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create model
    model = ClassificationAdaPerceiver.from_pretrained(
        cfg.model.pretrained_model_name_or_path
    )

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

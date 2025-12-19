import hydra
import torch
from nutils.benchmark import benchmark_model, measure_flops
from omegaconf import DictConfig
from timm import create_model
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
import timm
from nutils.benchmark import benchmark_model, measure_flops
from pprint import pprint
from torchmetrics.regression import MeanSquaredError
from modules.depth.depther import BaselineDepth


from dataset import create_dataset

torch.set_float32_matmul_precision("high")


@torch.inference_mode()
def eval(cfg, model: BaselineDepth, loader, device):
    metric = MeanSquaredError(squared=False).to(device)
    # Progress bar for batch processing
    for batch in tqdm(
        loader,
    ):
        img, label = batch
        img = img.to(device)
        label = label.to(device)
        label = label.unsqueeze(1)
        img_flipped = img.flip(-1) # Test augmentation: horizontal flip
        img = torch.cat([img, img_flipped], dim=0)
        with torch.no_grad():
            preds = model.forward(img)
            preds1, preds2 = preds.chunk(2, dim=0)
            preds2 = preds2.flip(-1)
            preds = (preds1 + preds2) / 2.0  # Average the predictions
            metric(preds, label)


    input_shape = img.shape
    # input_shape = (1, *input_shape[1:])  # for benchmarking, use batch size 1
    torch.cuda.reset_peak_memory_stats()
    benchmark_results = benchmark_model(
        model,
        input_shape,
        device,
    )
    memory = torch.cuda.max_memory_allocated()

    num_tokens = model.backbone.forward_features(torch.randn(1, *input_shape[1:]).to(device)).shape[1]

    with torch.inference_mode(False):
        flops_input_shape = (
            1,
            *input_shape[1:],
        )  # Use batch size of 1 for FLOPs measurement
        flops = measure_flops(model.backbone, flops_input_shape, device)

    return {
        "parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        "batch_size": cfg.dataset.val_batch_size,
        "rmse": float(metric.compute()),  # Convert tensor to float
        "median_ms": float(benchmark_results.median * 1e3),
        "mean_ms": float(benchmark_results.mean * 1e3),
        "memory (GB)": (memory / (1024**3)),
        "flops forward [GFLOPs]": flops["forward_total"] / 1e9,
        "flops backward [GFLOPs]": flops["backward_total"] / 1e9,
        "num_tokens": num_tokens,
    }


@hydra.main(version_base=None, config_path="config/evaluation/depth/baseline")
def main(cfg: DictConfig):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = timm.create_model(cfg.model.name, pretrained=True, dynamic_img_size=True)
    model = BaselineDepth(
        backbone,
        num_bins=cfg.dataset.bins,
        head_drop=cfg.model.head_drop,
        cat_cls=cfg.model.cat_cls,
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        use_mlp=cfg.model.use_mlp,
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
        model = torch.compile(model)

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
    print("\nEvaluation Result:")
    pprint(eval_results)
    print("-" * 40)  # Separator for better readability


if __name__ == "__main__":
    main()

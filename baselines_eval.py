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


from dataset import create_dataset
torch.set_float32_matmul_precision('high')

@torch.inference_mode()
def eval(cfg, model, loader, device):
    top_1_accuracy = Accuracy(
        task="multiclass", num_classes=cfg.model.num_classes, top_k=1
    ).to(device)
    top_5_accuracy = Accuracy(
        task="multiclass", num_classes=cfg.model.num_classes, top_k=5
    ).to(device)

    # Progress bar for batch processing
    for batch in tqdm(
        loader,
    ):
        img, label = batch
        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model.forward(img)
            top_1_accuracy(output, label)
            top_5_accuracy(output, label)

    input_shape = img.shape
    torch.cuda.reset_peak_memory_stats()
    benchmark_results = benchmark_model(
        model,
        input_shape,
        device,
    )
    memory = torch.cuda.max_memory_allocated()

    flops_input_shape = (1, *input_shape[1:])  # Use batch size of 1 for FLOPs measurement
    with torch.inference_mode(False):
        flops = measure_flops(model, flops_input_shape, device)

    return {
        "parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        "batch_size": cfg.dataset.val_batch_size,
        "top1": float(top_1_accuracy.compute()),  # Convert tensor to float
        "top5": float(top_5_accuracy.compute()),  # Convert tensor to float
        "median_ms": float(benchmark_results.median * 1e3),
        "mean_ms": float(benchmark_results.mean * 1e3),
        "memory (GB)": (memory / (1024**3)),
        "flops forward": f"{flops['forward_total'] / 1e9:.2f} GFLOPs",
        "flops backward": f"{flops['backward_total'] / 1e9:.2f} GFLOPs",
    }


@hydra.main(version_base=None, config_path="config/evaluation")
def main(cfg: DictConfig):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(cfg.model.name, pretrained=True)
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

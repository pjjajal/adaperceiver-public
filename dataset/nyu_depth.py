from omegaconf import DictConfig

import torch
from datasets import load_dataset
import torchvision.transforms.v2 as tvt
from torchvision import tv_tensors
import numpy as np

from .utils import image_collate_fn


URL = "adams-story/nyu-depthv2-wds"

KEYS = {"image": "jpg", "label": "depth.npy"}


def nyu_depth_collate(batch, keys=KEYS):
    return image_collate_fn(batch, keys=keys)

def nyu_depth_train_transform(cfg: DictConfig):
    jitter = (
        tvt.ColorJitter(
            brightness=cfg.jitter.brightness,
            contrast=cfg.jitter.contrast,
            saturation=cfg.jitter.saturation,
            hue=cfg.jitter.hue,
        )
        if cfg.jitter.enable
        else tvt.Identity()
    )
    transform = tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.uint8, scale=True),
            tvt.RandomRotation(
                cfg.rotation, interpolation=tvt.InterpolationMode.BILINEAR
            ),
            tvt.Resize(cfg.img_size, antialias=True),
            jitter,
            tvt.RandomHorizontalFlip(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )

    def _transform(sample):
        sample["depth.npy"] = tv_tensors.Mask(sample["depth.npy"])
        return transform(sample)

    return _transform


def nyu_depth_val_transform(cfg: DictConfig):
    transform = tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.uint8, scale=True),
            tvt.Resize(cfg.img_size, antialias=True),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )

    def _transform(sample):
        sample["depth.npy"] = tv_tensors.Mask(sample["depth.npy"])
        return transform(sample)

    return _transform


def nyu_depth_train(transform=None, num_proc=1, cache_dir=None):
    dataset = load_dataset(
        URL,
        split="train",
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(transform)
    return dataset


def nyu_depth_val(transform=None, num_proc=1, cache_dir=None):
    dataset = load_dataset(
        URL,
        split="validation",
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(transform)
    return dataset

from typing import Sequence, Literal

import torch
import torchvision.transforms.v2 as tvt
from omegaconf import DictConfig


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Normalize:
    return tvt.Normalize(mean=mean, std=std)


def imagenet_train(cfg: DictConfig):
    transforms = [
        tvt.ToImage(),
        tvt.ToDtype(torch.uint8, scale=True),
        tvt.RandomResizedCrop(cfg.image_size),
        tvt.RandomHorizontalFlip(),
    ]
    if cfg.rand_augment:
        transforms.append(
            tvt.RandAugment(num_ops=cfg.num_ops, magnitude=cfg.magnitude)
        )
    transforms.append(tvt.ToDtype(torch.float32, scale=True))
    transforms.append(make_normalize_transform(mean=cfg.mean, std=cfg.std))
    return tvt.Compose(transforms)


def imagenet_val(cfg: DictConfig):
    return tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.uint8, scale=True),
            tvt.Resize(cfg.max_crop_size, antialias=True),
            tvt.CenterCrop(cfg.val_image_size),
            tvt.ToDtype(torch.float32, scale=True),
            make_normalize_transform(mean=cfg.mean, std=cfg.std),
        ]
    )

from omegaconf import DictConfig
from .imagenet import imagenet_train, imagenet_val, imagenet_collate_fn
from .imagenet12k import imagenet12k_train, imagenet12k_val, imagenet12k_collate_fn
from .transform import (
    imagenet_train as imagenet_train_transform,
    imagenet_val as imagenet_val_transform,
)
from .scene_parsing_ade20k import (
    scene_parsing_ade20k_train,
    scene_parsing_ade20k_val,
    scene_parsing_ade20k_train_transform,
    scene_parsing_ade20k_val_transform,
    scene_parsing_ade20k_collate,
)
from .nyu_depth import (
    nyu_depth_train_transform,
    nyu_depth_val_transform,
    nyu_depth_train,
    nyu_depth_val,
    nyu_depth_collate
)


def create_dataset(cfg: DictConfig, num_proc=1, eval_only=False, cache_dir=None):
    train_dataset = None
    val_dataset = None
    collate_fn = None

    if cfg.name == "imagenet":
        train_transform = imagenet_train_transform(cfg.transforms)
        val_transform = imagenet_val_transform(cfg.transforms)
        if not eval_only:
            train_dataset = imagenet_train(
                transform=train_transform, num_proc=num_proc, cache_dir=cache_dir
            )
        val_dataset = imagenet_val(
            transform=val_transform, num_proc=num_proc, cache_dir=cache_dir
        )
        collate_fn = imagenet_collate_fn
    elif cfg.name == "imagenet12k":
        train_transform = imagenet_train_transform(cfg.transforms)
        val_transform = imagenet_val_transform(cfg.transforms)
        if not eval_only:
            train_dataset = imagenet12k_train(
                transform=train_transform, num_proc=num_proc, cache_dir=cache_dir
            )
        val_dataset = imagenet12k_val(
            transform=val_transform, num_proc=num_proc, cache_dir=cache_dir
        )
        collate_fn = imagenet12k_collate_fn
    elif cfg.name == "scene_parsing_ade20k":
        train_transform = scene_parsing_ade20k_train_transform(cfg.transforms)
        val_transform = scene_parsing_ade20k_val_transform(cfg.transforms)
        train_dataset = scene_parsing_ade20k_train(
            transform=train_transform, num_proc=num_proc, cache_dir=cache_dir
        )
        val_dataset = scene_parsing_ade20k_val(
            transform=val_transform, num_proc=num_proc, cache_dir=cache_dir
        )
        collate_fn = scene_parsing_ade20k_collate
    elif cfg.name == "nyu_depth":
        train_transform = nyu_depth_train_transform(cfg.transforms)
        val_transform = nyu_depth_val_transform(cfg.transforms)
        train_dataset = nyu_depth_train(
            transform=train_transform, num_proc=num_proc, cache_dir=cache_dir
        )
        val_dataset = nyu_depth_val(
            transform=val_transform, num_proc=num_proc, cache_dir=cache_dir
        )
        collate_fn = nyu_depth_collate
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")
    return train_dataset, val_dataset, collate_fn

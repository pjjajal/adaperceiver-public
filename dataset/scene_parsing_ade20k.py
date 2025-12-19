from omegaconf import DictConfig

from datasets import load_dataset
from functools import partial
from transformers import AutoImageProcessor
from torchvision.transforms.v2 import ColorJitter
from .utils import image_collate_fn


URL = "zhoubolei/scene_parse_150"
DATA_DIR = "scene_parsing"
REVISION = "refs/convert/parquet"

KEYS = {"image": "image", "label": "annotation"}
PROCESSOR_KEY = {"image": "pixel_values", "label": "labels"}


def scene_parsing_ade20k_collate(batch, keys=PROCESSOR_KEY):
    return image_collate_fn(batch, keys=keys)


def scene_parsing_ade20k_train_transform(cfg: DictConfig):
    image_processor = AutoImageProcessor.from_pretrained(
        cfg.image_processor,
        do_reduce_labels=True,
        image_mean=list(cfg.mean),
        image_std=list(cfg.std),
        size={"height": cfg.image_size, "width": cfg.image_size},
    )
    jitter = (
        ColorJitter(
            brightness=cfg.jitter.brightness,
            contrast=cfg.jitter.contrast,
            saturation=cfg.jitter.saturation,
            hue=cfg.jitter.hue,
        )
        if cfg.jitter.enable
        else lambda x: x
    )

    def _transform(batch):
        images = [jitter(img.convert("RGB")) for img in batch[KEYS["image"]]]
        labels = [ann for ann in batch[KEYS["label"]]]
        inputs = image_processor(images, labels)
        return inputs

    return _transform


def scene_parsing_ade20k_val_transform(cfg: DictConfig):
    image_processor = AutoImageProcessor.from_pretrained(
        cfg.image_processor,
        do_reduce_labels=True,
        image_mean=list(cfg.mean),
        image_std=list(cfg.std),
        size={"height": cfg.image_size, "width": cfg.image_size},
    )

    def _transform(batch):
        images = [img.convert("RGB") for img in batch[KEYS["image"]]]
        labels = [ann for ann in batch[KEYS["label"]]]
        inputs = image_processor(images, labels)
        return inputs

    return _transform


def scene_parsing_ade20k_train(transform=None, num_proc=1, cache_dir=None):
    dataset = load_dataset(
        URL,
        split="train",
        data_dir=DATA_DIR,
        revision=REVISION,
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(transform)
    return dataset


def scene_parsing_ade20k_val(transform=None, num_proc=1, cache_dir=None):
    dataset = load_dataset(
        URL,
        split="validation",
        data_dir=DATA_DIR,
        revision=REVISION,
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(transform)
    return dataset

from datasets import load_dataset
from functools import partial
from .utils import image_collate_fn, process_data

TOTAL_SAMPLES = 1_281_167
TOTAL_VAL_SAMPLES = 50_000
TOTAL_CLASSES = 1000

URL = "timm/imagenet-1k-wds"
KEYS = {"image": "jpg", "label": "cls"}


def imagenet_collate_fn(batch):
    return image_collate_fn(batch, keys=KEYS)


def imagenet_train(transform=None, num_proc=1, cache_dir=None):
    dataset = load_dataset(URL, split="train", num_proc=num_proc, cache_dir=cache_dir)
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(
        partial(process_data, transform=transform, image_key=KEYS["image"])
    )
    return dataset


def imagenet_val(transform=None, num_proc=1, cache_dir=None):
    dataset = load_dataset(URL, split="validation", num_proc=num_proc, cache_dir=cache_dir)
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(
        partial(process_data, transform=transform, image_key=KEYS["image"])
    )
    return dataset

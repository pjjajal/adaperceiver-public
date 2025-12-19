from datasets import load_dataset
from functools import partial
from .utils import image_collate_fn, process_data

TOTAL_SAMPLES = 12_129_687
TOTAL_VAL_SAMPLES = 472_840
TOTAL_CLASSES = 11_821

URL = "timm/imagenet-12k-wds"
KEYS = {"image": "jpg", "label": "cls"}

def imagenet12k_collate_fn(batch):
    return image_collate_fn(batch, keys=KEYS)

def imagenet12k_train(transform=None, num_proc=1, cache_dir=None):
    dataset = load_dataset(URL, split="train", num_proc=num_proc, cache_dir=cache_dir)
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(
        partial(process_data, transform=transform, image_key=KEYS["image"])
    )
    return dataset

def imagenet12k_val(transform=None, num_proc=1, cache_dir=None):
    dataset = load_dataset(URL, split="validation", num_proc=num_proc, cache_dir=cache_dir)
    dataset = dataset.select_columns(list(KEYS.values()))
    dataset.set_transform(
        partial(process_data, transform=transform, image_key=KEYS["image"])
    )
    return dataset
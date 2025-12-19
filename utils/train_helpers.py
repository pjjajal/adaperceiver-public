import sys
from datetime import datetime
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig


DEFAULT_CHECKPOINTS_PATH = Path("./checkpoints")


def setup_checkpoint_dir(
    cfg: DictConfig,
):
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S.%f")
    ### In case we want to add a more descriptive checkpoint dir folder
    if cfg.checkpoint.checkpoint_name:
        now = (
            now
            + "-"
            + cfg.checkpoint.checkpoint_name.replace(" ", "-").replace("_", "-")
        )
    save_base_path = (
        Path(cfg.checkpoint.checkpoint_save_dir)
        if cfg.checkpoint.checkpoint_save_dir
        else DEFAULT_CHECKPOINTS_PATH
    )
    save_loc = (
        cfg.checkpoint.save_loc if cfg.checkpoint.save_loc else save_base_path / now
    )
    return save_loc


def setup_logging(cfg: DictConfig, save_loc: Path):
    """
    Configure loguru and optional Weights & Biases logging.
    """
    # Clear any default handlers
    logger.remove()
    # Console logging
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    # File logging
    log_file = save_loc / "train.log"
    logger.add(log_file, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")
    # Initialize Weights & Biases if enabled
    run = None
    # if cfg.logging.wandb:
    #     run = wandb.init(
    #         project=cfg.logging.wandb_project,
    #         dir=save_loc,
    #         config=flatten_dict(cfg),
    #         tags=cfg.logging.wandb_tags,
    #         notes=cfg.logging.wandb_notes,
    #     )
    return run


def flatten_dict(d: DictConfig):
    out = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            for k2, v2 in flatten_dict(v).items():
                out[k + "." + k2] = v2
        else:
            out[k] = v
    return out

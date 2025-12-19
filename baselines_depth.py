from datetime import datetime
from pathlib import Path

import evaluate
import hydra
import lightning as L
import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tvt
from datasets.distributed import split_dataset_by_node
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig, OmegaConf
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanSquaredError

from dataset import create_dataset
from modules.depth.depther import BaselineDepth
from modules.depth.losses import SigLoss, GradientLoss

try:
    from distributed_shampoo import (
        AdamGraftingConfig,
        DefaultEigenvalueCorrectedShampooConfig,
        DistributedShampoo,
    )

    SHAMPOO_AVAILABLE = True
except ImportError:
    print("Distributed Shampoo not available")
    SHAMPOO_AVAILABLE = False

DEFAULT_CHECKPOINTS_PATH = Path("./checkpoints")

torch.set_float32_matmul_precision("medium")


def setup_checkpoint_dir(cfg: DictConfig):
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    ### In case we want to add a more descriptive checkpoint dir folder
    if cfg.trainer.checkpoint_name:
        now = (
            now + "-" + cfg.trainer.checkpoint_name.replace(" ", "-").replace("_", "-")
        )
    save_base_path = (
        Path(cfg.trainer.checkpoint_save_dir)
        if cfg.trainer.checkpoint_save_dir
        else DEFAULT_CHECKPOINTS_PATH
    )
    save_loc = cfg.trainer.save_loc if cfg.trainer.save_loc else save_base_path / now
    print(f"Saving checkpoints to: {save_loc}")
    return save_loc


def flatten_dict(d: DictConfig):
    out = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            for k2, v2 in flatten_dict(v).items():
                out[k + "." + k2] = v2
        else:
            out[k] = v
    return out


class Model(L.LightningModule):
    def __init__(
        self,
        model: BaselineDepth,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_bins = cfg.dataset.bins
        self.model = model

        self.sig_loss = SigLoss(
            loss_weight=cfg.loss.sig_loss_weight, max_depth=cfg.dataset.max_depth, warm_up=cfg.loss.sig_loss_warmup
        )
        self.grad_loss = GradientLoss(
            loss_weight=cfg.loss.grad_loss_weight, max_depth=cfg.dataset.max_depth
        )

        # EMA Student.
        self.ema_model = None
        if cfg.optimizer.ema > 0:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                self.model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                    cfg.optimizer.ema
                ),
            )

        # Counters/Metrics
        self.running_loss = 0

        self.best_rmse = float("inf")
        self.best_rmse_ema = float("inf")
        self.rmse = MeanSquaredError(squared=False)
        self.rmse_ema = MeanSquaredError(squared=False)

    def training_step(self, batch, batch_idx):
        x, label = batch
        label = label.unsqueeze(1)
        preds = self.model(x)
        sig_loss = self.sig_loss(preds, label)
        grad_loss = self.grad_loss(preds, label)
        loss = sig_loss + grad_loss
        self.running_loss += loss.detach().item()
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        label = label.unsqueeze(1)
        x_flipped = x.flip(-1) # Test augmentation: horizontal flip
        x = torch.cat([x, x_flipped], dim=0)
        preds = self.model(x)
        preds1, preds2 = preds.chunk(2, dim=0)
        preds2 = preds2.flip(-1)
        preds = (preds1 + preds2) / 2.0  # Average the predictions
        sig_loss = self.sig_loss(preds, label)
        grad_loss = self.grad_loss(preds, label)
        loss = sig_loss + grad_loss

        self.rmse(preds, label)
        self.log(
            "val/test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/rmse",
            self.rmse,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.ema_model:
            preds = self.ema_model(x)
            preds1, preds2 = preds.chunk(2, dim=0)
            preds2 = preds2.flip(-1)
            preds = (preds1 + preds2) / 2.0  # Average the predictions
            ema_loss = self.sig_loss(preds, label) + self.grad_loss(preds, label)
            self.rmse_ema(preds, label)
            self.log(
                "val/ema_test_loss",
                ema_loss,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "val/ema_rmse",
                self.rmse_ema,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        optimizer_cfg = self.cfg.optimizer
        # Determine parameters to optimize.
        parameters = list(self.model.head.parameters())
        if self.cfg.model.use_mlp:
            parameters += list(self.model.mlp.parameters())
        if optimizer_cfg.optimizer in ["shampoo", "shampoo-soap"]:
            assert SHAMPOO_AVAILABLE, "Shampoo not available."
            optimizer = DistributedShampoo(
                parameters,
                lr=optimizer_cfg.lr,
                betas=optimizer_cfg.betas,
                epsilon=1e-12,
                weight_decay=optimizer_cfg.weight_decay,
                precondition_frequency=optimizer_cfg.preconditioning_frequency,
                max_preconditioner_dim=optimizer_cfg.max_preconditioner_dim,
                start_preconditioning_step=optimizer_cfg.start_preconditioning_step,
                use_decoupled_weight_decay=True,
                grafting_config=AdamGraftingConfig(
                    beta2=optimizer_cfg.betas[-1],
                    epsilon=1e-8,
                ),
                preconditioner_config=(
                    DefaultEigenvalueCorrectedShampooConfig
                    if optimizer_cfg.optimizer == "shampoo-soap"
                    else None
                ),
            )
        else:
            optimizer = create_optimizer_v2(
                parameters,
                opt=optimizer_cfg.optimizer,
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay,
                betas=optimizer_cfg.betas,
                caution=optimizer_cfg.caution,
            )

        print(
            f"Using {optimizer_cfg.optimizer} optimizer with scheduler: ",
            optimizer_cfg.schedule,
            "for ",
            optimizer_cfg.total_steps,
            "steps",
            "with warmup steps: ",
            optimizer_cfg.warmup_steps,
        )

        scheduler, _ = create_scheduler_v2(
            optimizer=optimizer,
            sched=optimizer_cfg.schedule,
            num_epochs=optimizer_cfg.total_steps,
            warmup_epochs=optimizer_cfg.warmup_steps,
            min_lr=optimizer_cfg.min_lr,
            warmup_lr=optimizer_cfg.warmup_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        if self.cfg.trainer.gradnorm_logging:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema_model:
            self.ema_model.update_parameters(self.model)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.global_step)

    def on_train_epoch_end(self) -> None:
        # Save Model
        if self.global_rank == 0:
            save_path = self.cfg.trainer.save_loc / "latest_train_epoch.pth"
            torch.save(self.model.state_dict(), save_path)
            if self.ema_model:
                torch.save(
                    self.ema_model.state_dict(),
                    (self.cfg.trainer.save_loc / "latest_train_epoch_ema.pth"),
                )

    def on_validation_epoch_end(self) -> None:
        rmse = self.rmse.compute()
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            if self.global_rank == 0:
                save_path = self.cfg.trainer.save_loc / "best.pth"
                torch.save(self.model.state_dict(), save_path)
        if self.ema_model:
            rmse_ema = self.rmse_ema.compute()
            if rmse_ema < self.best_rmse_ema:
                self.best_rmse_ema = rmse_ema
                if self.global_rank == 0:
                    save_path = self.cfg.trainer.save_loc / "best_ema.pth"
                    torch.save(self.ema_model.state_dict(), save_path)


@hydra.main(version_base=None, config_path="config/depth/baselines")
def main(cfg: DictConfig):
    # Set seed
    L.seed_everything(cfg.seed)

    # checkpoint location
    save_loc = setup_checkpoint_dir(cfg)
    cfg.trainer.save_loc = save_loc

    # wandb logger
    if cfg.trainer.wandb:
        wandb_logger = WandbLogger(
            project=cfg.trainer.wandb_project, save_dir=cfg.trainer.wandb_save_dir
        )

    callbacks = []
    # lr monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # use rich printing
    if cfg.trainer.rich_print:
        callbacks.append(RichModelSummary())
        callbacks.append(RichProgressBar())

    # Checkpoint setup
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.trainer.save_loc,
        filename="{epoch:02d}",
        save_on_train_epoch_end=True,
        save_weights_only=(
            True if cfg.optimizer.optimizer in ["shampoo", "shampoo-soap"] else False
        ),
    )
    callbacks.append(checkpoint_callback)

    trainer = L.Trainer(
        # distributed settings
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        precision=cfg.trainer.precision,
        strategy=cfg.trainer.strategy,
        # callbacks and logging
        callbacks=[lr_monitor, *callbacks],
        logger=wandb_logger if cfg.trainer.wandb else None,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        # training related
        max_epochs=cfg.optimizer.max_epochs,
        gradient_clip_val=cfg.optimizer.grad_clip,
        accumulate_grad_batches=cfg.optimizer.accumulate_grad_batches,
        overfit_batches=cfg.optimizer.overfit_batches,
        use_distributed_sampler=cfg.dataset.use_distributed_sampler,
        benchmark=True,  # cudnn benchmarking, allows for faster training.
        # enable_checkpointing=False,  # Disable automatic checkpointing (we do this manually).
    )

    ### Model and Data Setup
    backbone = timm.create_model(cfg.model.name, pretrained=True, dynamic_img_size=True)
    seg_model = BaselineDepth(
        backbone,
        num_bins=cfg.dataset.bins,
        head_drop=cfg.model.head_drop,
        cat_cls=cfg.model.cat_cls,
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        use_mlp=cfg.model.use_mlp,
    )
    if cfg.model.compile:
        seg_model = torch.compile(seg_model, mode=cfg.model.compile_mode)

    # create dataset and dataloaders
    train_dataset, val_dataset, collate_fn = create_dataset(
        cfg.dataset, cfg.dataset.num_proc
    )
    train_dataset = split_dataset_by_node(
        train_dataset, rank=trainer.global_rank, world_size=trainer.world_size
    )
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False if cfg.optimizer.overfit_batches else True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    cfg.optimizer.total_steps = (
        len(loader)
        * cfg.optimizer.max_epochs
        // (cfg.optimizer.accumulate_grad_batches)
    )
    val_loader = None
    if val_dataset is not None:
        val_dataset = split_dataset_by_node(
            val_dataset, rank=trainer.global_rank, world_size=trainer.world_size
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.dataset.val_batch_size,
            num_workers=cfg.dataset.val_num_workers,
            pin_memory=cfg.dataset.pin_memory,
            collate_fn=collate_fn,
        )

    model = Model(seg_model, cfg)

    # log config info to wandb
    if trainer.global_rank == 0:
        if cfg.trainer.wandb:
            wandb_logger.experiment.config.update(
                {
                    # "seed": cfg.seed,
                    # **flatten_dict(cfg.trainer),
                    # **flatten_dict(cfg.model),
                    # **flatten_dict(cfg.dataset),
                    # **flatten_dict(cfg.optimizer),
                    **flatten_dict(cfg),
                }
            )
            save_loc = cfg.trainer.save_loc
            save_loc.mkdir(parents=True, exist_ok=True)
            # Save config.
            OmegaConf.save(cfg, save_loc / "config.yaml")

    # Trainer Fit.
    trainer.fit(
        model,
        train_dataloaders=loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.trainer.resume_checkpoint,
    )


if __name__ == "__main__":
    main()

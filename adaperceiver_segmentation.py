from datetime import datetime
from pathlib import Path

import random
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

from dataset import create_dataset
from modules.helpers.loss import depth_loss_weights, DEPTH_LOSSES
from modules.segmentation.segmenter import AdaptiveSeg
from modules.networks.dense_pred import DensePredictionAdaPerceiver
from modules.networks.factory import create_dense_prediction_model
from modules.layers.rope import precompute_freqs_cis
from torchmetrics.segmentation import MeanIoU


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

torch.set_float32_matmul_precision("high")


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
        model: AdaptiveSeg,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.dataset.num_classes
        self.model = model

        self.model_cfg = cfg.model

        # Set the mat_dims and num_tokens
        mat_dims = self.model_cfg.mat_dims
        token_grans = self.model_cfg.token_grans
        self.mat_dims = [self.model_cfg.embed_dim] if not mat_dims else mat_dims
        self.token_grans = (
            [self.model.max_latent_tokens] if not token_grans else token_grans
        )

        # Token Loss
        # NOTE: We allow for token loss to be applied regradless of masking type.
        self.token_loss = cfg.loss.token_loss.enable
        self.total_token_loss_weight = cfg.loss.token_loss.weight
        assert len(self.cfg.loss.token_loss.token_loss_weights) == len(
            self.token_grans
        ), "Adaptive Token loss weights must match num_tokens"
        self.sample_tokens = cfg.loss.token_loss.sample_tokens
        self.token_loss_weights = self.cfg.loss.token_loss.token_loss_weights

        # Layer Loss
        self.depth_loss = cfg.loss.depth_loss.enable
        self.total_depth_loss_weight = cfg.loss.depth_loss.weight
        self.depth_loss_type = self.cfg.loss.depth_loss.depth_loss_type
        assert (
            self.depth_loss_type in DEPTH_LOSSES
        ), f"Depth loss weights must be in {DEPTH_LOSSES}"
        self.depth_loss_weights = depth_loss_weights(
            depth=self.model.backbone.depth, depth_loss_type=self.depth_loss_type
        )

        # Matyroshka Loss
        self.matyr_loss = cfg.loss.matyr_loss.enable

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

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
        self.miou = evaluate.load("mean_iou")
        self.miou_ema = evaluate.load("mean_iou")
        self.best_miou = 0.0
        self.best_miou_ema = 0.0

    def compute_loss(self, pred_list, label, loss_weights):
        loss = sum(
            [
                w * self.criterion(output, label)
                for w, output in zip(loss_weights, pred_list)
            ]
        )
        loss = loss / sum(loss_weights)
        return loss

    def training_step(self, batch, batch_idx):
        x, label = batch

        # Pick a random number of tokens and mat_dim.
        tokens = random.choice(self.token_grans)
        # grab mat_dims per batch
        mat_dim = [random.choice(self.mat_dims) for _ in range(x.shape[0])]

        # Forward Pass
        freq_cis = precompute_freqs_cis(
            dim=self.model.backbone.embed_dim // self.model.backbone.num_heads,
            end=self.token_grans[-1],
            train_len=self.model.backbone.max_latent_tokens,
            theta=self.model.backbone.rope_theta,
        )
        output_list, inter_list = self.model.forward(
            x,
            num_tokens=tokens if self.sample_tokens else self.token_grans[-1],
            mat_dim=mat_dim,
            token_loss=self.token_loss,
            layer_loss=self.depth_loss,
            freq_cis=freq_cis,
        )
        # Compute Losses

        ## Token Loss
        loss = 0.0
        token_loss = self.compute_loss(
            output_list,
            label,
            self.token_loss_weights if self.token_loss else [1.0],
        )
        loss += (
            token_loss * self.total_token_loss_weight if self.token_loss else token_loss
        )
        ## Depth Loss
        depth_loss = 0.0
        if self.depth_loss:
            depth_loss = self.compute_loss(
                inter_list,
                label,
                self.depth_loss_weights,
            )
            loss += depth_loss * self.total_depth_loss_weight

        self.running_loss += loss.detach().item()
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/token_loss",
            token_loss,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/depth_loss",
            depth_loss,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        # Forward Pass
        freq_cis = precompute_freqs_cis(
            dim=self.model.backbone.embed_dim // self.model.backbone.num_heads,
            end=self.token_grans[-1],
            train_len=self.model.backbone.max_latent_tokens,
            theta=self.model.backbone.rope_theta,
        )
        output_list, inter_list = self.model.forward(
            x,
            num_tokens=self.model.backbone.max_latent_tokens,
            mat_dim=None,
            token_loss=self.token_loss,
            layer_loss=self.depth_loss,
            freq_cis=freq_cis,
        )
        # Compute Losses
        loss = 0.0
        ## Token Loss
        token_loss = self.compute_loss(
            output_list,
            label,
            self.token_loss_weights if self.token_loss else [1.0],
        )
        loss += (
            token_loss * self.total_token_loss_weight if self.token_loss else token_loss
        )
        ## Depth Loss
        depth_loss = 0.0
        if self.depth_loss:
            depth_loss = self.compute_loss(
                inter_list,
                label,
                self.depth_loss_weights,
            )
            loss += depth_loss * self.total_depth_loss_weight

        # For mIoU we only want the final output
        self.miou.add_batch(predictions=output_list[-1].argmax(dim=1), references=label)
        self.log(
            "val/test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if self.ema_model:
            ema_output_list, ema_inter_list = self.ema_model.forward(
                x,
                num_tokens=self.model.backbone.max_latent_tokens,
                mat_dim=None,
                token_loss=self.token_loss,
                layer_loss=self.depth_loss,
                freq_cis=freq_cis,
            )
            # Compute Losses
            ema_loss = 0.0
            ## Token Loss
            ema_token_loss = self.compute_loss(
                ema_output_list,
                label,
                self.token_loss_weights if self.token_loss else [1.0],
            )
            ema_loss += (
                ema_token_loss * self.total_token_loss_weight
                if self.token_loss
                else ema_token_loss
            )
            ## Depth Loss
            ema_depth_loss = 0.0
            if self.depth_loss:
                ema_depth_loss = self.compute_loss(
                    ema_inter_list,
                    label,
                    self.depth_loss_weights,
                )
                ema_loss += ema_depth_loss * self.total_depth_loss_weight
            self.miou_ema.add_batch(
                predictions=ema_output_list[-1].argmax(dim=1), references=label
            )
            self.log(
                "val/ema_test_loss",
                ema_loss,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        optimizer_cfg = self.cfg.optimizer
        # Determine parameters to optimize.
        parameters = []
        parameters += list(self.model.head.parameters())
        parameters += list(self.model.backbone.output_adapter.parameters())
        parameters += list(self.model.backbone.output_latents.parameters())
        parameters += list(self.model.backbone.write_head.parameters())
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
        metrics = self.miou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )
        self.log("val/miou", metrics["mean_iou"], sync_dist=True)
        if metrics["mean_iou"] > self.best_miou:
            self.best_miou = metrics["mean_iou"]
            if self.global_rank == 0:
                save_path = self.cfg.trainer.save_loc / "best_miou.pth"
                torch.save(self.model.state_dict(), save_path)
                print(
                    f"New best mIoU: {self.best_miou:.4f}. Saved model to {save_path}"
                )

        if self.ema_model:
            metrics_ema = self.miou_ema.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
            )
            self.log("val/miou_ema", metrics_ema["mean_iou"], sync_dist=True)
            if metrics_ema["mean_iou"] > self.best_miou_ema:
                self.best_miou_ema = metrics_ema["mean_iou"]
                if self.global_rank == 0:
                    save_path = self.cfg.trainer.save_loc / "best_miou_ema.pth"
                    torch.save(self.ema_model.state_dict(), save_path)
                    print(
                        f"New best EMA mIoU: {self.best_miou_ema:.4f}. Saved EMA model to {save_path}"
                    )


@hydra.main(version_base=None, config_path="config/segmentation")
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
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=cfg.trainer.save_loc,
    #     filename="{epoch:02d}",
    #     save_on_train_epoch_end=True,
    #     save_weights_only=(
    #         True if cfg.optimizer.optimizer in ["shampoo", "shampoo-soap"] else False
    #     ),
    # )
    # callbacks.append(checkpoint_callback)

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
    backbone = create_dense_prediction_model(cfg.model)
    if cfg.model.model_checkpoint:
        print(f"Loading from checkpoint: {cfg.model.model_checkpoint}")
        state_dict = torch.load(cfg.model.model_checkpoint, map_location="cpu")
        state_dict = {
            key.replace("module.", ""): val for key, val in state_dict.items()
        }
        state_dict = {
            key.replace("_orig_mod.", ""): val for key, val in state_dict.items()
        }
        state_dict["process_latents.token"] = state_dict[
            "process_latents.token"
        ]
        state_dict["patch_embed.pos_embed"] = state_dict[
            "patch_embed.pos_embed"
        ]
        state_dict["output_latents.token"] = state_dict[
            "output_latents.token"
        ]  # .squeeze(0)
        for key in list(state_dict.keys()):
            if "write_head_feat" in key:
                new_key = key.replace("write_head_feat", "write_head")
                state_dict[new_key] = state_dict[key].clone()
                state_dict.pop(key)
            if "output_adapter_feat" in key:
                new_key = key.replace("output_adapter_feat", "output_adapter")
                state_dict[new_key] = state_dict[key].clone()
                state_dict.pop(key)
            if "output_latents_feat" in key:
                new_key = key.replace("output_latents_feat", "output_latents")
                state_dict[new_key] = state_dict[key].clone()
                state_dict.pop(key)

        missing_keys = backbone.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")

    seg_model = AdaptiveSeg(
        backbone,
        num_classes=cfg.model.encoder_config.num_classes,
        head_drop=cfg.model.encoder_config.seg_head_drop,
        cat_cls=cfg.model.encoder_config.cat_cls,
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
                    "seed": cfg.seed,
                    **flatten_dict(cfg.trainer),
                    **flatten_dict(cfg.model),
                    **flatten_dict(cfg.dataset),
                    **flatten_dict(cfg.optimizer),
                    **flatten_dict(cfg.loss),
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

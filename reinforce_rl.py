import time
import argparse
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import product
from tqdm import tqdm
from pprint import pprint


import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader


from dataset import create_dataset
from modules.networks.classification import ClassificationAdaPerceiver
from modules.networks.factory import create_model
from modules.layers.rope import precompute_freqs_cis
from timm.models.mlp_mixer import MixerBlock
from timm.layers import Mlp
import wandb
from collections import Counter
from dataclasses import dataclass
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torchmetrics import Accuracy


torch.set_float32_matmul_precision("medium")


def flatten_dict(d: DictConfig):
    out = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            for k2, v2 in flatten_dict(v).items():
                out[k + "." + k2] = v2
        else:
            out[k] = v
    return out


class RewardFunction(nn.Module):
    def __init__(
        self,
        n_tokens,
        n_depth,
        n_mat_dim,
        reward_scale=1.0,
        lambda_cost=0.0,
        token_weight=1.0,
        depth_weight=1.0,
        mat_dim_weight=1.0,
        reward_type="ce",
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_depth = n_depth
        self.n_mat_dim = n_mat_dim
        self.reward_type = reward_type

        total_weight = token_weight + depth_weight + mat_dim_weight
        self.token_weight = token_weight / total_weight
        self.depth_weight = depth_weight / total_weight
        self.mat_dim_weight = mat_dim_weight / total_weight

        self.lambda_cost = lambda_cost
        self.reward_scale = reward_scale

    def computational_cost_reward(
        self,
        sampled_token_idxs=None,
        sampled_depth_idxs=None,
        sampled_mat_dim_idxs=None,
    ):

        norm_token = (
            sampled_token_idxs.float() / (self.n_tokens - 1)
            if sampled_token_idxs is not None
            else 0.0
        )
        norm_depth = (
            sampled_depth_idxs.float() / (self.n_depth - 1)
            if sampled_depth_idxs is not None
            else 0.0
        )
        norm_mat_dim = (
            sampled_mat_dim_idxs.float() / (self.n_mat_dim - 1)
            if sampled_mat_dim_idxs is not None
            else 0.0
        )

        cost = (
            self.token_weight * norm_token
            + self.depth_weight * norm_depth
            + self.mat_dim_weight * norm_mat_dim
        )
        return cost

    def accuracy_reward(self, predictions, targets):
        if self.reward_type == "argmax":
            rewards = (predictions.argmax(dim=-1) == targets).float()
        elif self.reward_type == "ce":
            # We use negative cross-entropy as reward
            rewards = -F.cross_entropy(
                predictions.squeeze(1), targets, reduction="none"
            )
        return rewards

    def forward(
        self,
        predictions,
        targets,
        sampled_token_idxs=None,
        sampled_depth_idxs=None,
        sampled_mat_dim_idxs=None,
    ):
        accuracy = self.accuracy_reward(predictions, targets)
        cost = self.computational_cost_reward(
            sampled_token_idxs, sampled_depth_idxs, sampled_mat_dim_idxs
        )
        return accuracy, cost


class PolicyNetwork(nn.Module):
    def __init__(self, dim, seq_len, token_choices, depth_choices, mat_dim_choices):
        super(PolicyNetwork, self).__init__()
        self.dim = dim
        self.token_choices = token_choices
        self.depth_choices = depth_choices
        self.mat_dim_choices = mat_dim_choices

        self.mixer_block = MixerBlock(dim=dim, seq_len=seq_len)
        self.mixer_block_2 = MixerBlock(dim=dim, seq_len=seq_len)
        # self.mixer_block_3 = MixerBlock(dim=dim, seq_len=seq_len)

        # Small fusion MLP after pooling
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        self.head_tokens = nn.Linear(dim, len(token_choices), bias=False)
        self.head_depth = nn.Linear(dim, len(depth_choices), bias=False)
        self.head_mat_dim = nn.Linear(dim, len(mat_dim_choices), bias=False)

    def forward(self, x):
        x = self.mixer_block(x)
        x = self.mixer_block_2(x)
        # x = self.mixer_block_3(x)
        x = x.mean(dim=1)  # Global average pooling

        h = self.fuse(x)

        logits_tokens = self.head_tokens(h)
        logits_depth = self.head_depth(h)
        logits_mat_dim = self.head_mat_dim(h)
        return logits_tokens, logits_depth, logits_mat_dim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load("config/rl_reinforce/soadaperciever_1k.yaml")

    wandb.init(project="adapercevier-rl", config=flatten_dict(cfg))

    # Create dataset and dataloader
    train_dataset, val_dataset, collate_fn = create_dataset(
        cfg.dataset, cfg.dataset.num_proc
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=8,
    )

    # Create model and load pretrained weights
    model = create_model(cfg.model)
    state_dict = torch.load(cfg.model.model_checkpoint, map_location="cpu")
    state_dict = {
        key.replace("module.", ""): val for key, val in state_dict.items()
    }  # this handles the cause
    state_dict = {key.replace("_orig_mod.", ""): val for key, val in state_dict.items()}
    state_dict["patch_embed.pos_embed"] = state_dict["patch_embed.pos_embed"].unsqueeze(
        0
    )
    state_dict.pop("n_averaged", None)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    model = torch.compile(model)

    # Create policy network
    seq_len = (
        cfg.dataset.transforms.image_size // cfg.model.encoder_config.patch_size
    ) ** 2
    token_choices = list(cfg.model.token_grans)
    depth_choices = list(range(1, cfg.model.encoder_config.depth + 1))
    mat_dim_choices = list(cfg.model.mat_dims)
    policy = PolicyNetwork(
        dim=cfg.model.encoder_config.embed_dim,
        seq_len=seq_len,
        token_choices=token_choices,
        depth_choices=depth_choices,
        mat_dim_choices=mat_dim_choices,
    )
    policy = policy.to(device)
    if cfg.policy.init_weights:
        policy.load_state_dict(torch.load(cfg.policy.init_weights, map_location=device))
        print("Loaded policy weights from", cfg.policy.init_weights)

    # Create reward function
    reward_fn = RewardFunction(
        len(token_choices),
        len(depth_choices),
        len(mat_dim_choices),
        reward_scale=cfg.reward.reward_scale,
        lambda_cost=cfg.reward.lambda_cost,
        token_weight=cfg.reward.token_weight,
        depth_weight=cfg.reward.depth_weight,
        mat_dim_weight=cfg.reward.mat_dim_weight,
        reward_type=cfg.reward.reward_type,
    ).to(device)

    # Optimizer
    total_steps = len(train_loader) * cfg.optimizer.max_epochs
    optimizer_cfg = cfg.optimizer
    grad_clip = optimizer_cfg.grad_clip
    optim = create_optimizer_v2(
        policy.parameters(),
        opt=optimizer_cfg.optimizer,
        lr=optimizer_cfg.lr,
        weight_decay=optimizer_cfg.weight_decay,
        betas=optimizer_cfg.betas,
        maximize=True,
        corrected_weight_decay=True,
        caution=True,
    )
    scheduler, _ = create_scheduler_v2(
        optimizer=optim,
        sched=optimizer_cfg.schedule,
        num_epochs=total_steps,
        warmup_epochs=optimizer_cfg.warmup_steps,
        min_lr=optimizer_cfg.min_lr,
        warmup_lr=optimizer_cfg.warmup_lr,
    )
    # optim = torch.optim.AdamW(policy.parameters(), lr=1e-4, maximize=True)

    baseline_ema = torch.tensor(0.0, device=device)
    ema_decay = (
        cfg.reward.ema_decay
    )  # decay in [0.9, 0.999]; higher = smoother baseline

    alpha = cfg.reward.alpha  # entropy regularization coefficient

    init_temp = cfg.reward.init_temp
    final_temp = cfg.reward.final_temp
    temp_schedule = torch.linspace(init_temp, final_temp, total_steps + 1)

    lambda_cost_schedule = torch.linspace(1, 1, total_steps + 1)
    step = 0
    for epoch in range(cfg.optimizer.max_epochs):
        for i, (img, label) in enumerate(tqdm(train_loader)):
            img = img.to(device, non_blocking=True)  # [B, 3, H, W]
            label = label.to(device, non_blocking=True)  # [B,]
            with torch.inference_mode():
                x = model.patch_embed(img)
            logits_tokens, logits_depth, logits_mat_dim = policy(
                x.clone()
            )  # [B, n_choices]

            dist_tokens = torch.distributions.Categorical(
                logits=logits_tokens / temp_schedule[step]
            )

            entropy = dist_tokens.entropy()  # [B,]

            # Single forward pass
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    freq_cis = precompute_freqs_cis(
                        dim=model.embed_dim // model.num_heads,
                        end=model.mask_token_grans[-1],
                        train_len=model.max_latent_tokens,
                        theta=model.rope_theta,
                    )
                    preds, _, (_, _) = model.forward(
                        img,
                        num_tokens=None,
                        mat_dim=None,
                        depth=None,
                        token_loss=True,
                        layer_loss=False,
                        freq_cis=freq_cis,
                    )

            log_probs_tokens_all = []
            accuracy_list = []
            cost_list = []
            for k in range(cfg.reward.num_samples):
                sampled_token_idxs = dist_tokens.sample()  # [B,]
                log_probs_tokens = dist_tokens.log_prob(sampled_token_idxs)  # [B,]
                log_probs_tokens_all.append(log_probs_tokens)

                # REINFORCE: maximize expected reward
                # gradient: E[ reward * grad log pi(a|s) ]
                for j, token_idx in enumerate(sampled_token_idxs):
                    # we grab the prediction corresponding to the sampled token choice
                    pred = preds[token_idx][j : j + 1]  # [1, num_classes]
                    correct, cost = reward_fn(
                        predictions=pred,
                        targets=label[j : j + 1],
                        sampled_token_idxs=token_idx,
                        sampled_depth_idxs=None,
                        sampled_mat_dim_idxs=None,
                    )
                    accuracy_list.append(correct)
                    cost_list.append(cost)

            accuracy = torch.stack(accuracy_list)
            cost = torch.stack(cost_list)
            reward = (
                reward_fn.reward_scale * accuracy
                - reward_fn.lambda_cost * lambda_cost_schedule[step] * cost
            )

            if step == 0:
                baseline_ema = reward.mean().detach()
            else:
                baseline_ema = (
                    ema_decay * baseline_ema + (1 - ema_decay) * reward.mean().detach()
                )
            adv = reward - baseline_ema
            adv = adv.mean() / (adv.std() + 1e-8)

            log_p = torch.cat(
                log_probs_tokens_all, dim=0
            )  # + log_probs_depth + log_probs_mat_dim

            loss = adv * log_p + alpha * entropy.mean()
            loss = loss.mean()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optim.step()
            step += 1
            scheduler.step(epoch=step)

            wandb.log(
                {
                    "reward": reward.mean().item(),
                    "baseline_reward": baseline_ema.mean().item(),
                    "loss": loss.mean().item(),
                    "accuracy": accuracy.mean().item(),
                    "cost": cost.mean().item(),
                    "entropy": entropy.mean().item(),
                    "log_p": log_p.mean().item(),
                    "log_p_tokens": log_probs_tokens.mean().item(),
                    "temp": temp_schedule[i].item(),
                    "lambda_cost": lambda_cost_schedule[step].item(),
                    "lr": optim.param_groups[0]["lr"],
                }
            )

            if (i + 1) % 200 == 0:
                print(
                    "Step:",
                    i + 1,
                    "Reward:",
                    reward.mean().item(),
                    "Loss:",
                    loss.item(),
                )
                print(Counter(F.softmax(logits_tokens, dim=-1).argmax(dim=-1).tolist()))
                print(dist_tokens.probs)

        torch.save(policy.state_dict(), f"policy_final_{reward_fn.lambda_cost}.pth")

        policy.load_state_dict(
            torch.load(f"policy_final_{reward_fn.lambda_cost}.pth", map_location=device)
        )

        # Evaluate on validation set
        policy.eval()

        # Top-1 Accuracy
        top_1_accuracy = Accuracy(
            task="multiclass", num_classes=cfg.model.encoder_config.num_classes, top_k=1
        ).to(device)

        counter = Counter()
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=cfg.dataset.val_num_workers,
            pin_memory=cfg.dataset.pin_memory,
            collate_fn=collate_fn,
        )
        total = 0
        correct = 0
        times_ms = []
        times_policy_ms = []
        for img, label in tqdm(val_loader):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.inference_mode():
                t0_policy = time.perf_counter()
                x = model.patch_embed(img)
                logits_tokens, logits_depth, logits_mat_dim = policy(x.clone())

                sampled_token_idxs = F.softmax(logits_tokens, dim=-1).argmax(dim=-1)
                sampled_token = token_choices[sampled_token_idxs.item()]
                counter.update([sampled_token])

                freq_cis = precompute_freqs_cis(
                    dim=model.embed_dim // model.num_heads,
                    end=model.mask_token_grans[-1],
                    train_len=model.max_latent_tokens,
                    theta=model.rope_theta,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds, _, (_, _) = model.forward(
                    img,
                    num_tokens=sampled_token,
                    mat_dim=None,
                    depth=None,
                    freq_cis=freq_cis,
                )
                top_1_accuracy(preds[-1].squeeze(1), label)
                logits = preds[-1].squeeze(1)
                preds = logits.argmax(dim=-1)

                if device == "cuda":
                    torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
                times_policy_ms.append((time.perf_counter() - t0_policy) * 1000.0)
                # Accuracy Calculation
                total += label.size(0)
                correct += (preds == label).sum().item()
        accuracy = correct / total
        times_ms = np.array(times_ms)
        mean_ms = times_ms.mean()
        std_ms = times_ms.std()
        times_policy_ms = np.array(times_policy_ms)
        mean_policy_ms = times_policy_ms.mean()
        std_policy_ms = times_policy_ms.std()
        wandb.log({"val_accuracy": accuracy * 100})
        wandb.log({"val_inference_time": mean_ms})
        wandb.log({"val_inference_time_std": std_ms})
        wandb.log({"val_policy_time": mean_policy_ms})
        wandb.log({"val_policy_time_std": std_policy_ms})
        wandb.log({"val_accuracy_metric": top_1_accuracy.compute().item() * 100})
        print(counter.most_common(5))
        print(f"Validation accuracy over 5 repeats: {accuracy * 100:.2f}%")
        print(
            f"Validation inference time (mean ± std): {mean_ms:.2f} ms ± {std_ms:.2f} ms"
        )


if __name__ == "__main__":
    main()

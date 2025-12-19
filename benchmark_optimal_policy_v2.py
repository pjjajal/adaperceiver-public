import json
import time
from dataclasses import dataclass
from collections import Counter

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from modules.layers.rope import precompute_freqs_cis
from dataset import create_dataset
from modules.networks.classification import ClassificationAdaPerceiver
from modules.networks.factory import create_model
import tarfile
from tqdm import tqdm
from reinforce_rl import PolicyNetwork
from nutils.benchmark import measure_flops


torch.set_float32_matmul_precision("medium")


# ====================== Load optimal configs (JSON / .tar.gz) ======================
def load_optimal_configs(path):
    if path.endswith(".tar.gz") or path.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tar:
            member = tar.getmembers()[0]
            if member.isfile() and member.name.endswith(".json"):
                f = tar.extractfile(member)
                obj = json.load(f)  # directly load JSON
                return obj
    elif path.endswith(".json"):
        with open(path, "r") as f:
            obj = json.load(f)
            return obj
    return None


# ====================== Dataset wrappers ======================
class IndexedDataset(torch.utils.data.Dataset):
    """Wraps an existing dataset to return (idx, x, y)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        return idx, data


def wrap_collate_fn(image_collate_fn):
    """Wrap an existing collate_fn to also return sample indices."""

    def collate_fn(batch):
        idxs, data = zip(*batch)
        batch_data = image_collate_fn(data)
        return list(idxs), batch_data

    return collate_fn


# ====================== Config helpers (lexicographic) ======================
@dataclass(frozen=True, order=True)
class MiniCfg:
    depth: int
    num_tokens: int
    mat_dim: int
    depth_tau: float = None  # optional, not used for ordering


def _lex_is_a_ge_b(a: MiniCfg, b: MiniCfg) -> bool:
    # Higher num_tokens first, then mat_dim, then depth
    if a.num_tokens != b.num_tokens:
        return a.num_tokens > b.num_tokens
    if a.mat_dim != b.mat_dim:
        return a.mat_dim > b.mat_dim
    return a.depth >= b.depth


def cfg_lex_max(a: MiniCfg, b: MiniCfg) -> MiniCfg:
    return a if _lex_is_a_ge_b(a, b) else b


def max_cfg(a: MiniCfg, b: MiniCfg) -> MiniCfg:
    return MiniCfg(
        mat_dim=max(a.mat_dim, b.mat_dim),
        num_tokens=max(a.num_tokens, b.num_tokens),
        depth=max(a.depth, b.depth),
    )


def lookup_policy(policy: dict, idx):
    """
    policy: dict[str idx] -> {"correct": bool, "optimal config": {"mat_dim","token_gran","depth",...}}
    Returns MiniCfg or None (if not correct / missing).
    """
    str_idx = str(idx)
    idx_policy = policy.get(str_idx, None)
    if idx_policy is None:
        return None
    if not idx_policy.get("correct", False):
        return None
    config = idx_policy["optimal config"]
    return MiniCfg(
        mat_dim=config["mat_dim"],
        num_tokens=config["token_gran"],
        depth=config["depth"],
    )


# ====================== Evaluation ======================
def eval_policy(cfg, model: ClassificationAdaPerceiver, val_loader, device, policy_obj):
    """
    policy_obj can be:
      - MiniCfg (baseline)
      - dict (json policy from load_optimal_configs)
      - PolicyNetwork (RL-based)
    """
    total_correct = 0
    total_seen = 0
    times_ms = []

    top_1_accuracy = Accuracy(
        task="multiclass", num_classes=cfg.model.encoder_config.num_classes, top_k=1
    ).to(device)

    # Determine mode
    is_baseline = (
        isinstance(policy_obj, MiniCfg) and getattr(policy_obj, "depth_tau") is None
    )
    is_sim_early_exit = (
        isinstance(policy_obj, MiniCfg)
        and getattr(policy_obj, "depth_tau", None) is not None
    )
    is_json = isinstance(policy_obj, dict)

    if hasattr(policy_obj, "_orig_mod"):
        is_rl = isinstance(policy_obj._orig_mod, PolicyNetwork)
    else:
        is_rl = isinstance(policy_obj, PolicyNetwork)

    if is_baseline:
        fixed_cfg = policy_obj
    elif is_sim_early_exit:
        fixed_cfg = policy_obj

    batch_cfgs = []
    flops_dict = {}
    for ds_idxs, (imgs, labels) in tqdm(val_loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- Choose batch config based on branch ---
        with torch.inference_mode():
            depth_counter = None
            if is_baseline:
                batch_cfg = fixed_cfg
            elif is_sim_early_exit:
                batch_cfg = fixed_cfg
                depth_counter = {
                    i: 0 for i in range(1, cfg.model.encoder_config.depth + 1)
                }

            elif is_json:
                # Per-sample configs from the loaded JSON policy
                sample_cfgs = [lookup_policy(policy_obj, idx) for idx in ds_idxs]
                sample_cfgs = [c for c in sample_cfgs if c is not None]
                batch_cfg = (
                    max(sample_cfgs)
                    if len(sample_cfgs) > 0
                    else MiniCfg(
                        mat_dim=cfg.model.mat_dims[-1],
                        num_tokens=cfg.model.token_grans[-1],
                        depth=cfg.model.encoder_config.depth,
                    )
                )
            elif is_rl:
                if cfg.policy.rl.get("depth_tau", None):
                    depth_counter = {
                        i: 0 for i in range(1, cfg.model.encoder_config.depth + 1)
                    }
                feats = model.patch_embed(imgs)
                logits_tokens, logits_depth, logits_matdim = policy_obj(feats.clone())
                token_idx = logits_tokens.argmax(dim=-1)
                depth_idx = logits_depth.argmax(dim=-1)
                matdim_idx = logits_matdim.argmax(dim=-1)
                sampled_tokens = [policy_obj.token_choices[t.item()] for t in token_idx]
                # sampled_depths = [policy_obj.depth_choices[d.item()] for d in depth_idx]
                # sampled_matdims = [
                #     policy_obj.mat_dim_choices[m.item()] for m in matdim_idx
                # ]
                sample_cfgs = [
                    MiniCfg(
                        num_tokens=token,
                        mat_dim=cfg.model.mat_dims[-1],
                        depth=cfg.model.encoder_config.depth,
                        depth_tau=cfg.policy.rl.get("depth_tau", None),
                    )
                    for token in sampled_tokens
                ]
                batch_cfg = max(sample_cfgs)
            else:
                raise ValueError("Unknown policy type passed to eval_policy.")

            # --- Timed forward pass with chosen batch_cfg ---
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            freq_cis = precompute_freqs_cis(
                dim=model.embed_dim // model.num_heads,
                end=model.mask_token_grans[-1],
                train_len=model.max_latent_tokens,
                theta=model.rope_theta,
            )
            output, _, _ = model(
                imgs,
                num_tokens=batch_cfg.num_tokens,
                mat_dim=batch_cfg.mat_dim,
                depth=batch_cfg.depth,
                freq_cis=freq_cis,
                depth_tau=batch_cfg.depth_tau,
                depth_counter=depth_counter,
            )
            if is_sim_early_exit or batch_cfg.depth_tau is not None:
                current_depth = depth_counter.get("current_depth", batch_cfg.depth)
                batch_cfg = MiniCfg(
                    num_tokens=batch_cfg.num_tokens,
                    mat_dim=batch_cfg.mat_dim,
                    depth=current_depth,
                )
            logits = output[-1].squeeze(1)

            if device == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

            # --- Accuracy ---
            top_1_accuracy(logits, labels)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.numel()

            input_shape = imgs.shape

            flops_input_shape = (
                1,
                *input_shape[1:],
            )  # Use batch size of 1 for FLOPs measurement
            if cfg.model.encoder_config.attn_layer == "full":
                if batch_cfg not in flops_dict:
                    with torch.inference_mode(False):
                        flops = measure_flops(
                            model,
                            flops_input_shape,
                            device,
                            num_tokens=batch_cfg.num_tokens,
                            mat_dim=batch_cfg.mat_dim,
                            depth=batch_cfg.depth,
                            freq_cis=freq_cis,
                        )
                    flops_dict[batch_cfg] = flops["forward_total"]
        batch_cfgs.append(batch_cfg)

    # --- Metrics ---
    acc = 100.0 * total_correct / max(1, total_seen)
    times_ms = np.array(times_ms)
    mean_ms, std_ms = times_ms.mean(), times_ms.std()
    cfg_counter = Counter(batch_cfgs)
    total_flops = 0.0
    if cfg.model.encoder_config.attn_layer == "full":
        total_flops = sum(count * flops_dict[cfg] for cfg, count in cfg_counter.items())

    total_batches = len(val_loader)
    print(
        f"Accuracy: {acc:.2f}%  |  Mean batch time: {mean_ms:.1f} ms | Std: {std_ms:.2f} | Examples: {total_seen} | Total FLOPs: {total_flops / 1e9 / total_batches:.2f} GFLOPs"
    )
    print("Accuracy metric from torchmetrics:", top_1_accuracy.compute().item() * 100.0)
    print("Most common config:", cfg_counter.most_common(5))
    return {
        "accuracy_top1": acc,
        "mean_batch_time_ms": mean_ms,
        "std_batch_time_ms": std_ms,
        "num_examples": int(total_seen),
        "total_flops_per_batch": total_flops / 1e9 / total_batches,
    }


# Small adapter to keep tqdm from re-wrapping an existing loader
def DataLoaderWrapper(loader):
    for x in loader:
        yield x


# ====================== Entry point ======================
@hydra.main(version_base=None, config_path="config/evaluation/policy/optimal")
def main(cfg: DictConfig):
    # Seed & device
    L.seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model ---
    if cfg.model.model_checkpoint:
        print(f"Loading from checkpoint: {cfg.model.model_checkpoint}")
        model = create_model(cfg.model)
        state_dict = torch.load(cfg.model.model_checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        state_dict["patch_embed.pos_embed"] = state_dict[
            "patch_embed.pos_embed"
        ].unsqueeze(0)
        state_dict.pop("n_averaged", None)
        state_dict.pop("ema_n_averaged", None)
        model.load_state_dict(state_dict, strict=True)
    else:
        model = create_model(cfg.model)

    if cfg.model.compile:
        model.compile()

    model = model.to(device).eval()

    # --- Data ---
    train_dataset, val_dataset, collate_fn = create_dataset(
        cfg.dataset, cfg.dataset.num_proc, eval_only=True
    )
    del train_dataset  # not needed
    assert val_dataset is not None, "Validation dataset is required."
    val_dataset = IndexedDataset(val_dataset)
    collate_fn = wrap_collate_fn(collate_fn)

    # Build loader; we’ll recreate it each repeat for shuffling with a seed
    results = []

    # --- Policy selection branch ---
    policy_obj = None
    if getattr(cfg.policy.baseline, "enable", False):
        policy_obj = MiniCfg(
            mat_dim=cfg.policy.baseline.mat_dim,
            num_tokens=cfg.policy.baseline.tokens,
            depth=cfg.policy.baseline.depth,
        )
        print(f"[Policy] Baseline fixed config: {policy_obj}")
    elif getattr(cfg.policy.sim_early_exit, "enable", False):
        policy_obj = MiniCfg(
            mat_dim=cfg.policy.sim_early_exit.mat_dim,
            num_tokens=cfg.policy.sim_early_exit.tokens,
            depth=model.depth,
            depth_tau=cfg.policy.sim_early_exit.depth_tau,
        )
        print(f"[Policy] Simulated Early Exit config: {policy_obj}")
    elif getattr(cfg.policy.json, "enable", False):
        json_path = cfg.policy.json.path
        print(f"[Policy] Loading JSON policy from: {json_path}")
        policy_obj = load_optimal_configs(json_path)
        if policy_obj is None:
            raise FileNotFoundError(f"Could not load JSON policy from {json_path}")

    elif getattr(cfg.policy.rl, "enable", False):
        path = cfg.policy.rl.path
        print(f"[Policy] Loading RL policy from: {path}")
        token_choices = list(cfg.model.token_grans)
        depth_choices = list(range(1, cfg.model.encoder_config.depth + 1))
        mat_dim_choices = list(cfg.model.mat_dims)
        seq_len = (
            cfg.dataset.transforms.image_size // cfg.model.encoder_config.patch_size
        ) ** 2
        policy_obj = PolicyNetwork(
            dim=cfg.model.encoder_config.embed_dim,
            seq_len=seq_len,
            token_choices=token_choices,
            depth_choices=depth_choices,
            mat_dim_choices=mat_dim_choices,
        )
        policy_obj = policy_obj.to(device).eval()
        state_dict = torch.load(path)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy_obj.load_state_dict(state_dict, strict=True)
        print(f"[Policy] Loaded RL policy network.")
        policy_obj = torch.compile(policy_obj)
    else:
        raise ValueError(
            "No policy branch enabled. Set one of:\n"
            "  - policy.baseline.enable = true\n"
            "  - policy.sim_early_exit.enable = true\n"
            "  - policy.json.enable = true\n"
            "  - policy.rl.enable = true"
        )

    # --- Eval with repeats (reshuffle per repeat) ---
    for r in range(cfg.policy.repeats):
        g = torch.Generator()
        g.manual_seed(cfg.seed + r)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.dataset.val_batch_size,
            num_workers=cfg.dataset.val_num_workers,
            pin_memory=cfg.dataset.pin_memory,
            collate_fn=collate_fn,
            shuffle=True,
            generator=g,
        )
        eval_results = eval_policy(cfg, model, val_loader, device, policy_obj)
        del val_loader
        results.append(eval_results)

    # --- Summarize repeats ---
    if len(results) > 1:
        print("Summary of results across repeats:")
        for k in results[0].keys():
            vals = [r[k] for r in results[1:]]
            print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()

# AdaPerceiver: Transformers with Adaptive Width, Depth, and Tokens


[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/collections/pjajal/adaperceiver-v1)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18105-b31b1b)](https://arxiv.org/abs/2511.18105)

This repository contains the **official PyTorch implementation of the AdaPerceiver model**, a transformer architecture designed for **adaptivity across tokens, depth, and width within a single network**.

**AdaPerceiver** extends the Perceiver family of models with *runtime-configurable computation*. At inference time, a single trained model can trade off **accuracy and FLOPs** by adjusting:
- the **number of latent tokens**,
- the **effective depth**, and
- the **embedding dimension**.


**Links**
- 📄 Paper: https://arxiv.org/abs/2511.18105  
- 📦 HuggingFace Models: https://huggingface.co/collections/pjajal/adaperceiver-v1


## Environment Setup
Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```


## HuggingFace Models
The weights for the feature distilled and the IN-1K finetuned models are available on HuggingFace.
These models can be loaded using the model classes provided in `hub/`.

*Note*: All HuggingFace models support **runtime configuration** of tokens, width, and depth and are intended for **inference, evaluation, and downstream use**; training scripts in this repository use the native module implementations directly.

### Available Hub Modules

- **`hub/adaperceiver.py`**  
  Base AdaPerceiver backbone implementing adaptive tokens, depth, and width. (This is merely a parent class for the other modules!)

- **`hub/adaperceiver_classification.py`**  
  AdaPerceiver model for image classification, with support for early exit and adaptive inference-time configuration.

- **`hub/networks/adaperceiver_distill.py`**  
  Distillation variant that exposes both **logits** and **intermediate features**, used for logit and feature distillation as well as downstream dense prediction tasks.

### Logit + Feature Distilled Model (ImageNet-12K)

This corresponds to the **feature + logit distilled AdaPerceiver model** described in the paper (Appendix D.1).

**Weights:** https://huggingface.co/pjajal/adaperceiver-v1

```python
import torch
from hub.networks.adaperceiver_distill import DistillAdaPerceiver

model = DistillAdaPerceiver.from_pretrained("pjajal/adaperceiver-v1")

# forward(
#   x: input image tensor (B, C, H, W)
#   num_tokens: number of latent tokens to process (optional)
#   mat_dim: embedding dimension (optional)
#   depth: early-exit depth (optional)
#   token_grans: block-mask granularities (optional)
# )
out = model(
    torch.randn(1, 3, 224, 224),
    num_tokens=256,
    mat_dim=128,
    depth=12,
)

print(out.logits.shape, out.features.shape)
```

### ImageNet-1K Classification Model

This is the ImageNet-1K fine-tuned AdaPerceiver model described in Appendix D.2 of the paper.

**Weights:** https://huggingface.co/pjajal/adaperceiver-v1-in1k-ft

```python
import torch
from hub.networks.adaperceiver_classification import ClassificationAdaPerceiver

model = ClassificationAdaPerceiver.from_pretrained(
    "pjajal/adaperceiver-v1-in1k-ft"
)

# forward(
#   x: input image tensor (B, C, H, W)
#   num_tokens: number of latent tokens to process (optional)
#   mat_dim: embedding dimension (optional)
#   depth: early-exit depth (optional)
#   depth_tau: confidence threshold for early exit (optional)
#   token_grans: block-mask granularities (optional)
# )
out = model(
    torch.randn(1, 3, 224, 224),
    num_tokens=256,
    mat_dim=192,
    depth=12,
)

print(out.logits.shape)
```

## Quickstart

Below are the most common entry points for training, fine-tuning, and evaluating AdaPerceiver models. All commands use **Hydra configs**; use `-cn <config_name>` to select a configuration and override fields inline as needed.

### Training

#### Distillation (ImageNet-12K)

```bash
python adaperceiver_distill_hydra.py -cn vit_huge_soadaperciever_12k
```

#### Image Classification Fine-Tuning (ImageNet-1K)

```bash
python adaperceiver_ft_hydra.py -cn soadaperciever_1k
```

#### Depth Estimation (NYUv2)

```bash
python adaperceiver_depth.py -cn vit_huge_soadaperciever_12k
```

#### Semantic Segmentation (ADE20K)

```bash
python adaperceiver_segmentation.py -cn vit_huge_soadaperciever_12k
```

### Evaluation

#### Baseline Models

```bash
# Classfication Evals
python baseline_eval.py -cn <config-name>

# Segmentation Evaluation
python baseline_eval_seg.py -cn <config-name>

# Depth Evaluation
python baseline_eval_depth.py -cn <config-name>
```

#### AdaPerceiver Evals
```bash
# Classfication Evals
python adaperceiver_eval.py -cn soadaperciever_1k

# Segmentation Evaluation
python adaperceiver_eval_seg.py -cn vit_huge_soadaperciever_12k

# Depth Evaluation
python adaperceiver_eval_depth.py -cn vit_huge_soadaperciever_12k

# Policy Evals
python benchmark_optimal_policy_v2.py -cn soadaperciever_1k_1
```


## Repository Layout
- `config/`: Hydra configs grouped by task.
  - `depth/`, `segmentation/`: dense prediction configs (e.g., NYU depth, ADE20K) with model sizes, token budgets, and loss weights.
  - `supervised/`: classification fine-tuning configs (ImageNet-1k/12k).
  - `distill/`: teacher-student distillation configs.
  - `evaluation/`: eval and baseline benchmarking configs.
  - `rl_reinforce/`: configs for reinforcement learning training of policy network.
- `dataset/`: dataset wrappers and transforms for ImageNet, ImageNet-12k, ADE20K scene parsing, and NYU Depth; includes collate functions and augmentation pipelines.
- `modules/`: Model and layer definitions.
  - `layers/`: attention, adapters, RoPE, masking, feedforward blocks, etc.
  - `networks/`: AdaPerceiver backbones for classification and dense prediction, plus distillation variants and factory helpers.
  - `depth/` and `segmentation/`: models, heads, and loss utilities for dense prediction.
  - `helpers/`: common losses and weighting utilities shared by tasks.
- `utils/`: helper functions (e.g., weight surgery, shared training helpers).
- Training scripts:
  - `adaperceiver_distill_hydra.py`: distillation script.
  - `adaperceiver_ft_hydra.py`: fine-tuning script.
  - `adaperceiver_depth.py`: depth estimation training script.
  - `adaperceiver_segmentation.py`: semantic segmentation training.
  - `reinforce_rl.py`: REINFORCE training script.
  - `baselines_depth.py`, `baselines_segmentation.py`: baseline training scripts for dense prediction.
- Evaluation and analysis:
  - `adaperceiver_eval.py`: sweeps AdaPerceiver classification models over token/depth settings and reports accuracy/FLOPs/latency.
  - `adaperceiver_eval_depth.py`, `adaperceiver_eval_seg.py`: validation loops for depth and segmentation checkpoints.
  - `baselines_eval.py`, `baselines_eval_depth.py`, `baselines_eval_seg.py`: benchmarking helper scripts for baseline models.
  - `benchmark_optimal_policy_v2.py`: measures FLOPs/accuracy for the various policies.
- `requirements.txt`: Python dependencies.

## Reference

If you use this repository, models, or training methodology in your research, please cite the AdaPerceiver paper:

```bibtex
@article{jajal2025adaperceiver,
  title={AdaPerceiver: Transformers with Adaptive Width, Depth, and Tokens},
  author={Jajal, Purvish and Eliopoulos, Nick John and Chou, Benjamin Shiue-Hal and Thiruvathukal, George K and Lu, Yung-Hsiang and Davis, James C},
  journal={arXiv preprint arXiv:2511.18105},
  year={2025}
}
```

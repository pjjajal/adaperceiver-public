AdaPerceiver
============
PyTorch/Lightning implementation of adaptive Perceiver models for image classification, depth estimation, and semantic segmentation. Training and evaluation are driven by Hydra configs so you can swap datasets, model sizes, and optimization settings from the command line.

Setup
-----
- Create a virtual environment (optional) and install dependencies with `pip install -r requirements.txt`.
- All entrypoints use Hydra. Pass `-cn <file>` to pick a config from the matching `config/<task>` folder and override fields inline (e.g., `trainer.devices=2 dataset.data_path=/path/to/imagenet`).
- All datasets make use of HuggingFace Datasets. The `data_path` field can be used to point the the location of the `HF_CACHE`.

Quickstart
----------
- Distillation: `python adaperceiver_distill_hydra.py -cn vit_huge_soadaperciever_12k`
- Classification fine-tuning: `python adaperceiver_ft_hydra.py -cn soadaperciever_1k`
- Depth estimation: `python adaperceiver_depth.py -cn vit_huge_soadaperciever_12k`
- Semantic segmentation: `python adaperceiver_segmentation.py -cn vit_huge_soadaperciever_12k`
- Adaptive eval sweep for classification: `python adaperceiver_eval.py -cn soadaperceiver_1k`
- Baseline eval (timm/backbone baselines): `python baselines_eval.py -cn baseline model.name=<timm_model>`
- Depth/segmentation baselines or eval: use `baselines_depth.py`, `baselines_segmentation.py`, `baselines_eval_depth.py`, and `baselines_eval_seg.py` with the matching configs.

Repository Layout
-----------------
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

Tips
----
- Hydra creates output directories per run; set `trainer.checkpoint_save_dir` if you want checkpoints in a fixed path.
- To compile models with `torch.compile`, leave `model.compile=true` (supported in most configs).

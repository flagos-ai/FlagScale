# DreamZero: Training

This guide covers how to train DreamZero 14B using FlagScale with FSDP2 distributed training.

## Overview

DreamZero is a World Action Model — a Wan2.1 14B video DiT repurposed as a zero-shot robot policy. Architecture:
- **T5-XXL** text encoder (~4.7B, frozen)
- **CLIP** image encoder (~0.6B, frozen)
- **VAE** video encoder (~0.1B, frozen)
- **CausalWan DiT** video diffusion transformer (~14B, trainable)
- Action/state encoder-decoder projectors (trainable)

FlagScale uses FSDP2 (ZeRO-3 style sharding) to distribute the 14B trainable DiT across GPUs.

## Installation

### Clone Repository

```sh
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

### Setup Environment

Create a conda environment with PyTorch 2.4+ and FSDP2 support:

```sh
conda create -n flagscale-dreamzero python=3.11
conda activate flagscale-dreamzero
pip install ".[cuda-train]" --verbose
```

Install additional dependencies:

```sh
pip install safetensors peft decord
```

DreamZero also requires the reference `groot` package for model architecture and data transforms:

```sh
git clone <dreamzero-reference-repo> /workspace/dreamzero
```

Add both FlagScale and the reference repo to your PYTHONPATH:

```sh
export PYTHONPATH=/path/to/FlagScale:/workspace/dreamzero:$PYTHONPATH
```

## Download Models

Download the DreamZero-AgiBot checkpoint (contains `config.json` + safetensors shards):

```sh
# Place at /workspace/models/DreamZero-AgiBot/
# Expected contents:
#   config.json
#   model.safetensors.index.json
#   model-00001-of-00010.safetensors
#   ...
#   model-00010-of-00010.safetensors
```

Download the Wan2.1-I2V-14B-480P pretrained weights (for tokenizer, and optionally T5/CLIP/VAE):

```sh
# Place at /workspace/models/Wan2.1-I2V-14B-480P/
# Required for tokenizer:
#   google/umt5-xxl/tokenizer.json
#   google/umt5-xxl/spiece.model
```

## Training

### Prepare Dataset

DreamZero uses a LeRobot-format dataset with `data/`, `meta/`, and `videos/` subdirectories.

```sh
# Place at /workspace/datasets/vla_arena_dreamzero/
# Expected structure:
#   data/chunk-000/, data/chunk-001/, ...
#   meta/info.json, meta/episodes.jsonl, ...
#   videos/chunk-000/, ...
```

### Edit Config

FlagScale uses a two-level configuration system:

1. **Experiment-level config** (`examples/dreamzero/conf/train.yaml`): Experiment settings, environment variables, and resource allocation
2. **Task-level config** (`examples/dreamzero/conf/train/dreamzero_14b.yaml`): Model, dataset, and training hyperparameters

#### Experiment-Level Config

```sh
vim examples/dreamzero/conf/train.yaml
```

Configure the following fields:

- `experiment.exp_name` — Experiment name
- `experiment.exp_dir` — Output directory for checkpoints and logs
- `experiment.envs.CUDA_VISIBLE_DEVICES` — GPU devices to use (e.g., `"0,1,2,3,4,5,6,7"`)
- `experiment.runner.nproc_per_node` — Number of GPUs

#### Task-Level Config

```sh
vim examples/dreamzero/conf/train/dreamzero_14b.yaml
```

Configure the following fields:

**System settings:**
- `system.batch_size` — Per-GPU micro batch size (default: `1`)
- `system.train_steps` — Total training steps (default: `5000`)
- `system.checkpoint.save_freq` — Steps between checkpoints

**Model settings:**
- `model.pretrained_model_path` — Path to DreamZero-AgiBot checkpoint (e.g., `/workspace/models/DreamZero-AgiBot`)
- `model.train_architecture` — `"full"` for full fine-tuning (recommended), `"lora"` for LoRA on DiT
- `model.optimizer.lr` — Learning rate (default: `1.0e-5`)
- `model.optimizer.betas` — Adam betas (default: `[0.95, 0.999]`)
- `model.optimizer.scheduler.warmup_ratio` — Warmup ratio (default: `0.05`)

**Data settings:**
- `data.data_path` — Path to LeRobot dataset (e.g., `/workspace/datasets/vla_arena_dreamzero`)
- `data.tokenizer_path` — Path to UMT5-XXL tokenizer (e.g., `/workspace/models/Wan2.1-I2V-14B-480P/google/umt5-xxl`)
- `data.embodiment_tag` — Embodiment tag (e.g., `"libero"`)

### Start Training

```sh
cd FlagScale/
python flagscale/run.py --config-path examples/dreamzero/conf --config-name train
```

Training logs are saved to `outputs/dreamzero_train/logs/host_0_localhost.output` by default.

### Stop Training

```sh
cd FlagScale/
python flagscale/run.py --config-path examples/dreamzero/conf --config-name train action=stop
```

## Hardware Requirements

With 8x H100/H800 80GB GPUs and `batch_size=1`:
- Full fine-tuning (FSDP2): ~65 GB per GPU
- Model loading takes ~30 minutes (14B params from 10 safetensors shards)
- Training throughput: ~8.7s per step (effective batch size 8)

## Known Issues

1. **torch.compile + FSDP2**: Compiling attention sub-methods with `mode="reduce-overhead"` causes CUDA graph tensor deallocation mismatches during backward. Disabled by default via `system.disable_attention_compile: true`.
2. **LoRA mode NaN**: `train_architecture: lora` can produce NaN due to bf16 backward overflow at certain timestep/data combinations. Use `train_architecture: full` to avoid this.
3. **batch_size=2 OOM**: Full fine-tuning with `batch_size=2` exceeds 80GB GPU memory. Use `batch_size=1` with gradient accumulation if larger effective batches are needed.

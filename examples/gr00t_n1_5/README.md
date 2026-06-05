# GR00T N1.5: Training and Serving

This guide covers how to train and serve GR00T N1.5 models using FlagScale.

## Installation

### Clone Repository

```sh
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

### Setup Conda Environment

Create a new conda environment for robotics training:

```sh
conda create -n flagscale-robo python=3.12
conda activate flagscale-robo
```

Install FlagScale and robotics dependencies:

```sh
cd FlagScale/
# replace "[cuda-train]" with "[ascend-train]" on Huawei Ascend, or "[musa-train]" on Moore Threads MUSA
pip install ".[cuda-train]" --verbose
```

Install additional dependencies for downloading models/datasets:

```sh
# For HuggingFace Hub
pip install huggingface_hub

# For ModelScope (optional)
pip install modelscope
```

## Download Models

Download the pretrained GR00T N1.5 model using the provided script. Choose either HuggingFace Hub or ModelScope:

**Using HuggingFace Hub:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id nvidia/GR00T-N1.5-3B \
    --output_dir /workspace/models \
    --source huggingface
```

**Using ModelScope:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id nvidia/GR00T-N1.5-3B \
    --output_dir /workspace/models \
    --source modelscope
```

The model will be downloaded to (example with `/workspace/models`):
- `/workspace/models/nvidia/GR00T-N1.5-3B`

## Training

### Prepare Dataset

FlagScale uses the **LeRobotDataset v3.0** format. For detailed information about the format structure, see the [LeRobotDataset v3.0 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3).

For example, to download the `libero_goal_no_noops` dataset:

**Using HuggingFace Hub:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
    --output_dir /workspace/datasets \
    --repo_type dataset \
    --source huggingface
```

**Using ModelScope:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
    --output_dir /workspace/datasets \
    --repo_type dataset \
    --source modelscope
```

The dataset will be downloaded to (example with `/workspace/datasets`):
- `/workspace/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot`

### Edit Config

FlagScale uses a two-level configuration system:

1. **Experiment-level config** (`examples/gr00t_n1_5/conf/train.yaml`): Defines experiment settings, environment variables, and resource allocation
2. **Task-level config** (`examples/gr00t_n1_5/conf/train/gr00t_n1_5.yaml`): Defines model, dataset, and training hyperparameters

#### Experiment-Level Config

Edit the experiment-level config for multi-GPU training:

```sh
cd FlagScale/
vim examples/gr00t_n1_5/conf/train.yaml
```

Configure the following fields:

- `experiment.envs.CUDA_VISIBLE_DEVICES` - GPU devices to use (e.g., `"0,1,2,3"` for 4 GPUs, `"0,1"` for 2 GPUs), Use `ASCEND_RT_VISIBLE_DEVICES` for Huawei Ascend, `MUSA_VISIBLE_DEVICES` for Moore Threads MUSA
- `experiment.envs.CUDA_DEVICE_MAX_CONNECTIONS` - Connection limit (typically `1`), Use `MUSA_DEVICE_MAX_CONNECTIONS` for Moore Threads MUSA
- `experiment.exp_name` - Experiment name
- `experiment.exp_dir` - Output directory for checkpoints and logs
- `experiment.runner.nproc_per_node` - Number of processes per node for multi-GPU training (required for Huawei Ascend)

#### Task-Level Config

Edit the task-level config for model and training settings:

```sh
cd FlagScale/
vim examples/gr00t_n1_5/conf/train/gr00t_n1_5.yaml
```

Configure the following fields:

**System settings** (training hyperparameters):
- `system.batch_size` - Batch size per GPU
- `system.train_steps` - Total training steps
- `system.checkpoint.save_checkpoint` - Whether to save checkpoints (default: `true`)
- `system.checkpoint.save_freq` - Steps between checkpoints (default: `1000`)
- `system.checkpoint.output_directory` - Checkpoint output directory (default: `${experiment.exp_dir}`)

**Model settings**:
- `model.model_name` - Model name: `"gr00t_n1_5"`
- `model.checkpoint_dir` - Path to pretrained model (e.g., `/workspace/models/nvidia/GR00T-N1.5-3B`)
- `model.embodiment_tag` - Embodiment tag for training (e.g., `"new_embodiment"`, `"gr1"`)
- `model.tune_llm` - Whether to fine-tune the language model (default: `true`)
- `model.tune_visual` - Whether to fine-tune the visual encoder (default: `true`)
- `model.tune_projector` - Whether to fine-tune the projector (default: `true`)
- `model.tune_diffusion_model` - Whether to fine-tune the diffusion action head (default: `true`)
- `model.compute_dtype` - Compute dtype (default: `bfloat16`)
- `model.optimizer.name` - Optimizer name (e.g., `"AdamW"`)
- `model.optimizer.lr` - Learning rate (e.g., `1.0e-4`)
- `model.optimizer.betas` - Optimizer betas (e.g., `[0.95, 0.999]`)
- `model.optimizer.eps` - Optimizer epsilon (e.g., `1.0e-8`)
- `model.optimizer.weight_decay` - Weight decay (e.g., `1.0e-5`)
- `model.optimizer.scheduler.warmup_steps` - Warmup steps (e.g., `500`)
- `model.optimizer.scheduler.decay_steps` - Decay steps (e.g., `10000`)
- `model.optimizer.scheduler.decay_lr` - Final learning rate after decay (e.g., `1.0e-5`)

**Data settings**:
- `data.data_path` - Path to LeRobot dataset root (e.g., `/workspace/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot`)
- `data.action_delta_indices` - List of action horizon indices to predict (default: `[0..15]` for 16-step horizon)
- `data.preprocessor.steps[groot_pack_inputs].config.embodiment_tag` - Must match `model.embodiment_tag`
- `data.preprocessor.steps[rename_observations_processor].config.rename_map` - Dictionary mapping dataset camera keys to policy keys (optional). Check the `features` key in your dataset's `meta/info.json` to determine the correct mapping:
  ```yaml
  rename_map:
    observation.images.cam_high: observation.images.top
    observation.images.cam_wrist: observation.images.wrist
  ```

### Start Training

```sh
cd FlagScale/
flagscale train gr00t_n1_5 --config ./examples/gr00t_n1_5/conf/train.yaml
# or
flagscale train gr00t_n1_5 -c ./examples/gr00t_n1_5/conf/train.yaml
```

Training logs are saved to `outputs/gr00t_n1_5_train/logs/host_0_localhost.output` by default.

Checkpoints are saved to `${experiment.exp_dir}/checkpoints` (default: `outputs/gr00t_n1_5_train/checkpoints`).

### Stop Training

```sh
cd FlagScale/
flagscale train gr00t_n1_5 --stop
```

## Serving

### Edit Config

```sh
cd FlagScale/
vim examples/gr00t_n1_5/conf/serve/gr00t_n1_5.yaml
```

Configure the following fields:

**Engine arguments:**
- `engine_args.host` - Server host (default: `"0.0.0.0"`)
- `engine_args.port` - Server port (default: `5000`)
- `engine_args.model_variant` - Model variant: `"Gr00tN15"`
- `engine_args.model` - Path to pretrained or fine-tuned model checkpoint (e.g., `/workspace/models/nvidia/GR00T-N1.5-3B`)
- `engine_args.device` - Device to use (e.g., `"cuda"`, `"npu"`, `"musa"`)

### Run Serving

```sh
cd FlagScale/
flagscale serve gr00t_n1_5 --config ./examples/gr00t_n1_5/conf/serve.yaml
# or
flagscale serve gr00t_n1_5 -c ./examples/gr00t_n1_5/conf/serve.yaml
```

Serving logs are saved to `outputs/gr00t_n1_5_serve/logs/host_0_localhost.output` by default.

### Stop Serving

```sh
cd FlagScale/
flagscale serve gr00t_n1_5 --stop
```

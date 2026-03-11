# Qwen-GR00T: Training, Inference, and Serving

This guide covers how to train, run inference, and serve Qwen-GR00T models using FlagScale. Qwen-GR00T uses a Qwen3-VL backbone as the vision-language model with a DiT-based flow matching action head.

## Installation

### Clone Repository

```sh
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

### Setup Conda Environment

Create a new conda environment for robotics training:

```sh
conda create -n flagos-robo python=3.12
conda activate flagos-robo
```

Install FlagScale and training dependencies:

```sh
cd FlagScale/
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

Download the base VLM model. Qwen-GR00T supports Qwen3-VL and Qwen2.5-VL as the VLM backbone:

**Using HuggingFace Hub:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id Qwen/Qwen3-VL-4B-Instruct \
    --output_dir /workspace/models \
    --source huggingface
```

**Using ModelScope:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id Qwen/Qwen3-VL-4B-Instruct \
    --output_dir /workspace/models \
    --source modelscope
```

The model will be downloaded to (example with `/workspace/models`):
- `/workspace/models/Qwen/Qwen3-VL-4B-Instruct`


## Training

### Prepare Dataset

FlagScale uses the **LeRobotDataset v3.0** format. For detailed information about the format structure, see the [LeRobotDataset v3.0 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3).

For example, to download the `libero_goal` dataset:

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

1. **Experiment-level config** (`examples/qwen_gr00t/conf/train.yaml`): Defines experiment settings, environment variables, and resource allocation
2. **Task-level config** (`examples/qwen_gr00t/conf/train/qwen_gr00t.yaml`): Defines model, dataset, and training hyperparameters

#### Experiment-Level Config

Edit the experiment-level config for multi-GPU training:

```sh
cd FlagScale/
vim examples/qwen_gr00t/conf/train.yaml
```

Configure the following fields:

- `experiment.envs.CUDA_VISIBLE_DEVICES` - GPU devices to use (default: `"0,1,2,3,4,5,6,7"` for 8 GPUs)
- `experiment.envs.CUDA_DEVICE_MAX_CONNECTIONS` - Connection limit (typically `1`)
- `experiment.exp_name` - Experiment name
- `experiment.exp_dir` - Output directory for checkpoints and logs

#### Task-Level Config

Edit the task-level config for model and training settings:

```sh
cd FlagScale/
vim examples/qwen_gr00t/conf/train/qwen_gr00t.yaml
```

Configure the following fields:

**System settings** (training hyperparameters):
- `system.batch_size` - Batch size per GPU (default: `16`)
- `system.train_steps` - Total training steps (default: `30000`)
- `system.grad_clip_norm` - Gradient clipping norm (default: `1.0`)
- `system.use_amp` - Whether to use automatic mixed precision (default: `true`)
- `system.shuffle` - Whether to shuffle training data (default: `true`)
- `system.num_workers` - Number of data loading workers (default: `4`)
- `system.checkpoint.save_checkpoint` - Whether to save checkpoints (default: `true`)
- `system.checkpoint.save_freq` - Steps between checkpoints (default: `1000`)
- `system.checkpoint.output_directory` - Checkpoint output directory (default: `${experiment.exp_dir}`)

**Model settings**:
- `model.model_name` - Model name: `"qwen_gr00t"`
- `model.checkpoint_dir` - Path to the pretrained base VLM model (e.g., `/workspace/models/Qwen/Qwen3-VL-4B-Instruct`)
- `model.vlm.type` - VLM backbone type: `"qwen3-vl"` or `"qwen2.5-vl"`
- `model.qwenvl.base_vlm` - Path to the base VLM (same as `model.checkpoint_dir`)
- `model.qwenvl.attn_implementation` - Attention implementation (default: `"flash_attention_2"`)
- `model.qwenvl.vl_hidden_dim` - VLM hidden dimension (default: `2048`)
- `model.dino.dino_backbone` - DINOv2 backbone variant (default: `"dinov2_vits14"`)
- `model.action_model.use_state` - Whether to condition the action model on proprioceptive state (default: `false`)
- `model.action_model.type` - Action model type (default: `"flow_matching"`)
- `model.action_model.action_model_type` - DiT variant (default: `"DiT-B"`)
- `model.action_model.action_dim` - Action dimension (default: `7`)
- `model.action_model.state_dim` - State dimension (default: `7`)
- `model.action_model.future_action_window_size` - Future action window (default: `7`)
- `model.action_model.action_horizon` - Action horizon (default: `8`)
- `model.action_model.num_inference_timesteps` - Inference diffusion steps (default: `4`)
- `model.reduce_in_full_precision` - Whether to reduce gradients in FP32 (default: `true`)

**Optimizer settings**:
- `model.optimizer.name` - Optimizer name (default: `"AdamW"`)
- `model.optimizer.lr` - Base learning rate (default: `2.5e-5`)
- `model.optimizer.betas` - Optimizer betas (default: `[0.9, 0.95]`)
- `model.optimizer.eps` - Optimizer epsilon (default: `1.0e-8`)
- `model.optimizer.weight_decay` - Weight decay (default: `1.0e-8`)
- `model.optimizer.param_groups` - Per-module learning rates:
  ```yaml
  param_groups:
    vlm:
      lr: 1.0e-05
    action_model:
      lr: 1.0e-04
  ```
- `model.optimizer.scheduler.name` - Scheduler name (default: `"cosine_with_min_lr"`)
- `model.optimizer.scheduler.warmup_steps` - Warmup steps (default: `5000`)
- `model.optimizer.scheduler.scheduler_kwargs.min_lr` - Minimum learning rate (default: `1.0e-6`)

**Module freezing** (optional):
```yaml
model:
  freeze:
    # Freeze VLM, train only action head
    freeze_patterns:
      - "qwen_vl_interface\\..*"
    # Optionally keep specific modules trainable
    keep_patterns:
      - "qwen_vl_interface\\.model\\.visual\\.merger\\..*"
```

**Data settings**:
- `data.data_path` - Path to LeRobot dataset root (e.g., `/workspace/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot`)
- `data.vla_data.data_mix` - Dataset mix name (e.g., `"libero_goal_old"`)
- `data.vla_data.action_type` - Action type (e.g., `"delta_qpos"`)
- `data.vla_data.default_image_resolution` - Image resolution `[C, H, W]` (default: `[3, 224, 224]`)
- `data.vla_data.obs` - Observation image keys (default: `["image_0"]`)
- `data.observation_delta_indices` - Observation delta indices (default: `[0]`)
- `data.action_delta_indices` - Action delta indices (default: `[0,1,2,3,4,5,6,7]`)
- `data.preprocessor` - Preprocessor pipeline configuration
- `data.postprocessor` - Postprocessor pipeline configuration

### Start Training
```sh
cd FlagScale/
flagscale train qwen_gr00t -c ./examples/qwen_gr00t/conf/train.yaml
```

Training logs are saved to `outputs/<exp_name>/logs/host_0_localhost.output` by default.

Checkpoints are saved to `${experiment.exp_dir}/checkpoints`.

### Stop Training
```sh
cd FlagScale/
flagscale train qwen_gr00t --stop
```

## Inference

### Prepare Inference Inputs

You can extract inference inputs (images, state, task) from a dataset using the provided script:

```sh
cd FlagScale/
python examples/pi0/dump_dataset_inputs.py \
    --dataset_root /workspace/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
    --output_dir ./qwen_gr00t_inference_inputs \
    --frame_index 100
```

This will create:
- `frame_100_observation_images_*.jpg` - Image files
- `frame_100_state.pt` - State tensor
- `frame_100_task.txt` - Task prompt
- `extraction_summary.json` - Summary of extracted files

### Edit Config

```sh
cd FlagScale/
vim examples/qwen_gr00t/conf/inference/qwen_gr00t.yaml
```

Configure the following fields:

**Engine settings:**
- `engine.model_variant` - Model variant (default: `"QwenGr00t"`)
- `engine.model` - Path to trained checkpoint (e.g., `/workspace/outputs/qwen_gr00t_train/checkpoints/last`)
- `engine.device` - Device to use (e.g., `"cuda"`)

**Generate settings:**
- `generate.images` - Dictionary mapping image keys to file paths:
  ```yaml
  images:
    observation.images.wrist_image: /path/to/wrist_image.jpg
    observation.images.image: /path/to/image.jpg
  ```
- `generate.state_path` - Path to state tensor file (`.pt` file)
- `generate.task_path` - Path to task prompt file (`.txt` file)

### Run Inference

```sh
cd FlagScale/
flagscale inference qwen_gr00t -c ./examples/qwen_gr00t/conf/inference.yaml
```

Inference logs are saved to `outputs/qwen_gr00t_inference/inference_logs/host_0_localhost.output` by default.

The predicted action tensor is printed to the console and saved in the log file.

## Serving

### Edit Config

```sh
cd FlagScale/
vim examples/qwen_gr00t/conf/serve/qwen_gr00t.yaml
```

Configure the following fields:

**Engine arguments:**
- `engine_args.host` - Server host (default: `"0.0.0.0"`)
- `engine_args.port` - Server port (default: `5000`)
- `engine_args.model_variant` - Model variant (default: `"QwenGr00t"`)
- `engine_args.model` - Path to trained checkpoint (e.g., `/workspace/outputs/qwen_gr00t_train/checkpoints/last`)
- `engine_args.device` - Device to use (e.g., `"cuda"`)

### Run Serving

```sh
cd FlagScale/
flagscale serve qwen_gr00t -c ./examples/qwen_gr00t/conf/serve.yaml
```

Serving logs are saved to `outputs/<exp_name>/logs/host_0_localhost.output` by default.

### Stop Serving

```sh
cd FlagScale/
flagscale serve qwen_gr00t --stop
```

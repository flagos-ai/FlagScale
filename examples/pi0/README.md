# PI0: Training, Inference, and Serving

This guide covers how to train, run inference, and serve PI0 models using FlagScale.

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

Install FlagScale and robotics dependencies:

```sh
cd FlagScale/
pip install . --verbose
pip install -r requirements/train/robotics/requirements.txt
```

Install additional dependencies for downloading models/datasets:

```sh
# For HuggingFace Hub
pip install huggingface_hub

# For ModelScope (optional)
pip install modelscope
```

## Download Models and Tokenizers

Download models and tokenizers using the provided script. Choose either HuggingFace Hub or ModelScope based on your preference:

**Using HuggingFace Hub:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id lerobot/pi0_base \
    --output_dir /workspace/models \
    --source huggingface

python examples/pi0/download.py \
    --repo_id google/paligemma-3b-pt-224 \
    --output_dir /workspace/models \
    --source huggingface
```

**Using ModelScope:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id lerobot/pi0_base \
    --output_dir /workspace/models \
    --source modelscope

python examples/pi0/download.py \
    --repo_id google/paligemma-3b-pt-224 \
    --output_dir /workspace/models \
    --source modelscope
```

The models will be downloaded to (example with `/workspace/models`):
- `/workspace/models/lerobot/pi0_base`
- `/workspace/models/google/paligemma-3b-pt-224`


## Training

### Prepare Dataset

FlagScale uses the **LeRobotDataset v3.0** format. For detailed information about the format structure, see the [LeRobotDataset v3.0 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3).

For example, to download the `aloha_mobile_cabinet` dataset:

**Using HuggingFace Hub:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id lerobot/aloha_mobile_cabinet \
    --output_dir /workspace/datasets \
    --repo_type dataset \
    --source huggingface
```

**Using ModelScope:**

```sh
cd FlagScale/
python examples/pi0/download.py \
    --repo_id lerobot/aloha_mobile_cabinet \
    --output_dir /workspace/datasets \
    --repo_type dataset \
    --source modelscope
```

The dataset will be downloaded to (example with `/workspace/datasets`):
- `/workspace/datasets/lerobot/aloha_mobile_cabinet`

### Edit Config

FlagScale uses a two-level configuration system:

1. **Experiment-level config** (`examples/pi0/conf/train.yaml`): Defines experiment settings, environment variables, and resource allocation
2. **Task-level config** (`examples/pi0/conf/train/pi0.yaml`): Defines model, dataset, and training hyperparameters

#### Experiment-Level Config

Edit the experiment-level config for multi-GPU training:

```sh
cd FlagScale/
vim examples/pi0/conf/train.yaml
```

Configure the following fields:

- `experiment.envs.CUDA_VISIBLE_DEVICES` - GPU devices to use (e.g., `"0,1,2,3"` for 4 GPUs, `"0,1"` for 2 GPUs)
- `experiment.envs.CUDA_DEVICE_MAX_CONNECTIONS` - Connection limit (typically `1`)
- `experiment.exp_name` - Experiment name
- `experiment.exp_dir` - Output directory for checkpoints and logs

#### Task-Level Config

Edit the task-level config for model and training settings:

```sh
cd FlagScale/
vim examples/pi0/conf/train/pi0.yaml
```

Configure the following fields:

**System settings** (training hyperparameters):
- `system.batch_size` - Batch size per GPU
- `system.train_steps` - Total training steps
- `system.optimizer_lr` - Learning rate
- `system.save_checkpoint` - Whether to save checkpoints (default: `true`)
- `system.save_freq` - Steps between checkpoints

**Model settings**:
- `model.model_variant` - Model variant: `"pi0"` or `"pi0.5"`
- `model.checkpoint_dir` - Path to pretrained model (e.g., `/workspace/models/lerobot/pi0_base`)
- `model.tokenizer_path` - Path to tokenizer (e.g., `/workspace/models/google/paligemma-3b-pt-224`)
- `model.tokenizer_max_length` - Maximum tokenizer sequence length
- `model.action_steps` - Number of action steps to predict

**Data settings**:
- `data.data_path` - Path to LeRobot dataset root (e.g., `/workspace/datasets/lerobot/aloha_mobile_cabinet`)
- `data.use_imagenet_stats` - Whether to use ImageNet normalization stats (default: `true`)
- `data.rename_map` - JSON string mapping dataset keys to policy keys (optional). Check the `features` key in your dataset's `meta/info.json` file to determine the correct mapping:
  ```yaml
  rename_map: '{"observation.images.cam_high": "observation.images.base_0_rgb", "observation.images.cam_left_wrist": "observation.images.left_wrist_0_rgb", "observation.images.cam_right_wrist": "observation.images.right_wrist_0_rgb"}'
  ```
- `data.use_quantiles` - Whether to use quantile normalization (for `pi0.5`, set to `false` to use MEAN_STD normalization)

### Start Training
```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name train action=run
```

Training logs are saved to `outputs/pi0_train/logs/host_0_localhost.output` by default.

Checkpoints are saved to `${experiment.exp_dir}/ckpt` (default: `outputs/pi0_train/ckpt`).

### Stop Training
```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name train action=stop
```

## Inference

### Prepare Inference Inputs

You can extract inference inputs (images, state, task) from a dataset using the provided script:

```sh
cd FlagScale/
python examples/pi0/dump_dataset_inputs.py \
    --dataset_root /workspace/datasets/lerobot/aloha_mobile_cabinet \
    --output_dir ./inference_inputs \
    --frame_index 100
```

This will create:
- `frame_100_observation_images_*.jpg` - Image files
- `frame_100_state.pt` - State tensor
- `frame_100_task.txt` - Task prompt
- `extraction_summary.json` - Summary of extracted files

Alternatively, you can extract from a specific episode and frame:

```sh
python examples/pi0/dump_dataset_inputs.py \
    --dataset_root /workspace/datasets/lerobot/aloha_mobile_cabinet \
    --output_dir ./inference_inputs \
    --episode_index 0 \
    --frame_in_episode 50
```

Or extract multiple samples at once:

```sh
python examples/pi0/dump_dataset_inputs.py \
    --dataset_root /workspace/datasets/lerobot/aloha_mobile_cabinet \
    --output_dir ./inference_inputs \
    --frame_indices 100 200 300
```

### Edit Config

```sh
cd FlagScale/
vim examples/pi0/conf/inference/pi0.yaml
```

Configure the following fields:

**Engine settings:**
- `engine.model` - Path to pretrained model (e.g., `/workspace/models/lerobot/pi0_base`)
- `engine.tokenizer` - Path to tokenizer (e.g., `/workspace/models/google/paligemma-3b-pt-224`)
- `engine.stat_path` - Path to dataset statistics (e.g., `/workspace/datasets/lerobot/aloha_mobile_cabinet/meta/stats.json`)
- `engine.device` - Device to use (e.g., `"cuda"`)

**Generate settings:**
- `generate.images` - Dictionary mapping image keys to file paths:
  ```yaml
  images:
    observation.images.cam_high: /path/to/image1.jpg
    observation.images.cam_left_wrist: /path/to/image2.jpg
    observation.images.cam_right_wrist: /path/to/image3.jpg
  ```
- `generate.state_path` - Path to state tensor file (`.pt` file)
- `generate.task_path` - Path to task prompt file (`.txt` file)
- `generate.rename_map` (optional) - Map input keys to policy expected keys:
  ```yaml
  rename_map:
    observation.images.cam_high: observation.images.base_0_rgb
    observation.images.cam_left_wrist: observation.images.left_wrist_0_rgb
    observation.images.cam_right_wrist: observation.images.right_wrist_0_rgb
  ```

### Run Inference

```sh
cd FlagScale/
python run.py \
    --config-path ./examples/pi0/conf \
    --config-name inference \
    action=run
```

Inference logs are saved to `outputs/pi0_inference/inference_logs/host_0_localhost.output` by default.

The predicted action tensor is printed to the console and saved in the log file.

## Serving

### Edit Config

```sh
cd FlagScale/
vim examples/pi0/conf/serve/pi0.yaml
```

Configure the following fields:

**Engine arguments:**
- `engine_args.host` - Server host (default: `"0.0.0.0"`)
- `engine_args.port` - Server port (default: `5000`)
- `engine_args.model` - Path to pretrained model (e.g., `/workspace/models/lerobot/pi0_base`)
- `engine_args.tokenizer` - Path to tokenizer (e.g., `/workspace/models/google/paligemma-3b-pt-224`)
- `engine_args.stat_path` - Path to dataset statistics (e.g., `/workspace/datasets/lerobot/aloha_mobile_cabinet/meta/stats.json`)
- `engine_args.device` - Device to use (e.g., `"cuda"`)
- `engine_args.images_keys` - List of image keys expected by the model (do not change):
  ```yaml
  images_keys:
    - observation.images.base_0_rgb
    - observation.images.left_wrist_0_rgb
    - observation.images.right_wrist_0_rgb
  ```
- `engine_args.images_shape` - Image shape `[C, H, W]` for warmup (e.g., `[3, 480, 640]`)
- `engine_args.state_key` - Key for state in the batch (e.g., `"observation.state"`)

### Run Serving

```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name serve action=run
```

Serving logs are saved to `outputs/pi0_serve/logs/host_0_localhost.output` by default.

### Stop Serving

```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name serve action=stop
```

### Test Server with Client

The client should send images using keys that match the `images_keys` in the config. For example, if using the default config:

```sh
cd FlagScale/
python examples/pi0/client_pi0.py \
  --host 127.0.0.1 \
  --port 5000 \
  --img1 ./inference_inputs/frame_100_observation_images_cam_high.jpg \
  --img2 ./inference_inputs/frame_100_observation_images_cam_left_wrist.jpg \
  --img3 ./inference_inputs/frame_100_observation_images_cam_right_wrist.jpg \
  --state-path ./inference_inputs/frame_100_state.pt \
  --instruction "Grab the orange and put it into the basket."
```

**Note**: The client must send image keys that match the `engine_args.images_keys` in the config.

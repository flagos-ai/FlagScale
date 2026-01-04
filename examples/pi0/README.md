#  Install FlagScale

Clone FlagScale code from github.

```sh
git clone -b refactor https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

Install train and inference env according to [README](https://github.com/FlagOpen/FlagScale/blob/main/README.md)

# Download Model

```sh
git lfs install

mkdir -p path-to-your-pi0_base-model
cd path-to-your-pi0_base-model
git clone https://huggingface.co/lerobot/pi0_base

mkdir -p path-to-your-paligemma-3b-pt-224-tokenizer
cd path-to-your-paligemma-3b-pt-224-tokenizer
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

If you don't have access to the international internet, download from modelscope.

```sh
modelscope download --model lerobot/pi0_base --local_dir path-to-your-pi0_base-model
modelscope download --model google/paligemma-3b-pt-224 --local_dir path-to-your-paligemma-3b-pt-224-tokenizer
```

# Training

## Prepare Dataset

FlagScale uses LeRobot dataset format. The training script expects a LeRobot dataset with the following structure:

```
dataset_root/
├── data/
│   ├── chunk-000/
│   │   ├── file-000.parquet
│   │   └── ...
│   └── ...
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── ...
└── videos/  (optional)
    └── ...
```

You can use an existing LeRobot dataset (e.g., `aloha_mobile_cabinet`) or convert your own data to LeRobot format. The dataset statistics (`stats.json`) are automatically loaded from the dataset's `meta/` directory.

For example, to use the `aloha_mobile_cabinet` dataset:

```sh
# Download the dataset from HuggingFace or modelscope
# The dataset should be located at: path-to-your-dataset
```

## Edit Config

```sh
cd FlagScale/
vim examples/pi0/conf/train/pi0.yaml
```

Configure the following fields:

**System settings** (training hyperparameters):
- `system.use_accelerator` -> Whether to use HuggingFace Accelerator (default: `false`)
- `system.batch_size` -> Batch size per GPU
- `system.train_steps` -> Total training steps
- `system.optimizer_lr` -> Learning rate
- `system.save_checkpoint` -> Whether to save checkpoints
- `system.save_freq` -> Steps between checkpoints

**Model settings**:
- `model.model_variant` -> Model variant: `"pi0"` or `"pi0.5"`
- `model.checkpoint_dir` -> Path to pretrained model (e.g., `path-to-your-pi0_base-model`)
- `model.tokenizer_path` -> Path to tokenizer (e.g., `path-to-your-paligemma-3b-pt-224-tokenizer`)
- `model.tokenizer_max_length` -> Maximum tokenizer sequence length
- `model.action_steps` -> Number of action steps to predict

**Data settings**:
- `data.data_path` -> Path to LeRobot dataset root (e.g., `path-to-your-dataset`)
- `data.use_imagenet_stats` -> Whether to use ImageNet normalization stats (default: `true`)
- `data.rename_map` -> JSON string mapping dataset keys to policy keys (optional):
  ```yaml
  rename_map: '{"observation.images.cam_high": "observation.images.base_0_rgb", "observation.images.cam_left_wrist": "observation.images.left_wrist_0_rgb", "observation.images.cam_right_wrist": "observation.images.right_wrist_0_rgb"}'
  ```
- `data.use_quantiles` -> Whether to use quantile normalization (for `pi0.5`, set to `false` to use MEAN_STD normalization)

## Start Training
```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name train action=run
```

# Inference

## Prepare Inference Inputs

You can extract inference inputs (images, state, task) from a dataset using the provided script:

```sh
cd FlagScale/
python examples/pi0/dump_dataset_inputs.py \
    --dataset_root /path/to/your/dataset \
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
    --dataset_root /path/to/your/dataset \
    --output_dir ./inference_inputs \
    --episode_index 0 \
    --frame_in_episode 50
```

Or extract multiple samples at once:

```sh
python examples/pi0/dump_dataset_inputs.py \
    --dataset_root /path/to/your/dataset \
    --output_dir ./inference_inputs \
    --frame_indices 100 200 300
```

## Edit Config

```sh
cd FlagScale/
vim examples/pi0/conf/inference/pi0.yaml
```

Configure the following fields:

**Engine settings:**
- `engine.model` -> Path to pretrained model (e.g., `path-to-your-pi0_base-model`)
- `engine.tokenizer` -> Path to tokenizer (e.g., `path-to-your-paligemma-3b-pt-224-tokenizer`)
- `engine.stat_path` -> Path to dataset statistics (e.g., `path-to-your-dataset/meta/stats.json`)
- `engine.device` -> Device to use (e.g., `"cuda"`)

**Generate settings:**
- `generate.images` -> Dictionary mapping image keys to file paths:
  ```yaml
  images:
    observation.images.cam_high: /path/to/image1.jpg
    observation.images.cam_left_wrist: /path/to/image2.jpg
    observation.images.cam_right_wrist: /path/to/image3.jpg
  ```
- `generate.state_path` -> Path to state tensor file (`.pt` file)
- `generate.task_path` -> Path to task prompt file (`.txt` file)
- `generate.rename_map` (optional) -> Map input keys to policy expected keys:
  ```yaml
  rename_map:
    observation.images.cam_high: observation.images.base_0_rgb
    observation.images.cam_left_wrist: observation.images.left_wrist_0_rgb
    observation.images.cam_right_wrist: observation.images.right_wrist_0_rgb
  ```

## Run Inference

```sh
cd FlagScale/
python run.py \
    --config-path ./examples/pi0/conf \
    --config-name inference \
    action=run
```

The script will:
1. Load the model and preprocessor/postprocessor pipelines
2. Load images, state, and task from the specified paths
3. Apply preprocessing (including rename_map if provided)
4. Run inference using `policy.select_action()`
5. Apply postprocessing to denormalize the action
6. Output the predicted action tensor

# Serving

## Edit Config

```sh
cd FlagScale/
vim examples/pi0/conf/serve/pi0.yaml
```

Configure the following fields:

**Engine settings:**
- `engine.host` -> Server host (default: `"0.0.0.0"`)
- `engine.port` -> Server port (default: `5000`)
- `engine.model` -> Path to pretrained model (e.g., `path-to-your-pi0_base-model`)
- `engine.tokenizer` -> Path to tokenizer (e.g., `path-to-your-paligemma-3b-pt-224-tokenizer`)
- `engine.stat_path` -> Path to dataset statistics (e.g., `path-to-your-dataset/meta/stats.json`)
- `engine.device` -> Device to use (e.g., `"cuda"`)
- `engine.model_variant` (optional) -> Model variant: `"pi0"` or `"pi0.5"` (default: `"pi0"`)
- `engine.use_quantiles` (optional) -> For `pi0.5`, set to `false` to use MEAN_STD normalization (default: `false`)

**Generate settings:**
- `generate.images_keys` -> List of image keys expected by the model:
  ```yaml
  images_keys:
    - observation.images.base_0_rgb
    - observation.images.left_wrist_0_rgb
    - observation.images.right_wrist_0_rgb
  ```
- `generate.images_shape` -> Image shape `[C, H, W]` for warmup (e.g., `[3, 480, 640]`)
- `generate.state_key` -> Key for state in the batch (e.g., `"observation.state"`)
- `generate.rename_map` (optional) -> Map client keys to model keys:
  ```yaml
  rename_map:
    observation.images.cam_high: observation.images.base_0_rgb
    observation.images.cam_left_wrist: observation.images.left_wrist_0_rgb
    observation.images.cam_right_wrist: observation.images.right_wrist_0_rgb
  ```

## Run Serving

```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name serve action=run
```

The server will:
1. Load the model and preprocessor/postprocessor pipelines
2. Perform a warmup inference
3. Start a Flask server on the specified host and port
4. Accept POST requests to `/infer` endpoint

## Test Server with Client

The client should send images using keys that match the `images_keys` in the config. For example, if using the default config:

```sh
cd FlagScale/
python examples/pi0/client_pi0.py \
  --host 127.0.0.1 \
  --port 5000 \
  --img1 /path/to/image1.jpg \
  --img2 /path/to/image2.jpg \
  --img3 /path/to/image3.jpg \
  --state-path /path/to/state.pt \
  --instruction "Grab the orange and put it into the basket."
```

**Note**: The client must send image keys that match the `generate.images_keys` in the config. If your dataset uses different keys, configure `generate.rename_map` to map them correctly.

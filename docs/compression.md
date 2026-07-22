<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Model Compression

## Overview

FlagScale provides two model compression pipelines:

| Pipeline | What It Does | Input | Output |
|----------|-------------|-------|--------|
| Mix-Precision | Auto-searches the best quantization strategy (8-bit or 4-bit) for each layer | BF16/FP16 model | Mixed 8-bit/4-bit model |
| FP8-to-INT8 | Converts FP8 model to W8A8 INT8 | FP8 or BF16 model | W8A8 INT8 model |

Both produce `compressed-tensors` format models. The FP8-to-INT8 output is directly loadable in vLLM; the mix-precision output requires dequantized validation (see [Use the Compressed Model](#use-the-compressed-model)).

## Setup

```sh
git clone https://github.com/flagos-ai/FlagScale.git
cd FlagScale
pip install .
pip install llmcompressor compressed-tensors
```

## Mix-Precision Quantization

This pipeline tests 4 strategies per layer and picks the best one by measuring output similarity:

| Strategy | Bits | Method |
|----------|------|--------|
| Std-8bit | 8 | Symmetric per-channel INT8 |
| QuIP-4bit | 4 | QuIP with Hadamard rotation |
| W4A16 | 4 | Symmetric per-channel INT4 |
| W4A16_ASYM | 4 | Asymmetric per-channel INT4 |

### Run

```sh
python run.py \
    --config-path examples/qwen35/conf/inference \
    --config-name compress_mix \
    action=run \
    compress.model.model_path=/path/to/your/model \
    compress.system.save_dir=/path/to/output \
    +experiment.runner.deploy.use_fs_serve=true \
    +compress.data.path=null
```

> Replace `/path/to/your/model` with your model directory (e.g., `./Qwen3-0.6B`).
>
> Replace `/path/to/output` with your desired output directory.
>
> The `+` prefix is Hydra syntax for adding a new config key that doesn't exist in the YAML file.

### View Logs

```sh
tail -f outputs/qwen35_mix/compress_logs/host_0_localhost.output
```

### Output

| Item | Location |
|------|----------|
| Compressed model | The `save_dir` you specified in the command (e.g., `/path/to/output`) |
| Run logs | `outputs/qwen35_mix/compress_logs/host_0_localhost.output` |
| Hydra config snapshot | `outputs/qwen35_mix/hydra/` |

The output `config.json` will contain a `quantization_config` with multiple groups, for example:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "config_groups": {
      "group_0": { "weights": { "num_bits": 8, "symmetric": true } },
      "group_3": { "weights": { "num_bits": 4, "symmetric": false } }
    }
  }
}
```

This means some layers were quantized to 8-bit (`group_0`) and others to 4-bit (`group_3`), based on the auto-search result.

### Configuration Files

| File | Purpose |
|------|---------|
| `examples/qwen35/conf/inference/compress_mix.yaml` | Experiment entry point. Defines `exp_name`, `entrypoint`, and `envs`. |
| `examples/qwen35/conf/inference/compress/mix_precision.yaml` | Compress settings. Change `scheme`, `targets`, `num_calibration_samples` here. |
| `examples/qwen35/conf/inference/compress/model.yaml` | Model settings. Change `model_path`, `model_cls`, `device_map`, `torch_dtype` here. |

### Configurable Parameters

| Parameter | How to Override | Default | Description |
|-----------|----------------|---------|-------------|
| Model path | `compress.model.model_path=...` | `BAAI/RoboBrain` | Path to the input model |
| Output dir | `compress.system.save_dir=...` | `Qwen3_30B_MixPrecision_Search` | Where to save the compressed model |
| Model class | `compress.model.model_cls=...` | `AutoModelForCausalLM` | Transformers model class |
| Device | `compress.model.device_map=...` | `cuda:0` | Device placement |
| Dtype | `compress.model.torch_dtype=...` | `float16` | Model dtype |

## FP8-to-INT8 Quantization

This pipeline works in 3 steps:

1. **Dequantize**: If the input is an FP8 model, restores weights to BF16. If the input is already BF16, this step is skipped automatically.
2. **Quantize**: Applies data-free post-training quantization to produce W8A8 INT8.
3. **Cleanup**: Removes temporary files.

### Run

```sh
python run.py \
    --config-path examples/qwen35/conf/inference \
    --config-name compress_fp8toint8 \
    experiment.envs.MODEL_ID=/path/to/your/model \
    experiment.envs.SAVE_DIR=/path/to/output
```

> Replace `/path/to/your/model` with your input model directory.
>
> Replace `/path/to/output` with your desired output directory.

### View Logs

```sh
tail -f outputs/fp8toint8_example/compress_logs/host_0_localhost.output
```

### Output

| Item | Location |
|------|----------|
| W8A8 model | The `SAVE_DIR` you specified in the command (e.g., `/path/to/output`) |
| Run logs | `outputs/fp8toint8_example/compress_logs/host_0_localhost.output` |
| Hydra config snapshot | `outputs/fp8toint8_example/hydra/` |

The output `config.json` will contain a W8A8 quantization config:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "config_groups": {
      "W8A8": {
        "weights": { "num_bits": 8, "symmetric": true, "strategy": "channel" },
        "input_activations": { "num_bits": 8, "symmetric": true, "strategy": "token" }
      }
    },
    "quantization_status": "compressed"
  }
}
```

### Configuration File

`examples/qwen35/conf/inference/compress_fp8toint8.yaml`:

| Field | How to Override | Default | Description |
|-------|----------------|---------|-------------|
| `MODEL_ID` | `experiment.envs.MODEL_ID=...` | `./models` | Path to input model (FP8 or BF16) |
| `SAVE_DIR` | `experiment.envs.SAVE_DIR=...` | `./outputs` | Path to save W8A8 model |
| `DEVICE` | `experiment.envs.DEVICE=...` | `auto` | Device: `auto`, `cuda:0`, `cpu` |
| `MAX_WORKERS` | `experiment.envs.MAX_WORKERS=...` | `8` | Parallel workers for shard processing |

## Use the Compressed Model

### FP8-to-INT8 Output (vLLM Ready)

The W8A8 INT8 model uses a uniform quantization config across all layers and can be loaded directly in vLLM:

```sh
vllm serve /path/to/compressed/model \
    --quantization compressed-tensors \
    --trust-remote-code
```

### Mix-Precision Output (Validation via Dequantization)

vLLM does not currently support loading models with per-layer mixed quantization (i.e., different layers quantized to different bit widths). Therefore, the mix-precision pipeline validates model quality through **dequantized inference** rather than vLLM loading.

During the auto-search process, each candidate strategy is evaluated by:

1. Applying the quantization to the layer's weights.
2. Dequantizing the weights back to the original dtype (e.g., float16).
3. Running a forward pass and comparing the output against the unquantized baseline.

The similarity scores reported in the logs reflect actual inference quality. However, the saved mix-precision model cannot be served by vLLM until native per-layer mixed quantization support is added.
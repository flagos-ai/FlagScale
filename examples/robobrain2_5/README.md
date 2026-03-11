
# Quick Start

## Installation

### Clone Repository

```sh
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

### Setup Conda Environment

Create a new conda environment:

```sh
conda create -n flagscale-inference python=3.12
conda activate flagscale-inference
```

### Install FlagScale

```sh
cd FlagScale/
pip install ".[cuda-train]"
```

### Install vllm-plugin-FL

Follow Setup in the vllm-plugin-FL [setup guide](https://github.com/flagos-ai/vllm-plugin-FL?tab=readme-ov-file#setup).

### Install Transformers

```sh
pip install transformers==4.57.0
```

## Download Model

```sh
git lfs install

mkdir -p /tmp/models/BAAI/
cd /tmp/models/BAAI/
git clone https://huggingface.co/BAAI/RoboBrain2.5-8B-NV
```

If you don't have access to the international internet, download from modelscope.

```sh
mkdir -p /tmp/models/
cd /tmp/models/
modelscope download --model BAAI/RoboBrain2.5-8B-NV --local_dir BAAI/RoboBrain2.5-8B-NV
```

## Inference

### Edit Inference Config

```sh
cd FlagScale/
vim examples/robobrain2_5/conf/inference/8b.yaml
```

Change 2 fields:

- llm.model: change to "/tmp/models/BAAI/RoboBrain2.5-8B-NV".
- generate.prompts: change to your customized input text.

### Run Inference

```sh
flagscale inference robobrain2_5 --config ./examples/robobrain2_5/conf/inference.yaml
# or
flagscale inference robobrain2_5 -c ./examples/robobrain2_5/conf/inference.yaml
```

### Check Logs

```sh
cd FlagScale/
tail -f outputs/robobrain2.5_8b/inference_logs/host_0_localhost.output
```

## Serving

### Edit Serving Config

```sh
cd FlagScale/
vim examples/robobrain2_5/conf/serve/8b.yaml
```

Change 1 fields:

- engine_args.model: change to "/tmp/models/BAAI/RoboBrain2.5-8B-NV".

## Run Serving

```sh
cd FlagScale/
flagscale serve robobrain2_5 --config ./examples/robobrain2_5/conf/serve.yaml
# or
flagscale serve robobrain2_5 -c ./examples/robobrain2_5/conf/serve.yaml
```

## Test Server with CURL

```sh
curl http://localhost:9010/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
"model": "",
"messages": [
{
    "role": "system",
    "content":
    [{
        "type": "text",
        "text": "123"
    }]
},
{
    "role": "user",
    "content":
    [{
        "type": "text",
        "text": "123"
    }]
}
],
"temperature": 0.0,
"max_completion_tokens": 200,
"stream": true,
"stream_options": {"include_usage": true}, "max_tokens": 4, "n_predict": 200
}'
```

## Training

Refer to [Qwen3-VL](../qwen3_vl/README.md)

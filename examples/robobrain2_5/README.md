
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

Install FlagScale:

```sh
cd FlagScale/
pip install . --verbose
```

### Install vLLM and Transformers

```sh
git clone https://github.com/flagos-ai/vllm-FL.git
cd vllm-FL
pip install packaging==24.2
pip install --no-build-isolation .
pip install transformers==4.57.0
```

## Download Model

```sh
git lfs install

mkdir -p /tmp/models/Qwen/
cd /tmp/models/Qwen/
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
```

If you don't have access to the international internet, download from modelscope.

```sh
mkdir -p /tmp/models/
cd /tmp/models/
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir Qwen/Qwen3-VL-4B-Instruct
```

## Inference

### Edit Inference Config

```sh
cd FlagScale/
vim examples/robobrain2_5/conf/inference/4b.yaml
```

Change 2 fields:

- llm.model: change to "/tmp/models/Qwen/Qwen3-VL-4B-Instruct".
- generate.prompts: change to your customized input text.

### Run Inference

```sh
python run.py --config-path ./examples/robobrain2_5/conf --config-name inference action=run
```

### Check Logs

```sh
cd FlagScale/
tail -f outputs/robobrain2.5_4b/serve_logs/host_0_localhost.output
```

## Serving

### Edit Serving Config

```sh
cd FlagScale/
vim examples/robobrain2_5/conf/serve/3b.yaml
```

Change 1 fields:

- engine_args.model: change to "/tmp/models/Qwen/Qwen3-VL-4B-Instruct".

## Run Serving

```sh
cd FlagScale/
python run.py --config-path ./examples/robobrain2_5/conf --config-name serve action=run
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

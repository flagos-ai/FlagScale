
# Quick Start

## Install FlagScale

1. Clone FlagScale code from github.

    ```sh
    git clone https://github.com/FlagOpen/FlagScale.git
    cd FlagScale/
    ```

2. Install Requirements

    We provide two installation methods: source code installation and whl package installation.
    - Source Installation

        ```sh
        cd FlagScale
        PYTHONPATH=./:$PYTHONPATH pip install . --config-settings=domain=robotics --config-settings=device=gpu  --verbose --no-build-isolation
        ```

    - Whl Installation

        ```sh
        pip install flag-scale[robotics-gpu]
        flagscale install --domain=robotics --device=gpu
        ```

    > **⚠️ Attention**: The robo environment depends on transformers (v4.53.0). Higher version will cause problem on image pre-processing.

## Install vLLM and Transformers

```sh
pip install vllm
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

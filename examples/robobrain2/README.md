
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

## Install vLLM

```sh
pip install vllm
```

## Download Model

```sh
git lfs install

mkdir -p /tmp/models/BAAI/
cd /tmp/models/BAAI/
git clone https://huggingface.co/BAAI/RoboBrain2.0-3B
```

If you don't have access to the international internet, download from modelscope.

```sh
mkdir -p /tmp/models/
cd /tmp/models/
modelscope download --model BAAI/RoboBrain2.0-3B --local_dir BAAI/RoboBrain2.0-3B
```

## Inference

### Edit Inference Config

```sh
cd FlagScale/
vim examples/robobrain2/conf/inference/3b.yaml
```

Change 2 fields:

- llm.model: change to "/tmp/models/BAAI/RoboBrain2.0-3B".
- generate.prompts: change to your customized input text.

### Run Inference

```sh
python run.py --config-path ./examples/robobrain2/conf --config-name inference action=run
```

### Check Logs

```sh
cd FlagScale/
tail -f  outputs/robobrain2.0_3b/inference_logs/host_0_localhost.output
```

## Serving

### Edit Serving Config

```sh
cd FlagScale/
vim examples/robobrain2/conf/serve/3b.yaml
```

Change 1 fields:

- engine_args.model: change to "/tmp/models/BAAI/RoboBrain2.0-3B"

## Run Serving

```sh
cd FlagScale/
python run.py --config-path ./examples/robobrain2/conf --config-name serve action=run
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

Refer to [Qwen2.5-VL](../qwen2_5_vl/README.md)

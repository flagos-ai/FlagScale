## Introduce

We introduce support for deploying large models with FlagScale, leveraging the Ray framework for efficient orchestration and scalability. Currently, this implementation supports the Qwen model, enabling users to easily deploy and manage large-scale machine learning services.

Future Key features include:

- Easy distributed Serve on base of eamless integration with Ray.
- Optimized resource management for large model inference.
- Simplified deployment process for the LLM and Multimodal models.

This enhancement will significantly improve the usability of FlagScale for large model deployment scenarios.

## Setup

[Install vLLM](../../README.md#setup)

## Prepare Model

[Prepare Qwen data](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/summary)

```shell
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /models/
```

## Serve run

```shell
cd FlagScale
python run.py --config-path ./examples/qwen/conf --config-name config_qwen2.5_7b action=run
```

## Serve call

```shell
curl http://127.0.0.1:4567/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "/models/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Introduce Bruce Lee in details."}
        ]
    }'
```

## Serve stop

```shell
cd FlagScale
python run.py --config-path ./examples/qwen/conf --config-name config_qwen2.5_7b action=stop
```

## logs

Since serve is the distributed mode, the logs are stored separately. \
The default logs of are located in `/outputs`.\


## Config Template

Flagscale.serve will support multiple scenarios. For better performance and usage, Flagscale.serve will optimize for specific scenarios, and these optimizations can be applied through different configurations.

### Command Line Mode with vLLM

If origin model is executed in command line mode with vLLM, we can use Flagscale.serve to deploy it easily.

```shell
vllm serve /models/Qwen2.5-7B-Instruct --tensor-parallel-size=1 --gpu-memory-utilization=0.9 --max-model-len=32768 --max-num-seqs=256 --port=4567 --trust-remote-code --enable-chunked-prefill
```

All the args remain the same as vLLM.

```YAML
- serve_id: vllm_model
  engine: vllm
  engine_args:
    model: /models/Qwen2.5-7B-Instruct
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_len: 32768
    max_num_seqs: 256
    port: 4567
    trust_remote_code: true
    enable_chunked_prefill: true
```

### How to config serve parameters
***deploy*** block is used to specify the parameters of serve. The ***models*** block is used to specify the parameters of each model decorated by "serve.remote".

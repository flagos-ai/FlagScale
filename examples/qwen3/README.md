# Serving via llama.cpp

## Download Model

Using the Qwen3-0.6B model for testing is an excellent choice, as its small parameter count enables a fast test. And it aligns perfectly with the context of edge-side deployment.

```sh
pip install modelscope
mkdir -p /tmp/flagscale_test
modelscope download --model Qwen/Qwen3-0.6B  --local_dir /tmp/flagscale_test/Qwen/Qwen3-0.6B
```

## Clone Source Code of llama.cpp

```sh
cd /tmp/flagscale_test
git clone https://github.com/ggml-org/llama.cpp.git
```

## Convert Model Format to GGUF

Llama.cpp provides converting tools, for example:

```sh
cd /tmp/flagscale_test/llama.cpp
python convert_hf_to_gguf.py /tmp/flagscale_test/Qwen/Qwen3-0.6B/ --outfile /tmp/flagscale_test/Qwen/Qwen3-0.6B/ggml_model_f16.gguf
```

## Build and Install llama.cpp

Build according to [this](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) if you don't build with CUDA.

```sh
sudo apt update && sudo apt install -y libcurl4-openssl-dev

cd /tmp/flagscale_test/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 64

export PATH=$PATH:/tmp/flagscale_test/llama.cpp/build/bin
```

## Inference Test in Conversation Mode

```sh
llama-cli -m /tmp/flagscale_test/Qwen/Qwen3-0.6B/ggml_model_f16.gguf
```

## Serving Test

Start server:

```sh
llama-server -m /tmp/flagscale_test/Qwen/Qwen3-0.6B/ggml_model_f16.gguf
```

Start a client with curl:

```sh
curl http://localhost:30000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
"model": "",
"messages": [
{
    "role": "system",
    "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
},
{
    "role": "user",
    "content": "Write a limerick about python exceptions"
}
]
}'
```

## Serving with FlagScale

Edit serve config file:

- examples/qwen3/conf/serve.yaml
  - key: defaults[1].serve
    - value: 8b -> 0_6b
  - key: experiment.envs.CUDA_VISIBLE_DEVICES
    - value: change to your available CUDA devices
  - key: experiment.exp_name
    - value: qwen3_8b -> qwen3_0.6b_llama.cpp
  - key: experiment.task.backend
    - value: sglang -> llama_cpp
- examples/qwen3/conf/serve/0_6b.yaml
  - key: [0].engine_args.model
    - value: /tmp/models/Qwen3-0.6B/ -> /tmp/flagscale_test/Qwen/Qwen3-0.6B/
  - key: [0].engine
    - remove key and value, deprecated

Start server with FlagScale:

```sh
cd FlagScale
python run.py --config-path ./examples/qwen3/conf --config-name serve action=run
```

Check logs:

```sh
cd FlagScale
cat outputs/qwen3_0.6b_llama.cpp/serve_logs/host_0_localhost.output
```

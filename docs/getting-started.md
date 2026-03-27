# Getting Started

## Overview

FlagScale leverages [Hydra](https://github.com/facebookresearch/hydra) for configuration management.
The configurations are organized into two levels: an outer experiment-level YAML file
and an inner task-level YAML file.

- The experiment-level YAML file defines the experiment directory, backend engine,
  task type, and other related environmental configurations.

- The task-level YAML file specifies the model, dataset, and parameters
  for specific tasks such as training or inference.

All valid configurations in the task-level YAML file correspond to the arguments
used in backend engines such as Megatron-LM and vllm, with hyphens (`-`)
replaced by underscores (`_`).
For a complete list of available configurations, please refer to the backend engine documentation.
You can simply copy and modify the existing YAML files in the [examples](../examples/)
folder to get started.

## 🔧 Setup

1. Install FlagScale

   **Option 1: Install via pip**
    ```sh
    pip install flagscale --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
    ```

   **Option 2: Install from source**
    ```sh
    git clone https://github.com/flagos-ai/FlagScale.git
    cd FlagScale
    pip install .
    ```


2. Install backends

- **Inference/Serving backend**

    We recommend using the latest release of flagscale-inference image.
    ```sh
    docker pull harbor.baai.ac.cn/flagscale/flagscale-inference:dev-cu128-py3.12-20260302102033
    docker run -itd --privileged --gpus all --net=host --ipc=host --device=/dev/infiniband --shm-size 512g --ulimit memlock=-1 --name <name>  harbor.baai.ac.cn/flagscale/flagscale-inference:dev-cu128-py3.12-20260302102033
    docker exec -it <name> /bin/bash
    conda activate flagscale-inference
    ```

    vLLM:
    ```sh
    pip install vllm==0.13.0
    ```

    vLLM-plugin-FL:
    ```sh
    pip install vllm-plugin-fl==0.1.0+vllm0.13.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
    ```
    See more details in [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL)

    FlagGems:
    ```sh
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    git clone https://github.com/flagos-ai/FlagGems
    cd FlagGems
    pip install --no-build-isolation . 
    ```
    See more details in [FlagGems](https://github.com/flagos-ai/FlagGems)


- **Training backend**

    We recommend using the latest release of flagscale-train image.
    ```sh
    docker pull harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856
    docker run -itd --gpus all --shm-size=500g --name <name>  harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856 /bin/bash
    docker exec -it <name> /bin/bash
    conda activate flagscale-train
    ```

    Megatron-LM-FL:
    ```sh
    pip install megatron_core==0.1.0+megatron0.15.0rc7 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
    ```
    See more details in [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL)

    TransformerEngine-FL:
    ```sh
    pip install transformer_engine==0.1.0+te2.9.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
    ```
    See more details in [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL)


- **RL backend**

    We recommend using the latest release of flagscale-train image.
    ```sh
    docker pull harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856
    docker run -itd --gpus all --shm-size=500g --name <name>  harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856 /bin/bash
    docker exec -it <name> /bin/bash
    conda activate flagscale-train
    ```
    verl-FL:
    ```sh
    pip install verl==0.1.0+verl0.7.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
    ```
    See more details in [verl-FL](https://github.com/flagos-ai/verl-FL.git) to get full installation instructions.


## Run a Task

FlagScale provides a unified runner for various tasks, including *training*,
*inference* and *serving*.
Simply specify the configuration file to run the task with a single command.
The runner will automatically load the configurations and execute the task.
The following sections demonstrate how to run a distributed training task.

### Train

Require Megatron-LM-FL env

1. Prepare dataset demo and tokenizer:

    - dataset

        We provide a small processed data ([bin](https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.bin) and [idx](https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.idx)) from the [Pile](https://pile.eleuther.ai/) dataset.
        ```sh
        mkdir -p ./data && cd ./data
        wget https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.idx
        wget https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.bin
        ```

    - tokenizer
        ```sh
        mkdir -p ./qwentokenizer && cd ./qwentokenizer
        wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/tokenizer_config.json" -O tokenizer_config.json
        wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/qwen.tiktoken" -O qwen.tiktoken
        wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/qwen_generation_utils.py" -O qwen_generation_utils.py
        wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/tokenization_qwen.py" -O tokenization_qwen.py
        ```

2. Edit config:

    Modify the data_path and tokenizer_path in ./examples/qwen3/conf/train/0_6b.yaml
    ```yaml
    data:
        data_path: ./data/enron_emails_demo_text_document_qwen    # modify data_path here
        split: 1
        no_mmap_bin_files: true
        tokenizer:
            legacy_tokenizer: true
            tokenizer_type: QwenTokenizerFS
            tokenizer_path: ./qwentokenizer   # modify tokenizer_path here
            vocab_size: 151936
            make_vocab_size_divisible_by: 64
    ```

    Modify config in ./examples/qwen3/conf/train.yaml
    ```yaml
    defaults:
      - _self_
      - train: 0_6b  # modify: train value must match its corresponding config file name
    ```


3. Start the distributed training job:
    ```sh
    flagscale train qwen3 --config ./examples/qwen3/conf/train.yaml
    # or
    flagscale train qwen3 -c ./examples/qwen3/conf/train.yaml
    ```

4. Stop the distributed training job:
    ```sh
    flagscale train qwen3 --stop
    ```


### Inference

Require vLLM-Plugin-FL env

1. Prepare model
    ```sh
    modelscope download --model Qwen/Qwen3-4B --local_dir ./Qwen3-4B
    ```

2. Edit config

    Modify model path in ./examples/qwen3/conf/inference/4b.yaml
    ```yaml
    llm:
        model: ./Qwen3-4B         # modify: Set model directory
        trust_remote_code: true
        tensor_parallel_size: 1
        pipeline_parallel_size: 1
        gpu_memory_utilization: 0.9
        seed: 1234
    ```

    Modify config in ./examples/qwen3/conf/inference_fl.yaml
    ```yaml
    defaults:
      - _self_
      - inference: 4b    # modify: Inference value must match its corresponding config file name
    ```

3. Start inference:
    ```sh
    flagscale inference qwen3 --config ./examples/qwen3/conf/inference_fl.yaml
    # or
    flagscale inference qwen3 -c ./examples/qwen3/conf/inference_fl.yaml
    ```

### Serve
1. Prepare model
    ```sh
    modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B
    ```

2. Edit Config

    Modify model path in ./examples/qwen3/conf/serve/0_6b.yaml
    ```yaml
    - serve_id: vllm_model
      engine_args:
        model: ./Qwen3-0.6B          # modify: Set model directory
        host: 0.0.0.0
        max_model_len: 4096
        max_num_seqs: 4
        uvicorn_log_level: warning
        port: 30000                  # A port available in your env, for example: 30000
    ```

    Modify config in ./examples/qwen3/conf/serve.yaml
    ```yaml
    defaults:
      - _self_
      - serve: 0_6b         # modify: Serve value must match its corresponding config file name
    experiment:
      exp_name: qwen3-0.6b  # modify as needed for test clarity
      exp_dir: outputs/${experiment.exp_name}
      task:
        type: serve
        backend: vllm
      runner:
        hostfile: null
        deploy:
        use_fs_serve: false
      envs:
        CUDA_VISIBLE_DEVICES: 0
        CUDA_DEVICE_MAX_CONNECTIONS: 1
    ```

3. Start the server:
    ```sh
    flagscale serve qwen3 --config ./examples/qwen3/conf/serve.yaml
    # or
    flagscale serve qwen3 -c ./examples/qwen3/conf/serve.yaml
    ```

4. Stop the server:
    ```sh
    flagscale serve qwen3 --stop
    ```

### RL
Require verl-FL env

1. Prepare model
    ```sh
    modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B
    ```
2. Prepare dataset
    ```
    mkdir gsm8k && cd gsm8k
    wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/train.parquet"
    wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/test.parquet"

    ```

3. Edit config

    Modify model path in ./examples/qwen3/conf/rl/0_6b.yaml
    ```yaml
    data:
        train_files: /workspace/data/gsm8k/train.parquet # modify: Set your train dataset
        val_files: /workspace/data/gsm8k/test.parquet # modify: Set your test dataset
        train_batch_size: 1024
        max_prompt_length: 512
        max_response_length: 1024
        filter_overlong_prompts: true
        truncation: "error"
    ```

    Modify model path in ./examples/qwen3/conf/rl/0_6b.yaml
    ```yaml
    actor_rollout_ref:
        model:
            path: /workspace/data/ckpt/Qwen3-0.6B # modify: Set your model checkpoint directory
            use_remove_padding: true
            enable_gradient_checkpointing: true
            trust_remote_code: true
    ```

    Modify config in ./examples/qwen3/conf/rl.yaml for experiment
    ```yaml
    experiment:
        exp_name: 0_6b
        exp_dir: /workspace/qwen3-rl/ # modify: Set your experiment directory
        runner:
          runtime_env: /path/to/verl-FL/verl/trainer/runtime_env.yaml # modify: Set your runtime_env.yaml
    ```

4. Start rl:
    ```sh
    flagscale rl qwen3 --config ./examples/qwen3/conf/rl.yaml
    # or
    flagscale rl qwen3 -c ./examples/qwen3/conf/rl.yaml
    ```
You can check the output in your experiment directory.

5. Stop rl:
    ```sh
    flagscale rl qwen3 --stop
    ```
    or force to stop ray cluster.
    ```sh
    ray stop
    ```

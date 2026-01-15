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
You can simply copy and modify the existing YAML files in the [examples](./examples)
folder to get started.

## 🔧 Setup
We recommend using the latest release of [NGC's PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for setup.

1. Clone the repository:
    ```sh
    git clone https://github.com/flagos-ai/FlagScale.git
    ```

2. Install FlagScale requirements
    ```sh
    pip install . --verbose
    ```

3. Install backends

- **Inference/Serving backend**

    vLLM-FL:
    ```sh
    git clone https://github.com/flagos-ai/vllm-FL
    cd vllm-FL
    pip install -e .
    ```
    See more details in [vLLM-FL](https://github.com/flagos-ai/vllm-FL)

    If you need vLLM-plugin-FL, see details in [vLLM-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL)


- **Training backend**
    Megatron-LM-FL:
    ```sh
    git clone https://github.com/flagos-ai/Megatron-LM-FL
    cd Megatron-LM-FL
    pip install --no-build-isolation .[mlm,dev]

    git clone https://github.com/NVIDIA/apex
    cd apex
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
    ```
    See more details in [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL)


- **RL backend**
    verl-FL:
    ```sh
    git clone https://github.com/flagos-ai/verl-FL.git
    cd verl-FL
    pip install --no-deps -e .
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

        We provide a small processed data ([bin](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin) and [idx](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx)) from the [Pile](https://pile.eleuther.ai/) dataset.
        ```sh
        mkdir -p ./data && cd ./data
        wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx
        wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin
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

    Modify the data_path and tokenizer_path in ./examples/qwen3/conf/train/0_6b.yaml at line 81 and line 87
    ```yaml
    data:
        data_path: ./data/pile_wikipedia_demo    # modify data_path here
        split: 1
        no_mmap_bin_files: true
        tokenizer:
            legacy_tokenizer: true
            tokenizer_type: QwenTokenizerFS
            tokenizer_path: ./qwentokenizer   # modify tokenizer_path here
            vocab_size: 151936
            make_vocab_size_divisible_by: 64
    ```

    Modify config in ./examples/qwen3/conf/train.yaml at line 3
    ```yaml
    defaults:
      - _self_
      - train: 0_6b  # modify: train value must match its corresponding config file name
    ```


3. Start the distributed training job:
    ```sh
    python run.py --config-path ./examples/qwen3/conf --config-name train action=run
    ```

4. Stop the distributed training job:
    ```sh
    python run.py --config-path ./examples/qwen3/conf --config-name train action=stop
    ```


### Inference

Require vLLM-FL env

1. Prepare model
    ```sh
    modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B
    ```

2. Edit config

    Modify model path in ./examples/qwen3/conf/inference/0_6b.yaml at line 2
    ```yaml
    llm:
        model: ./Qwen3-0.6B         # modify: Set model directory
        trust_remote_code: true
        tensor_parallel_size: 1
        pipeline_parallel_size: 1
        gpu_memory_utilization: 0.9
        seed: 1234
    ```

    Modify config in ./examples/qwen3/conf/inference_fl.yaml at line 3
    ```yaml
    defaults:
      - _self_
      - inference: 0_6b    # modify: Inference value must match its corresponding config file name
    ```

3. Start inference:
    ```sh
    python run.py --config-path ./examples/qwen3/conf --config-name inference_fl action=run
    ```

### Serve
1. Prepare model
    ```sh
    modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B
    ```

2. Edit Config

    Modify model path in ./examples/qwen3/conf/serve/0_6b.yaml at line 3
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

    Modify config in ./examples/qwen3/conf/serve.yaml at line 3
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
    python run.py --config-path ./examples/qwen3/conf --config-name serve action=run
    ```

4. Stop the server:
    ```sh
    python run.py --config-path ./examples/qwen3/conf --config-name serve action=stop
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

    Modify model path in ./examples/qwen3/conf/rl/0_6b.yaml at line 12 for dataset
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

    Modify model path in ./examples/qwen3/conf/rl/0_6b.yaml at line 21 for model checkpoint
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
    ```

4. Start rl:
    ```sh
    python run.py --config-path ./examples/qwen3/conf --config-name rl action=run
    ```
You can check the output in your experiment directory.

5. Stop rl:
    ```sh
    python run.py --config-path ./examples/qwen3/conf --config-name rl action=stop
    ```
    or force to stop ray cluster.
    ```sh
    ray stop
    ```

### Serving DeepSeek-R1 <a name="deepseek-r1-serving"></a>

We support serving the DeepSeek-R1 model and have implemented the `flagscale serve`
command for one-click deployment. By configuring just two YAML files,
you can easily serve the model using the `flagscale serve` command.

1. **Configure the YAML files:**

   ```none
   FlagScale/
    ├─ examples/
    │   └─ deepseek_r1/
    │        └─ conf/
    │            └─ serve.yaml
    |            └─ hostfile.txt # Set hostfile (optional)
    │            └─ serve/
    │               └─ 671b.yaml # Set model parameters and server port
   ```

   > [!Note]
   > When a task spans more than one nodes, a [hostfile.txt](./examples/deepseek/conf/hostfile.txt)
   > is required. Its path should be set in the `serve.yaml` configuration file.

2. Install FlagScale CLI:

   ```shell
   cd FlagScale
   pip install . --verbose --no-build-isolation
   ```

3. Start serving:

   ```shell
   flagscale serve deepseek_r1
   ```

Note that the `flagscale` command line supports customzation of service parameters:

```shell
flagscale serve <MODEL_NAME> <MODEL_CONFIG_YAML>
```

The configuration files allow you to specify the necessary parameters and settings
for your deployment, ensuring a smooth and efficient serving process.

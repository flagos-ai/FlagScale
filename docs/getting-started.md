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

### Setup

We recommend using the latest release of NGC's
[PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
for setup.

1. Clone the repository:

   ```shell
   git clone https://github.com/flagos-ai/FlagScale.git
   ```

1. Install FlagScale requirements:

   ```shell
   pip install . --verbose
   ```

1. Install backends:

   If you want to run a *serving*/*inference* task, please install vLLM-FL
   by following the instructions from the [vLLM-FL](https://github.com/flagos-ai/vllm-FL)
   project.

   If you want to run a *training* task, please install
   `Megatron-LM-FL` by following the documentation from the
   [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL) project.

## Run a Task

FlagScale provides a unified runner for various tasks, including *training*,
*inference* and *serving*.
Simply specify the configuration file to run the task with a single command.
The runner will automatically load the configurations and execute the task.
The following sections demonstrate how to run a distributed training task.

### Training

In this sample scenario, we will run a training task using the
[Megatron-LM backend](https://github.com/flagos-ai/Megatron-LM-FL).
After having installed the `Megatron-LM-FL` project, follow the steps below:

1. Prepare the demo dataset:

   We provide a small processed data ([bin](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin)
   and [idx](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx))
   from the [Pile](https://pile.eleuther.ai/) dataset.

   ```shell
   mkdir -p /path/to/data && cd /path/to/data
   wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx
   wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin
   ```

1. Edit the configuration:

   Modify the `data_path` in the [sample configuration file](../examples/aquila/conf/train/7b.yaml)
   file, as shown below:

   ```yaml
   data:
     data_path: /path/to/data  # modify data path here
     split: 1
     tokenizer:
       legacy_tokenizer: true
       tokenizer_type: AquilaTokenizerFS
       vocab_file: ./examples/aquila/tokenizer/vocab.json
       merge_file: ./examples/aquila/tokenizer/merges.txt
       special_tokens_file: ./examples/aquila/tokenizer/special_tokens.txt
       vocab_size: 100008
   ```

1. Start the distributed training job:

   ```shell
   python run.py --config-path ./examples/aquila/conf --config-name train action=run
   ```

1. Stop the distributed training job:

   ```shell
   python run.py --config-path ./examples/aquila/conf --config-name train action=stop
   ```

### Inference

In this section, we run an *inference* task using the [vLLM backend](https://github.com/flagos-ai/vllm-FL).
After having installed the vLLM-FL backend, follow the steps below:

1. Prepare the model:

   ```shell
   modelscope download --model BAAI/Aquila-7B README.md --local_dir ./
   ```

1. Edit the configuration:

   Change the `llm.model` and the `tokernizer` value in the
   [sample configuration file](../examples/aquila/conf/inference/7b.yaml),
   as shown below:

   ```yaml
   llm:
     model: /path/to/the/model      # e.g. /workspace/models/BAAI/Aquila-7B
     tokenizer: /path/to/the/model  # e.g. /workspace/models/BAAI/Aquila-7B
     trust_remote_code: true
     tensor_parallel_size: 1
     pipeline_parallel_size: 1
     gpu_memory_utilization: 0.5
     seed: 1234
   ```

1. Start the inference task:

   ```shell
   python run.py --config-path ./examples/aquila/conf --config-name inference action=run
   ```

### Serving

This section demonstrates how to start a *serving* task.

1. Download the tokenizer:

   ```shell
   mkdir -p /models/physical-intelligence/
   cd /models/physical-intelligence/
   git lfs install
   git clone https://huggingface.co/physical-intelligence/fast
   ```

1. Edit the configuration:

   Edit the [sample configuration](../examples/robobrain_x0/conf/serve/robobrain_x0.yaml)
   by changing the following three fields under `engine_args`:

   - set `model_sub_task` to model path, e.g. `/models/BAAI/RoboBrain-X0-Preview`;
   - set `port` to a port that is available in your environment, e.g. `5001`;
   - set `tokenizer_path` to the tokernizer path, e.g. `/models/physical-intelligence/fast`.

1. Start the server:

   ```shell
   python run.py --config-path ./examples/robobrain_x0/conf --config-name serve action=run
   ```

1. To stop the server, run

   ```shell
   python run.py --config-path ./examples/robobrain_x0/conf --config-name serve action=stop
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

1. Install FlagScale CLI:

   ```shell
   cd FlagScale
   pip install . --verbose --no-build-isolation
   ```

1. Start serving:

   ```shell
   flagscale serve deepseek_r1
   ```

Note that the `flagscale` command line supports customzation of service parameters:

```shell
flagscale serve <MODEL_NAME> <MODEL_CONFIG_YAML>
```

The configuration files allow you to specify the necessary parameters and settings
for your deployment, ensuring a smooth and efficient serving process.

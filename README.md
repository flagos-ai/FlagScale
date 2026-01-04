
[<img width="4750" height="958" alt="github+banner__2025-11-11+13_27_10" src="https://github.com/user-attachments/assets/e63014d8-ac72-4b82-98f7-aeed9833672a" />](https://www.flagopen.ac.cn/)


## 🔥 Latest News

- **[2026/01]** Released [v1.0.0-alpha.0](https://github.com/flagos-ai/FlagScale/releases/tag/v1.0.0-alpha.0):
  - Major updates: Refactored the codebase by moving hardware-specific (multi-chip) support into plugin repositories such as [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) and [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL). These plugins build on top of [FlagOS](https://github.com/flagos-ai), a unified open-source AI system software stack. We also re-initialized the Git commit history to reduce repository size.
  - Compatibility (legacy users): If you are using or upgrading from a version earlier than v1.0.0-alpha.0, please use the [main-legacy](https://github.com/flagos-ai/FlagScale/tree/main-legacy) branch. It will continue to receive critical bug fixes and minor updates for a period of time.

- **[2025/09]** Released [v0.9.0](https://github.com/FlagOpen/FlagScale/tree/release/v0.9.0):
  - Training & Finetuning: Added LoRA for efficient finetuning, improved the autotuner for cross-chip heterogeneous training, and enabled distributed RWKV training.
  - Inference & Serving: Introduced DiffusionEngine for FLUX.1-dev, Qwen-Image, and Wan2.1-T2V, support multi-model automatic orchestration and dynamic scaling.
  - Embodied AI: Full lifecycle support for Robobrain, Robotics, and PI0, plus semantic retrieval for MCP-based skills for RoboOS.
  - Elastic & Fault Tolerance: Detect task status automatically (errors, hangs, etc.) and periodically record them.
  - Hardware & System: Broader chip support, upgraded patch mechanism with file-level diffs, and enhanced CICD for different chips.

- **[2025/04]** Released [v0.8.0](https://github.com/FlagOpen/FlagScale/tree/release/v0.8.0):
  - Introduced a new flexible and robust multi-backend mechanism and updated vendor adaptation methods.
  - Enabled heterogeneous prefill-decoding disaggregation across vendor chips within a single instance via FlagCX (beta).
  - Upgraded DeepSeek-V3 pre-training with the new Megatron-LM and added heterogeneous pre-training across different chips for MoE models like DeepSeek-V3.
- **[2025/02]** Released [v0.6.5](https://github.com/FlagOpen/FlagScale/tree/release/v0.6.5):
  - Added support for DeepSeek-V3 distributed pre-training (beta) and [DeepSeek-V3/R1 serving](#deepseek-r1-serving) across multiple chips.
  - Introduced an auto-tuning feature for serving and a new CLI feature for one-click deployment.
  - Enhanced the CI/CD system to support more chips and integrated the workflow of [FlagRelease](https://huggingface.co/FlagRelease).
- **[2024/11]** Released [v0.6.0](https://github.com/FlagOpen/FlagScale/tree/release/v0.6.0):
  - Introduced general multi-dimensional heterogeneous parallelism and CPU-based communication between different chips.
  - Added the full support for LLaVA-OneVision, achieving SOTA results on the [Infinity-MM](https://arxiv.org/abs/2410.18558) dataset.
  - Open-sourced the optimized CFG implementation and accelerated the generation and understanding tasks for [Emu3](https://arxiv.org/abs/2409.18869).
  - Implemented the auto-tuning feature and enhanced the CI/CD system.
- **[2024/4]** Released [v0.3](https://github.com/FlagOpen/FlagScale/tree/release/v0.3): Achieved heterogeneous hybrid training of the Aquila2-70B-Expr model on a cluster using both NVIDIA and Iluvatar chips. Adapted the Aquila2 series to AI chips from six different manufacturers.
- **[2023/11]** Released [v0.2](https://github.com/FlagOpen/FlagScale/tree/v0.2): Introduced training support for Aquila2-70B-Expr, enabling heterogeneous training across chips with the same or compatible architectures.
- **[2023/10]** Released [v0.1](https://github.com/FlagOpen/FlagScale/tree/v0.1): Supported Aquila models with optimized training schemes for Aquila2-7B and Aquila2-34B, including parallel strategies, optimizations, and hyper-parameter settings.
## 🔗 About

[FlagScale](https://github.com/FlagOpen/FlagScale.git) is a comprehensive toolkit designed to support the entire lifecycle of large models, developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI). It builds on the strengths of several prominent open-source projects, including [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [vllm](https://github.com/vllm-project/vllm), to provide a robust, end-to-end solution for managing and scaling large models.

The primary objective of FlagScale is to enable seamless scalability across diverse hardware architectures while maximizing computational resource efficiency and enhancing model performance. By offering essential components for model development, training, and deployment, FlagScale seeks to establish itself as an indispensable toolkit for optimizing both the speed and effectiveness of large model workflows.

FlagScale is also a part of [FlagAI-Open](https://flagopen.baai.ac.cn/), an open-source initiative by BAAI that aims to foster an open-source ecosystem for AI technologies. It serves as a platform where developers, researchers, and AI enthusiasts can collaborate on various AI projects, contribute to the development of cutting-edge AI solutions, and share their work with the global community.

**join our WeChat Group**
</p> <align=center>
<img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/566bd17d-c43f-4af7-9a29-7a6c7e610ffa" />
</p>

## ✏️ Support List
### Training
| Model                                                    | Example config File                        |
| -------------------------------------------------------- | ------------------------------------------------|
| [DeepSeek-V3](https://huggingface.co/deepseek-ai)  | [16b_a3b.yaml](examples/deepseek_v3/conf/train/16b_a3b.yaml)  |
| [Qwen2/2.5/3](https://huggingface.co/Qwen)             | [235b_a22b.yaml](examples/qwen3/conf/train/235b_a22b.yaml)  |
| [Qwen2.5-VL](https://huggingface.co/Qwen)             | [7b.yaml](examples/qwen2_5_vl/conf/train/7b.yaml)  |
| [QwQ](https://huggingface.co/Qwen)             | [32b.yaml](examples/qwq/conf/train/32b.yaml)  |
| [LLaMA2](https://huggingface.co/meta-llama)             | [7b.yaml](examples/llama2/conf/train/7b.yaml)  |
| [LLaMA3/3.1](https://huggingface.co/meta-llama)             | [70b.yaml](examples/llama3/conf/train/70b.yaml)  |
| [LLaVA-OneVision](https://huggingface.co/lmms-lab)             | [7b.yaml](examples/llava_onevision/conf/train/7b.yaml)  |
| [LLaVA1.5](https://huggingface.co/llava-hf)             | [7b.yaml](examples/llava1_5/conf/train/7b.yaml)  |
| [Mixtral](https://huggingface.co/mistralai)             | [8x7b.yaml](examples/mixtral/conf/train/8x7b.yaml)  |
| [RWKV](https://huggingface.co/RWKV)             | [7b.yaml](examples/rwkv/conf/train/7b.yaml)  |
| [Aquila](https://huggingface.co/BAAI)             | [7b.yaml](examples/aquila/conf/train/7b.yaml)  |
| ... | ... |


### Serve/Inference
| Model                                                    | Example config File                        |
| -------------------------------------------------------- | ------------------------------------------------|
| [DeepSeek-V3](https://huggingface.co/deepseek-ai)  | [671b.yaml](examples/deepseek_v3/conf/serve/671b.yaml)  |
| [DeepSeek-R1](https://huggingface.co/deepseek-ai)  | [671b.yaml](examples/deepseek_r1/conf/serve/671b.yaml)  |
| [Qwen2.5](https://huggingface.co/Qwen)             | [72b.yaml](examples/qwen2_5/conf/serve/72b.yaml)  |
| [Qwen3](https://huggingface.co/Qwen)             | [8b.yaml](examples/qwen3/conf/serve/8b.yaml)  |
| [Qwen2.5-VL](https://huggingface.co/Qwen)             | [32b_instruct.yaml](examples/qwen2_5_vl/conf/serve/32b_instruct.yaml)  |
| [Qwen3-Omni](https://huggingface.co/Qwen)             | [30b.yaml](examples/qwen3_o/conf/serve/30b.yaml)  |
| [QwQ](https://huggingface.co/Qwen)             | [32b.yaml](examples/qwq/conf/serve/32b.yaml)  |
| [Grok2](https://huggingface.co/xai-org)             | [270b.yaml](examples/grok2/conf/serve/270b.yaml)  |
| [Kimi-K2](https://huggingface.co/MoonshotAI)             | [1t.yaml](examples/kimi_k2/conf/serve/1t.yaml)  |
| ... | ... |


## 🚀 Quick Start

FlagScale leverages [Hydra](https://github.com/facebookresearch/hydra) for configuration management. The configurations are organized into two levels: an outer experiment-level YAML file and an inner task-level YAML file.

- The experiment-level YAML file defines the experiment directory, backend engine, task type, and other related environmental configurations.
- The task-level YAML file specifies the model, dataset, and parameters for specific tasks such as training or inference.

All valid configurations in the task-level YAML file correspond to the arguments used in backend engines such as Megatron-LM and vllm, with hyphens (-) replaced by underscores (_). For a complete list of available configurations, please refer to the backend engine documentation. Simply copy and modify the existing YAML files in the [examples](./examples) folder to get started.

### 🔧 Setup
We recommend using the latest release of [NGC's PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for setup.

1. Clone the repository:
    ```sh
    git clone https://github.com/FlagOpen/FlagScale.git
    ```

2. Install FlagScale requirements
    ```sh
    pip install . --verbose 
    ```

3. Install backends 

    If you want to run a serving/inference task, please install vLLM-FL according to https://github.com/flagos-ai/vllm-FL

    If you want to run a training task, please install Megatron-LM-Fl according to https://github.com/flagos-ai/Megatron-LM-FL



### 🎈 Run a Task

FlagScale provides a unified runner for various tasks, including training，inference and serve. Simply specify the configuration file to run the task with a single command. The runner will automatically load the configurations and execute the task. The following example demonstrates how to run a distributed training task.

#### Train

Require megatron env. See details in [Setup](#-setup)

1. Prepare dataset demo:

    We provide a small processed data ([bin](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin) and [idx](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx)) from the [Pile](https://pile.eleuther.ai/) dataset.
    ```
    mkdir -p /path/to/data && cd /path/to/data
    wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx
    wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin
    ```
2. Edit config:

    Modify the data path in ./examples/aquila/conf/train/7b.yaml
    ```yaml
    data:
        data_path: ${data_path:??}  # modify data path here
        split: 1
        tokenizer:
            legacy_tokenizer: true
            tokenizer_type: AquilaTokenizerFS
            vocab_file: ./examples/aquila/tokenizer/vocab.json
            merge_file: ./examples/aquila/tokenizer/merges.txt
            special_tokens_file: ./examples/aquila/tokenizer/special_tokens.txt
            vocab_size: 100008
    ```



3. Start the distributed training job:
    ```sh
    python run.py --config-path ./examples/aquila/conf --config-name train action=run
    ```


4. Stop the distributed training job:
    ```sh
    python run.py --config-path ./examples/aquila/conf --config-name train action=stop
    ```

#### Inference

Require vllm env. See details in [Setup](#-setup)

1. Prepare model
    ```sh
    modelscope download --model BAAI/Aquila-7B README.md --local_dir ./
    ```
2. Edit config

    FlagScale/examples/aquila/conf/inference/7b.yaml
    ```yaml
    llm:
        model: /workspace/models/BAAI/Aquila-7B         # modify path here
        tokenizer: /workspace/models/BAAI/Aquila-7B     # modify path here
        trust_remote_code: true
        tensor_parallel_size: 1
        pipeline_parallel_size: 1
        gpu_memory_utilization: 0.5
        seed: 1234

    ```

3. Start inference:
    ```sh
    python run.py --config-path ./examples/aquila/conf --config-name inference action=run
    ```
    
#### Serve
1. Download Tokenizer
    ```sh
    mkdir -p /models/physical-intelligence/
    cd /models/physical-intelligence/
    git lfs install
    git clone https://huggingface.co/physical-intelligence/fast
    ```

2. Edit Config

    ./examples/robobrain_x0/conf/serve/robobrain_x0.yaml

    Change 3 fields:
    - engine_args.model_sub_task -> /models/BAAI/RoboBrain-X0-Preview
    - engine_args.port -> A port available in your env, for example: 5001
    - engine_args.tokenizer_path ->/models/physical-intelligence/fast

3. Start the server:
    ```sh
    python run.py --config-path ./examples/robobrain_x0/conf --config-name serve action=run
    ```
4. Stop the server:
    ```sh
    python run.py --config-path ./examples/robobrain_x0/conf --config-name serve action=stop
    ```


### 🧱 DeepSeek-R1 Serving <a name="deepseek-r1-serving"></a>

We support the model serving of DeepSeek R1 and have implemented the `flagscale serve` command for one-click deployment. By configuring just two YAML files, you can easily serve the model using the `flagscale serve` command.

1. **Configure the YAML files:**
    ```
    FlagScale/
    ├── examples/
    │   └── deepseek_r1/
    │       └── conf/
    │           └── serve.yaml
    |           └── hostfile.txt # Set hostfile (optional)
    │           └── serve/
    │               └── 671b.yaml # Set model parameters and server port
    ```
    Note: When task covers multiple nodes, [hostfile.txt](./examples/deepseek/conf/hostfile.txt) is required. The file path should be set in serve.yaml.

2. **Install FlagScale CLI:**
    ```sh
    cd FlagScale
    pip install . --verbose --no-build-isolation
    ```

3. **One-click serve:**
    ```sh
    flagscale serve deepseek_r1
    ```

4. **Custom service parameters:**
    ```sh
    flagscale serve <MODEL_NAME> <MODEL_CONFIG_YAML>
    ```

The configuration files allow you to specify the necessary parameters and settings for your deployment, ensuring a smooth and efficient serving process.


## 📄 License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE). This project also contains other third-party components under other open-source licenses. See the [LICENSE](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE) file for more information.

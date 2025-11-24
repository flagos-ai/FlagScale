
[<img width="4750" height="958" alt="github+banner__2025-11-11+13_27_10" src="https://github.com/user-attachments/assets/e63014d8-ac72-4b82-98f7-aeed9833672a" />](https://www.flagopen.ac.cn/)


## Latest News

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

## About

[FlagScale](https://github.com/FlagOpen/FlagScale.git) is a comprehensive toolkit designed to support the entire lifecycle of large models, developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI). It builds on the strengths of several prominent open-source projects, including [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [vllm](https://github.com/vllm-project/vllm), to provide a robust, end-to-end solution for managing and scaling large models.

The primary objective of FlagScale is to enable seamless scalability across diverse hardware architectures while maximizing computational resource efficiency and enhancing model performance. By offering essential components for model development, training, and deployment, FlagScale seeks to establish itself as an indispensable toolkit for optimizing both the speed and effectiveness of large model workflows.

FlagScale is also a part of [FlagAI-Open](https://flagopen.baai.ac.cn/), an open-source initiative by BAAI that aims to foster an open-source ecosystem for AI technologies. It serves as a platform where developers, researchers, and AI enthusiasts can collaborate on various AI projects, contribute to the development of cutting-edge AI solutions, and share their work with the global community.

**join our WeChat Group**
</p> <align=center>
<img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/566bd17d-c43f-4af7-9a29-7a6c7e610ffa" />
</p>

## Quick Start

FlagScale leverages [Hydra](https://github.com/facebookresearch/hydra) for configuration management. The configurations are organized into two levels: an outer experiment-level YAML file and an inner task-level YAML file.

- The experiment-level YAML file defines the experiment directory, backend engine, task type, and other related environmental configurations.
- The task-level YAML file specifies the model, dataset, and parameters for specific tasks such as training or inference.

All valid configurations in the task-level YAML file correspond to the arguments used in backend engines such as Megatron-LM and vllm, with hyphens (-) replaced by underscores (_). For a complete list of available configurations, please refer to the backend engine documentation. Simply copy and modify the existing YAML files in the [examples](./examples) folder to get started.

### Setup
We recommend using the latest release of [NGC's PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for setup.

1. Clone the repository:
    ```sh
    git clone https://github.com/FlagOpen/FlagScale.git
    ```

2. Install the requirements:
    ```sh
    cd FlagScale
    ./install/install-requirements.sh --env train
    ./install/install-requirements.sh --env inference
    ```
    The above instructions create two conda environments: `flagscale-train` and `flagscale-inference`, which contain the dependency environments for training and inference, respectively.

    The llama.cpp backend (CPU by default) is alternative, for example:
    ```sh
    ./install/install-requirements.sh --env inference --llama-cpp-backend cuda
    ```

    **Note**: To customize the environment according to your specific needs, follow the methods below:  
    1. To install apt packages or basic drivers, add them to the platform-specific `Dockerfile` in the `docker` directory.
    2. To include cross-platform Python packages, declare them in the corresponding files in the `requirements` directory.
    3. To include platform-specific Python packages, configure them in the corresponding files in the `install` directory.

3. Unpatch the backend code adaptation of FlagScale as needed
    ```
    cd FlagScale
    python tools/patch/unpatch.py --backend Megatron-LM
    python tools/patch/unpatch.py --backend vllm
    ```

4. Install the packages with customized extensions:
    ```sh
    cd FlagScale/third_party/vllm
    pip install .

    cd FlagScale
    python tools/patch/unpatch.py --backend Megatron-Energon
    cd third_party/Megatron-Energon
    pip install -e .
    cd FlagScale
    cp -r third_party/Megatron-Energon/src/megatron/energon third_party/Megatron-LM/megatron
    ```
    And we currently support two one-click installation methods: source installation and whl installation. Usage is as follows:

    - Source Installation
        ```sh
        PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation --verbose \
        --config-settings=device=<device> \
        --config-settings=backend=<backend>
        ```

    - Whl Installation
        ```sh
        PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation --verbose
        flagscale install <backend> --device=<device>
        ```

    More backends and chips will be supported in the future.

5. Patch the modifications to the specified third_party backend for PR.
    ```
    cd FlagScale
    python tools/patch/patch.py --backend Megatron-LM
    python tools/patch/patch.py --backend vllm
    ```

### Run a Task

FlagScale provides a unified runner for various tasks, including training，inference and serve. Simply specify the configuration file to run the task with a single command. The runner will automatically load the configurations and execute the task. The following example demonstrates how to run a distributed training task.

#### Train

1. Start the distributed training job:
    ```sh
    python run.py --config-path ./examples/aquila/conf --config-name train action=run
    ```
    The `data_path` in the demo is the path of the training datasets following the [Megatron-LM format](./megatron/README.md#data-preprocessing). For quickly running the pretraining process, we also provide a small processed data ([bin](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin) and [idx](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx)) from the [Pile](https://pile.eleuther.ai/) dataset.

2. Stop the distributed training job:
    ```sh
    python run.py --config-path ./examples/aquila/conf --config-name train action=stop
    ```

#### Serve

1. Start the server:
    ```sh
    python run.py --config-path ./examples/qwen/conf --config-name serve action=run
    ```
2. Stop the server:
    ```sh
    python run.py --config-path ./examples/qwen/conf --config-name serve action=stop
    ```
For more details, please refer to [Quick Start](./flagscale/serve/README.md).

### DeepSeek-R1 Serving <a name="deepseek-r1-serving"></a>

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
    PYTHONPATH=./:$PYTHONPATH pip install . --verbose --no-build-isolation
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

## License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE). This project also contains other third-party components under other open-source licenses. See the [LICENSE](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE) file for more information.

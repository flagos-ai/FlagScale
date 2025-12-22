
[<img width="4750" height="958" alt="github+banner__2025-11-11+13_27_10" src="https://github.com/user-attachments/assets/e63014d8-ac72-4b82-98f7-aeed9833672a" />](https://www.flagopen.ac.cn/)

<!--TODO(Qiming): Leave a placeholder for announcements.
## 🔥 Latest News
[Changelog](./docs/CHANGELOG.md)
-->

## About

FlagCX is part of [FlagOS](https://flagos.io/), a unified, open-source AI system software stack that
aims to foster an open technology ecosystem by seamlessly integrating various models, systems and chips.
By "develop once, migrate across various chips", FlagOS aims to unlock the full computational potential
of hardware, break down the barriers between different chip software stacks, and effectively reduce
migration costs.

FlagScale is a comprehensive toolkit designed to support the entire lifecycle of large models.
It builds on the strengths of several prominent open-source projects, including
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [vLLM](https://github.com/vllm-project/vllm),
to provide a robust, end-to-end solution for managing and scaling large models.

The primary objective of FlagScale is to enable seamless scalability across diverse hardware architectures
while maximizing computational resource efficiency and enhancing model performance.
By offering essential components for model development, training, and deployment, FlagScale seeks to
establish itself as an indispensable toolkit for optimizing both the speed and effectiveness of large model workflows.

## Getting Started

Follow the [setup](./docs/setup.md) documentation to deploy the software.
FlagScale leverages [Hydra](https://github.com/facebookresearch/hydra) for configuration management.
The configurations are organized into two levels: an outer experiment-level YAML file and an inner task-level YAML file.

- The experiment-level YAML file defines the experiment directory, backend engine, task type, and other related environmental configurations.
- The task-level YAML file specifies the model, dataset, and parameters for specific tasks such as training or inference.

All valid configurations in the task-level YAML file correspond to the arguments used in backend engines
such as Megatron-LM and vllm, with hyphens (`-`) replaced by underscores (`_`).
For a complete list of available configurations, please refer to the backend engine documentation.
Simply copy and modify the existing YAML files in the [examples](./examples) folder to get started.

## Run a Task

FlagScale provides a unified runner for various tasks, including *training*，*inference* and *serve*.
Simply specify the configuration file to run the task with a single command.
The runner will automatically load the configurations and execute the task.
The following examples demonstrate how to run a distributed training task.

- [Training using Megatron-LM](./docs/task-train.md)
- [Inferencing using vLLM](./docs/task-inference.md)
- [Serving using vLLM](./docs/task-serve.md)
- [Serving using DeepSeek-R1](./docs/task-serve-ds-r1.md)

## Platforms Supported

| Vendor                        | vLLM | Megatron-LM |
| ----------------------------- | ---- | ----------- |
| BI V150                       | ✅   | ✅          |
| Cambricon MLU                 | ✅   | ✅          |
| Huawei Atlas800 TA3 (Ascend)  | ✅   | ✅          |
| Hygon BW1000                  | ✅   | ✅          |
| Kunlunxin R310p               | ✅   | ✅          |
| Metax C550                    | ✅   | ✅          |
| MUSA S5000                    | ✅   | ✅          |
| TsingMicro                    | ✅   | ✅          |
| NVIDIA+Cambricon MLU          |      | ✅          |

## Models Supported

### Training

| Model                                               | Example config File                                           |
| --------------------------------------------------- | ------------------------------------------------------------- |
| [Aquila](https://huggingface.co/BAAI)               | [7b.yaml](examples/aquila/conf/train/7b.yaml)                 |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai)   | [16b_a3b.yaml](examples/deepseek_v3/conf/train/16b_a3b.yaml)  |
| [LLaMA2](https://huggingface.co/meta-llama)         | [7b.yaml](examples/llama2/conf/train/7b.yaml)                 |
| [LLaMA3/3.1](https://huggingface.co/meta-llama)     | [70b.yaml](examples/llama3/conf/train/70b.yaml)               |
| [LLaVA-OneVision](https://huggingface.co/lmms-lab)  | [7b.yaml](examples/llava_onevision/conf/train/7b.yaml)        |
| [LLaVA1.5](https://huggingface.co/llava-hf)         | [7b.yaml](examples/llava1_5/conf/train/7b.yaml)               |
| [Mixtral](https://huggingface.co/mistralai)         | [8x7b.yaml](examples/mixtral/conf/train/8x7b.yaml)            |
| [Qwen2/2.5/3](https://huggingface.co/Qwen)          | [235b_a22b.yaml](examples/qwen3/conf/train/235b_a22b.yaml)    |
| [Qwen2.5-VL](https://huggingface.co/Qwen)           | [7b.yaml](examples/qwen2_5_vl/conf/train/7b.yaml)             |
| [QwQ](https://huggingface.co/Qwen)                  | [32b.yaml](examples/qwq/conf/train/32b.yaml)                  |
| [RWKV](https://huggingface.co/RWKV)                 | [7b.yaml](examples/rwkv/conf/train/7b.yaml)                   |
| ... | ... |


### Serve/Inference

| Model                                               | Example config File                                                   |
| --------------------------------------------------- | ------------------------------------------------                      |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai)   | [671b.yaml](examples/deepseek_v3/conf/serve/671b.yaml)                |
| [DeepSeek-R1](https://huggingface.co/deepseek-ai)   | [671b.yaml](examples/deepseek_r1/conf/serve/671b.yaml)                |
| [Grok2](https://huggingface.co/xai-org)             | [270b.yaml](examples/grok2/conf/serve/270b.yaml)                      |
| [Kimi-K2](https://huggingface.co/MoonshotAI)        | [1t.yaml](examples/kimi_k2/conf/serve/1t.yaml)                        |
| [Qwen2.5](https://huggingface.co/Qwen)              | [72b.yaml](examples/qwen2_5/conf/serve/72b.yaml)                      |
| [Qwen3](https://huggingface.co/Qwen)                | [8b.yaml](examples/qwen3/conf/serve/8b.yaml)                          |
| [Qwen2.5-VL](https://huggingface.co/Qwen)           | [32b_instruct.yaml](examples/qwen2_5_vl/conf/serve/32b_instruct.yaml) |
| [Qwen3-Omni](https://huggingface.co/Qwen)           | [30b.yaml](examples/qwen3_o/conf/serve/30b.yaml)                      |
| [QwQ](https://huggingface.co/Qwen)                  | [32b.yaml](examples/qwq/conf/serve/32b.yaml)                          |
| ... | ... |

## Contributing

Patch the modifications to the specified third-party backend for PR.

```
cd FlagScale
python tools/patch/patch.py --backend Megatron-LM
python tools/patch/patch.py --backend vllm
```

**join our WeChat Group**

</p> <align=center>
<img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/566bd17d-c43f-4af7-9a29-7a6c7e610ffa" />
</p>

## License

This project is licensed under the [Apache License (Version 2.0)](./LICENSE).
This project also contains other third-party components under other open-source licenses.
See the [LICENSE](./LICENSE) file for more information.

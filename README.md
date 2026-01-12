[中文](./README_cn.md)

<!--TODO(Qiming): replace this banner-->
[<img width="4750" height="958" alt="github+banner__2025-11-11+13_27_10" src="https://github.com/user-attachments/assets/e63014d8-ac72-4b82-98f7-aeed9833672a" />](https://www.flagopen.ac.cn/)

<!--TODO(Qiming): Placeholder for announcements.-->

## Overview

FlagScale is part of [FlagOS](https://flagos.io/), a unified, open-source AI system software stack that
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

## ⚠️ Important: Major Refactoring (2026/01)
- Refactored the codebase since [v1.0.0-alpha.0](https://github.com/flagos-ai/FlagScale/releases/tag/v1.0.0-alpha.0) by moving hardware-specific (multi-chip) support into plugin repositories such as [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) and [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL). These plugins build on top of [FlagOS](https://github.com/flagos-ai), a unified open-source AI system software stack. If you are using or upgrading from a version earlier than v1.0.0-alpha.0, please use the [main-legacy](https://github.com/flagos-ai/FlagScale/tree/main-legacy) branch. It will continue to receive critical bug fixes and minor updates for a period of time.

## Resources

- [Project change log](./docs/CHANGELOG.md)
- [Getting started](./docs/getting-started.md)
- [FlagOS-Robo related](./docs/robo.md)

## Support List

### Training

| Model                                                    | Example config File                        |
| -------------------------------------------------------- | ------------------------------------------------------------- |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai)        | [16b_a3b.yaml](examples/deepseek_v3/conf/train/16b_a3b.yaml)  |
| [Qwen2/2.5/3](https://huggingface.co/Qwen)               | [235b_a22b.yaml](examples/qwen3/conf/train/235b_a22b.yaml)    |
| [Qwen2.5-VL](https://huggingface.co/Qwen)                | [7b.yaml](examples/qwen2_5_vl/conf/train/7b.yaml)             |
| [QwQ](https://huggingface.co/Qwen)                       | [32b.yaml](examples/qwq/conf/train/32b.yaml)                  |
| [LLaMA2](https://huggingface.co/meta-llama)              | [7b.yaml](examples/llama2/conf/train/7b.yaml)                 |
| [LLaMA3/3.1](https://huggingface.co/meta-llama)          | [70b.yaml](examples/llama3/conf/train/70b.yaml)               |
| [LLaVA-OneVision](https://huggingface.co/lmms-lab)       | [7b.yaml](examples/llava_onevision/conf/train/7b.yaml)        |
| [LLaVA1.5](https://huggingface.co/llava-hf)              | [7b.yaml](examples/llava1_5/conf/train/7b.yaml)               |
| [Mixtral](https://huggingface.co/mistralai)              | [8x7b.yaml](examples/mixtral/conf/train/8x7b.yaml)            |
| [RWKV](https://huggingface.co/RWKV)                      | [7b.yaml](examples/rwkv/conf/train/7b.yaml)                   |
| [Aquila](https://huggingface.co/BAAI)                    | [7b.yaml](examples/aquila/conf/train/7b.yaml)                 |
| ... | ... |

### Serve/Inference

| Model                                                    | Example config File                                                   |
| -------------------------------------------------------- | --------------------------------------------------------------------- |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai)        | [671b.yaml](examples/deepseek_v3/conf/serve/671b.yaml)                |
| [DeepSeek-R1](https://huggingface.co/deepseek-ai)        | [671b.yaml](examples/deepseek_r1/conf/serve/671b.yaml)                |
| [Qwen2.5](https://huggingface.co/Qwen)                   | [72b.yaml](examples/qwen2_5/conf/serve/72b.yaml)                      |
| [Qwen3](https://huggingface.co/Qwen)                     | [8b.yaml](examples/qwen3/conf/serve/8b.yaml)                          |
| [Qwen2.5-VL](https://huggingface.co/Qwen)                | [32b_instruct.yaml](examples/qwen2_5_vl/conf/serve/32b_instruct.yaml) |
| [Qwen3-Omni](https://huggingface.co/Qwen)                | [30b.yaml](examples/qwen3_o/conf/serve/30b.yaml)                      |
| [QwQ](https://huggingface.co/Qwen)                       | [32b.yaml](examples/qwq/conf/serve/32b.yaml)                          |
| [Grok2](https://huggingface.co/xai-org)                  | [270b.yaml](examples/grok2/conf/serve/270b.yaml)                      |
| [Kimi-K2](https://huggingface.co/MoonshotAI)             | [1t.yaml](examples/kimi_k2/conf/serve/1t.yaml)                        |
| ... | ... |

## Contribution

**join our WeChat Group**

<p align=center>
<img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/566bd17d-c43f-4af7-9a29-7a6c7e610ffa" />
</p>

## License

This project is licensed under the [Apache License (Version 2.0)](./LICENSE).
This project also contains other third-party components under other open-source licenses.

[<img width="2182" height="602" alt="github+banner-20260130" src=".github/assets/banner-20260130.png" />](https://flagos.io/)

[[中文版](./README_cn.md)|English]

<div align="right">
  <a href="https://www.linkedin.com/company/flagos-community" target="_blank">
    <img src="./docs/assets/Linkedin.png" alt="LinkIn" width="32" height="32" />
  </a>

  <a href="https://www.youtube.com/@FlagOS_Official" target="_blank">
    <img src="./docs/assets/youtube.png" alt="YouTube" width="32" height="32" />
  </a>

  <a href="https://x.com/FlagOS_Official" target="_blank">
    <img src="./docs/assets/x.png" alt="X" width="32" height="32" />
  </a>

  <a href="https://www.facebook.com/FlagOSCommunity" target="_blank">
    <img src="./docs/assets/Facebook.png" alt="X" width="32" height="32" />
  </a>
</div>

<!--Begin Announcements.-->

> [!IMPORTANT]
>
> **2026/03 UPDATE**
>
> [v1.0.0](https://github.com/flagos-ai/FlagScale/releases/tag/v1.0.0) is now officially released as the first stable version.
> The codebase has been significantly refactored since [v1.0.0-alpha.0](https://github.com/flagos-ai/FlagScale/releases/tag/v1.0.0-alpha.0).
> The hardware-specific (multi-chip) support has been moved into plugin repositories such as
> [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) and
> [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL).
> These plugins build on top of [FlagOS](https://flagos.io/), a unified open-source AI system software stack.
> If you are using or upgrading from a version earlier than *v1.0.0-alpha.0*, please use the
> [`main-legacy`](https://github.com/flagos-ai/FlagScale/tree/main-legacy) branch.
> It will continue to receive critical bug fixes and minor updates for a period of time.
<!--End Announcements.-->


## Overview

FlagScale is a core component of [FlagOS](https://flagos.io/) — a unified, open-source AI system software stack that fosters an open technology ecosystem by seamlessly integrating various models, systems, and chips. Following the principle of "develop once, migrate across various chips", FlagOS aims to unlock the full computational potential of hardware, break down barriers between different chip software stacks, and effectively reduce migration costs.To get started, see the [Getting Started Guide](./docs/getting-started.md).

As the central toolkit of this ecosystem, FlagScale provides a unified interface covering the complete lifecycle of large language models, multimodal models, and embodied AI models. It integrates multiple open-source backend engines under a single configuration and CLI interface, supporting key workflows including model training, reinforcement learning, and inference — with consistent operation across diverse chip vendors.

Within the FlagOS ecosystem, FlagScale works together with several other components:
- FlagOS Plugins – hardware-adapted integrations of upstream AI frameworks
- [FlagCX](https://docs.flagos.io/projects/FlagCX/en/latest/) – a scalable and adaptive cross-chip communication library
- [FlagOS-Robo related](https://github.com/flagos-ai/FlagOS-Robo) – infrastructure for embodied AI workloads

FlagOS plugin projects are built on top of widely used upstream open-source frameworks and extend them to support multiple AI chips. These plugins provide hardware compatibility and runtime integration for training, reinforcement learning, and inference.

The following table lists the mapping between FlagOS plugins and their corresponding upstream projects.

| Task | FlagOS Plugin Projects | Upstream Projects |
|------|----------------------|-------------------|
| Training | [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL) <br> [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) <br> [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) |
| Reinforcement Learning | [VeRL-FL](https://github.com/flagos-ai/verl-FL) | [veRL](https://github.com/verl-project/verl) |
| Serve / Inference | [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) | [vllm](https://github.com/vllm-project/vllm) |


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

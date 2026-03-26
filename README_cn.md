[<img width="2182" height="602" alt="github+banner-20260130" src=".github/assets/banner-20260130.png" />](https://flagos.io/)

[中文版|[English](./README.md)]

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
> **2026/03 更新**
>
> [v1.0.0](https://github.com/flagos-ai/FlagScale/releases/tag/v1.0.0) 现已正式发布，这是首个稳定版本。
> 自 [v1.0.0-alpha.0](https://github.com/flagos-ai/FlagScale/releases/tag/v1.0.0-alpha.0) 起，代码库已进行重大重构。
> 针对特定硬件的多芯片支持已迁移至插件仓库，例如
> [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) 和
> [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL)。
> 这些插件基于 [FlagOS](https://flagos.io/)（统一的开源 AI 系统软件栈）构建。
> 如果您正在使用或从早于 *v1.0.0-alpha.0* 的版本升级，请使用
> [`main-legacy`](https://github.com/flagos-ai/FlagScale/tree/main-legacy) 分支。
> 该分支将在一段时间内继续接收关键错误修复和小版本更新。

<!--End Announcements.-->


## 介绍

FlagScale 是 [FlagOS](https://flagos.io/) 的核心组件。FlagOS 是一个统一的开源 AI 系统软件栈，通过无缝集成各类模型、系统与芯片，构建开放的技术生态。秉承"一次开发，多芯迁移"的理念，FlagOS 旨在充分释放硬件算力潜能，打破不同芯片软件栈之间的壁垒，有效降低迁移成本。

作为该生态的核心工具包，FlagScale 提供统一的接口，覆盖大语言模型、多模态模型及具身智能模型的完整生命周期。它在统一的配置项和命令行界面下集成了多个开源后端引擎，支持模型训练、强化学习和推理等关键工作流，并在多种芯片厂商间保持一致的运行体验。快速上手请参阅 [快速入门指南](./docs/getting-started.md)。

在 FlagOS 生态中，FlagScale 与以下组件协同工作：
- **FlagOS 插件** — 对上游 AI 框架进行硬件适配的集成组件
- [**FlagCX**](https://docs.flagos.io/projects/FlagCX/en/latest/) — 可扩展的自适应跨芯片通信库
- [**FlagOS-Robo**](https://github.com/flagos-ai/FlagOS-Robo) — 具身智能工作负载的基础设施

FlagOS 插件项目基于广泛使用的上游开源框架构建，并对其进行扩展以支持多种 AI 芯片，为训练、强化学习和推理提供硬件兼容性和运行时集成。

下表列出了 FlagOS 插件与对应上游项目的映射关系：

| 任务 | FlagOS 插件项目 | 上游项目 |
|------|----------------|---------|
| 训练 | [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL) <br> [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) <br> [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) |
| 强化学习 | [VeRL-FL](https://github.com/flagos-ai/verl-FL) | [veRL](https://github.com/verl-project/verl) |
| 推理 / 服务 | [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) | [vllm](https://github.com/vllm-project/vllm) |


## 资源

- [项目变更历史](./docs/CHANGELOG.md)
- [快速上手指南](./docs/getting-started.md)

## 支持列表

### 模型训练

| 模型                                                   | 示例配置文件                                                  |
| ------------------------------------------------------ | ------------------------------------------------------------- |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai)      | [16b_a3b.yaml](examples/deepseek_v3/conf/train/16b_a3b.yaml)  |
| [Qwen2/2.5/3](https://huggingface.co/Qwen)             | [235b_a22b.yaml](examples/qwen3/conf/train/235b_a22b.yaml)    |
| [Qwen2.5-VL](https://huggingface.co/Qwen)              | [7b.yaml](examples/qwen2_5_vl/conf/train/7b.yaml)             |
| [QwQ](https://huggingface.co/Qwen)                     | [32b.yaml](examples/qwq/conf/train/32b.yaml)                  |
| [LLaMA2](https://huggingface.co/meta-llama)            | [7b.yaml](examples/llama2/conf/train/7b.yaml)                 |
| [LLaMA3/3.1](https://huggingface.co/meta-llama)        | [70b.yaml](examples/llama3/conf/train/70b.yaml)               |
| [LLaVA-OneVision](https://huggingface.co/lmms-lab)     | [7b.yaml](examples/llava_onevision/conf/train/7b.yaml)        |
| [LLaVA1.5](https://huggingface.co/llava-hf)            | [7b.yaml](examples/llava1_5/conf/train/7b.yaml)               |
| [Mixtral](https://huggingface.co/mistralai)            | [8x7b.yaml](examples/mixtral/conf/train/8x7b.yaml)            |
| [RWKV](https://huggingface.co/RWKV)                    | [7b.yaml](examples/rwkv/conf/train/7b.yaml)                   |
| [Aquila](https://huggingface.co/BAAI)                  | [7b.yaml](examples/aquila/conf/train/7b.yaml)                 |
| ... | ... |

### 服务、推理

| 模型                                                     | 示例配置文件                                                          |
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

## 参与贡献

**请加入我们的微信群**

<p align=center>
<img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/566bd17d-c43f-4af7-9a29-7a6c7e610ffa" />
</p>

## 授权许可

FlagScale 采用 [Apache License (Version 2.0)](./LICENSE) 授权许可。
本项目中也包含一些使用其他开源授权许可的第三方组件。

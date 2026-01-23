[English](./README.md)

<!--TODO(Qiming): replace this banner-->
[<img width="4750" height="958" alt="github+banner__2025-11-11+13_27_10" src="https://github.com/user-attachments/assets/e63014d8-ac72-4b82-98f7-aeed9833672a" />](https://www.flagopen.ac.cn/)

<!--Begin  announcements.-->
> [!IMPORTANT]
>
> **2026/01 重要更新**
>
> 与之前的 [v1.0.0-alpha.0](https://github.com/flagos-ai/FlagScale/releases/tag/v1.0.0-alpha.0)
> 版本相比，FlagScale 代码仓库完成重大重构。与硬件（多芯片支持）相关的代码已经被迁移到
> [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) 和
> [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) 这类插件仓库（Plugin Repository）中。
> 这些插件基于统一、开源的 AI 系统软件堆栈 [FlagOS](https://flagos.io/) 构建。
> 如果你在使用或迁移到比 *v1.0.0-alpha.0* 更早的版本，请使用代码仓库的
> [`main-legacy`](https://github.com/flagos-ai/FlagScale/tree/main-legacy) 分支。
> 此分支将在一定时间内继续接受重大缺陷修复以及一些小的更新。
<!--End announcements.-->

## 介绍

FlagScale 是 [FlagOS](https://flagos.io/) 的一部分，而 FlagOS 是一个统一的开源 AI 系统软件堆栈，
通过无缝集成各种模型、系统和芯片技术，打造一个开放的技术生态系统。
通过实现“一次开发、跨多种芯片迁移”，FlagOS 力图充分释放硬件的计算潜能，
破除不同芯片软件堆栈之间的壁垒，进而有效降低解决方案的迁移开销。

FlagScale 是一个综合而全面的软件工具包，设计用来支持大模型的整个生命周期。
构建于若干主流开源项目（例如
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 和 [vLLM](https://github.com/vllm-project/vllm) 等）
之上，FlagScale 为大模型的管理和规模扩展提供一套稳定而强大的端到端解决方案。

FlagScale 的核心目标是跨多种硬件体系结构实现平滑的规模扩缩，同时完成对计算资源极致利用，
提升模型的性能表现。通过为模型开发、训练和部署提供至关重要的组件，
FlagScale 力图将自身打磨成为优化大模型工作流的速度与效能时不可或缺的工具套件。

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

### 伺服、推理

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

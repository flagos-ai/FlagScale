# Qwen3.5-4B 图文训练验证记录

最新的 8 卡性能对比见 [吞吐报告](throughput_report.md)，统一使用固定序列长度并直接统计第 6–15 步日志。本文保留早期 2 卡接入、权重及恢复验证记录。

验证日期：2026-09-07。本文记录 Qwen3.5 适配过程中的实测结果；相关 FlagScale 代码已随 PR #1284 合入 `main`，主干代码基准为 `b5741d04760a3fcdd33c70524697540b1322a3fe`。复现时使用该提交或包含该适配的后续版本及下述配套环境。合入主干不代表重新执行过本文测试。

## 模型与验证范围

使用官方 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) 完整权重，包含视觉编码器、Gated DeltaNet / full attention 混合语言模型和 1 层辅助 MTP。没有缩小模型层数或隐藏维度。

环境为 2 张 H100 80GB、TP=2、PP=1、BF16、序列长度 512、global batch size 2。Torch 2.7.1+cu128、Transformers 5.3.0、FLA/fla-core 0.4.2、Energon 6.0.1；使用现有 Megatron-LM-FL 和 TransformerEngine-FL 安装。

数据为 LLaVA-Pretrain 中的 5000 条 BLIP/LAION/CC/SBU 图文样本，转换为 ChatML WebDataset。复现步骤见 [README](README.md)，使用 [train_smoke.yaml](conf/train_smoke.yaml)。

## 上游 MTP + mRoPE 支持

FlagScale `main` 中的 Qwen3.5 支持 MTP 与 mRoPE 同时使用。接入时恢复了升级过程中遗漏的 mRoPE 参数校验支持，并适配 Megatron-LM-FL v0.18.2 的 `mtp_on_this_rank` 参数接口；无需再按历史提交单独应用修复。

## 实际训练与恢复

先从官方权重转换出的 release checkpoint 训练 4 步，在第 2、4 步保存；再从第 4 步 checkpoint 恢复到第 6 步。日志确认恢复 iteration 4 和对应的 dataloader state；恢复配置未设置 finetune/no_load_optim/no_load_rng，并使用 checkpoint 中的 scheduler 配置。

| 步数 | LM loss | MTP loss | 消费样本数 |
| --- | --- | --- | --- |
| 1 | 3.279989 | 3.349526 | 2 |
| 4 | 2.421995 | 2.920817 | 8 |
| 5（恢复后） | 2.589418 | 2.825871 | 10 |
| 6 | 2.017563 | 2.123785 | 12 |

所有 6 步均无 skipped iteration、无 NaN iteration。第 6 步模型、优化器和数据游标 checkpoint 已保存，训练正常结束。

## 早期 FLOPs 与吞吐记录（不用于性能对比）

这次早期运行开启了动态序列长度，框架公式却按配置长度统计，回算吞吐不能视为实际执行长度对应的性能结论。后续性能验证已改为固定长度，直接取日志吞吐。

早期日志未开启 `log_throughput`，这里只记录 checkpoint 中的累计 FLOPs；吞吐统一以八卡报告中的直接日志读数为准。`train_smoke.yaml` 现已开启吞吐日志。

框架计算入口为 `flagscale/train/megatron/training/training.py::num_floating_point_operations`，按配置估算语言模型及 MTP 计算量，不包含视觉编码器完整 FLOPs，也不是 profiler 测量。

每步全局 batch=2、序列长度=512；公式结果为 **30,944,347,029,504 FLOPs/global step（30.944347 TFLOPs/step）**。这里是两张卡共同处理一个全局 batch 的总计算量，不应再乘 GPU 数。

| 保存步数 | checkpoint 中累计 FLOPs | 累计 TFLOPs |
| --- | --- | --- |
| 2 | 61,888,694,059,008 | 61.888694 |
| 4 | 123,777,388,118,016 | 123.777388 |
| 6 | 185,666,082,177,024 | 185.666082 |

三份 checkpoint 的 `num_floating_point_operations_so_far` 均与「步数 × 单步公式 FLOPs」完全一致，恢复后累计量连续。

累计 FLOPs 的读取命令见教程第 9 步。上述累计量沿用早期配置长度口径，不能据此推断实际硬件利用率。

## 权重转换验证

- 官方 HF → TP2 Megatron → HF：738 个张量逐一完全相等，最大绝对误差 0，没有缺失或多余权重。
- 训练第 6 步 → HF：成功导出 738 个张量；键、形状完整，所有数值有限。
- 与初始官方权重比较，597 个张量发生更新，其中包括 10 个 MTP 张量。
- 转换器修复了视觉 patch embedding 的 Torch 版本布局选择、release checkpoint 版本号，以及 TE 模块的 `_extra_state` 元数据。HF 权重往返检查只比较权重，不包含 Megatron 专用元数据。

## 回归及严格加载

新增回归测试 17 项全部通过，覆盖真实参数校验中的 MTP + mRoPE、词表 padding、ChatML decoder，以及 checkpoint 版本、视觉布局和 TE 元数据。

最终转换后的 release checkpoint 在两张 GPU 上通过真实 Qwen35Model 的 `strict=True` 加载：两个 rank 均记录 `STRICT_RELEASE_LOAD_OK`，没有缺失键、意外键或非严格加载回退。检查在模型加载完成后退出，没有额外执行训练。

## 本教程数据制作验证

使用原有 5000 条 JSON 和图片，实际重新执行教程中的转换命令（DP1、每 tar 1000 条），生成了 5 个 tar 和 Energon 索引。随后通过 Energon 6 的 `get_val_dataset(..., split_part="train", task_encoder=TaskEncoder())` 完整遍历，读出 5000 条 ChatML 样本，全部图片通过 Pillow 校验。

该补充验证没有重新下载约 27.4 GB 的图片压缩包；下载文件名已对照官方数据仓库核实，教程下载/解压命令通过语法检查。数据转换和读取是实际执行通过的。

## 按教程复现后的产物位置

以下变量由读者在教程中设置，路径均位于自己的运行环境。模型权重和数据产物不应提交到代码仓库。

| 产物 | 路径 |
| --- | --- |
| 原始下载数据 | `$QWEN35_RAW_DATA` |
| 提取的标注和图片 | `$QWEN35_VISION_ROOT` |
| Energon tar 与索引 | `$QWEN35_DATA_PATH` |
| 官方 HF 模型 | `$QWEN35_HF_PATH` |
| 训练及恢复日志 | `$QWEN35_OUTPUT_DIR/logs/host_0_localhost.output` |
| 第 6 步训练 checkpoint | `$QWEN35_OUTPUT_DIR/checkpoints/iter_0000006/` |
| 第 6 步数据游标 | `$QWEN35_OUTPUT_DIR/dataloader/iter_0000006/` |
| 训练后 HF 权重 | `$QWEN35_EXPORT_DIR/model.safetensors` |

这是完整 4B 模型的短程训练链路验证，不代表收敛、长程稳定性或精度评测。未验证视频、其他并行组合、其他硬件、导出后的推理精度，也未比较断点续训与不中断训练的逐步数值等价性。

# Qwen3.5-4B 八卡吞吐验证

2026-09-07，8 张 H100 80GB，TP=2、DP=4、PP=1，BF16。完整 4B 图文模型，辅助 MTP=1；固定 padding 长度 2048，micro batch size=4、global batch size=64，关闭语言模型和视觉重计算。环境及数据制作步骤见 [教程](README.md)。

## 测量方式和结果

每轮训练 15 步，直接读取日志第 6–15 步的 `throughput per GPU (TFLOP/s/GPU)`，未根据耗时重新计算吞吐。显存为同一稳定窗口内每秒采样的 `nvidia-smi memory.used`，取八张卡最大值并将 MiB 转为 GiB；不是全程瞬时峰值保证。

| 设置 | 平均 TFLOP/s/GPU | 中位数 | 最小–最大 | 稳定窗口最高显存 GiB |
| --- | --- | --- | --- | --- |
| 优化前 | 108.01 | 108.25 | 100.3–112.6 | 60.77 |
| 缓存复用 + native CE + DP overlap | 206.68 | 206.85 | 202.3–211.2 | 67.06 |
| 再开启 TP overlap（保留配置） | 202.96 | 202.55 | 199.4–207.2 | 67.15 |

三轮均完成训练及最终 checkpoint 保存。缓存复用、native 交叉熵融合与 DP overlap 的组合将同配置均值从 108.01 提高到 206.68 TFLOP/s/GPU。额外开启 TP overlap 后为 202.96，本次短窗口未显示进一步收益；保留配置覆盖已修复的 TP overlap 执行链路。没有逐项消融或 profiler 结果，不能将组合收益分配给单项优化。

保留配置的逐步日志值：

| iteration | TFLOP/s/GPU |
| --- | --- |
| 6 | 199.4 |
| 7 | 199.8 |
| 8 | 204.0 |
| 9 | 201.9 |
| 10 | 201.1 |
| 11 | 201.6 |
| 12 | 203.2 |
| 13 | 207.2 |
| 14 | 205.8 |
| 15 | 205.6 |

## 代码修复和依赖

- 移除 Qwen3.5 数据读取路径在满长度 microbatch 上调用 `empty_cache()` 的行为，允许缓存分配器复用显存。
- 吞吐配置开启 native 交叉熵融合、梯度归约和参数收集 overlap。native 融合与原交叉熵的 loss/gradient 对比已在八个 rank 上通过，loss 最大绝对差约 9.54e-7；梯度容差 rtol=0.01、atol=1e-5。native 路径仍包含 FP32 计算。
- TP overlap 依赖 TransformerEngine-FL 提交 `12595d06` 或等效修复，修正 CUDA 构造参数转发，并对齐五个其他 vendor 的 Python 接口。CUDA 路径完成此次八卡训练；其他 vendor 仅完成接口回归测试。

## 显存与指标边界

优化后稳定窗口占用约 67.15 GiB，**不满足 64 GiB 的目标**。优化前约 60.77 GiB 的结果不能作为当前配置的显存承诺。64G 设备需要缩小 micro batch size 或序列长度后重新测量；其他国产芯片尚未真机验证。

日志 FLOPs 是框架对语言模型（含 MTP）的公式估算，不包含视觉编码器完整 FLOPs；除以实际图文训练耗时后的指标不能等同于完整多模态模型的硬件 MFU。固定 padding 消除了先前动态长度与公式长度不一致的问题，但该吞吐也不是有效非 padding token 吞吐。

仅保留 [train_throughput.yaml](conf/train_throughput.yaml) 作为八卡性能入口，[train_smoke.yaml](conf/train_smoke.yaml) 用于两卡小规模链路验证。历史对比不再维护单独的 baseline 配置。

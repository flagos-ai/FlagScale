# 天数 BI 训练镜像与公共 CI

## 验收结果

2026-10-30 至 2026-10-31，在 tianshu-n32（10.31.28.32）完成验证：

- 最终候选镜像通过公共镜像构建 pre/post 钩子，两个 smoke rank 均 PASS。
- smoke 覆盖厂商 PyTorch、GPU 矩阵反向传播、两卡 all-reduce、
  TE reference BF16 算子和 AdamW 权重更新。
- 使用公共 `run_unit_tests.sh`，每个 rank 为 **685 passed、1 deselected**，退出码 0。
  原有 VLA 排除和 parallel_config 取消选择保持不变，没有为本次失败新增排除项。
- 两次从头运行的双卡 Qwen3 0.6B 均完成 10 步，loss 有限，
  skipped/NaN 迭代为 0，并保存 iteration-10 checkpoint。
  日志 loss 序列完全一致，最大差值为 0。
- 在最终重建镜像的新容器内，再次通过完整公共单测，以及
  `run_functional_tests.sh --platform iluvatar --device iluvatar --task train --model qwen3 --list 0_6b_iluvatar`。
  公共 golden 比较器为 **1 passed**。
- 本地公共 CI 框架测试：401 项通过；公共运行时脚本测试 75 项通过。

这是双卡正确性验证，未宣称完成 8 卡训练、性能验收或推理验证。
宿主机 `ixsmi` 枚举 16 个 BI-V150 逻辑设备，每个 32 GiB，
驱动/IX-ML 为 5.0.0；本配置映射前 8 个设备，日常测试使用逻辑设备 0、1。

## 已推送训练镜像

```text
harbor.baai.ac.cn/flagos-dev/flagscale-train:manual-20261030-lluvatar-bi-gpu8-dev
sha256:73f105dd02dde974355f29f0cc85382913bba2554a3e8c7d786f5aeb28bc3436
```

Harbor 推送及按 digest 拉取已核对。
正式 `ci_train_image` 固定该 digest；删除临时的 `ci_enabled: false`，
按公共平台发现机制的默认行为启用。没有新增天数专用 workflow。

SDK 基础镜像固定为：

```text
harbor.baai.ac.cn/flagos-base/flagos-base-iluvatar-corex4.5.0:2.3.0
sha256:187f7443fc0a6bf53f9f55cc599e7f95b7d6a9260b0281b927c8acff86a7bfce
```

它只作为 `image_build.tasks.train.base_image`，不能代替训练镜像。
厂商版本参考 [build-infra 配置](https://github.com/flagos-ai/build-infra/blob/main/configs.yaml)
和 [COREX 4.5.0 说明](https://github.com/flagos-ai/build-infra/blob/main/runtime/iluvatar-corex4.5.0.md)。

## 公共框架中的接入位置

- `.github/configs/iluvatar.yml`：平台、Runner、设备、依赖、训练镜像构建配置。
- `docker/iluvatar/Dockerfile.train`：从 SDK 构建 Python 3.12 训练环境，
  调用公共 `tools/install/install.sh --platform iluvatar --task train`。
- `requirements/iluvatar/` 和 `tools/install/iluvatar/`：厂商运行时约束、平台安装实现，
  以及公共镜像 workflow 调用的 pre/post 验证钩子。
- `.github/scripts/set_env_iluvatar.sh`：公共测试环境入口。
- `tests/test_utils/config/platforms/iluvatar.yaml`：两进程单测和已登记的 Qwen3 用例。
- `tests/functional_tests/train/qwen3/conf/0_6b_iluvatar.yaml` 及其 `train/` 子配置：
  已从 examples 移入标准功能测试目录，输出使用公共 test_results 路径。
- `tests/functional_tests/train/qwen3/gold_values/0_6b_iluvatar.json`：
  天数实测基线，rtol=1e-5、atol=1e-8，没有复制 PPU loss。

自动 CI、nightly、评论触发复用公共计划及平台流水线；
手动测试和依赖预热复用已有的公共 workflows。Runner 标签为
`flagscale-lluvatar-bi-gpu8`，平台名称使用官方拼写 `iluvatar`，
分支和镜像标签保留用户指定的 `lluvatar`。

## 两个设备参数

`runtime_device_count: '8'` 检查至少 8 个设备可见，
`runtime_smoke_nproc: '2'` 启动两个 smoke 进程，各使用一个逻辑 GPU。
训练和公共单测也配置两个进程。设为 2 可以减少计算占用，但不提供资源独占；
运行前仍需检查卡 0、1 是否空闲。容器映射 8 个设备不代表同时训练使用 8 个。

## 本次解决的兼容问题

- COREX 的 `torch.__version__` 为 `2.10.0`，发行包 METADATA 为
  `2.10.0+corex.4.5.0.20260804`。安装器和 smoke 同时严格校验两者，
  不接受仅有上游同名版本的 wheel。
- PyTorch/torchaudio、torchvision、Triton 固定厂商 wheel；NumPy 固定 1.26.4。
  训练依赖补齐 pandas 2.2.3、datasets 4.5.0，并使用 s3fs 2025.10.0、
  PyArrow 21.0.0，避免存储依赖冲突以及新版 PyArrow 要求 NumPy 2 的 ABI 问题。
- 安装 `python3-dev`，提供 Megatron 编译数据 helper 所需的 `python3-config`；
  仅安装 `python3.12-dev` 不提供该默认命令。
- TE-FL 使用 `TE_FL_SKIP_CUDA=1` 和 reference/unfused 后端，
  不下载不参与该构建的 NVIDIA 子模块。跳过子模块检查的上游选项仅作用于天数 TE 安装。
  Git HTTP/1.1 和低速超时仅作用于 Docker 源码安装 RUN。
- 公共单测实测发现 KERV 动态前缀优化内核在 64-lane COREX target 上数值错误。
  种子 42 时，优化内核相对 CPU 最大误差 13.828125，
  GPU dense reference 最大误差约 3.05e-5。
  动态前缀优化路径限制在已验证的 32-lane warp，其余复用已有精确参考。
  保留原数值测试和容限，另验证 32/64-lane 分派；静态前缀和 32-lane 路径保持原有行为。

## 数据、源码与验证记录

容器内数据路径沿用公共约定：

```text
/home/gitlab-runner/data/pile_wikipedia_demo/pile_wikipedia_demo.{bin,idx}
/home/gitlab-runner/tokenizers/qwentokenizer
```

n32 宿主机挂载来源为 `/data/flagscale-ci/assets/data` 和 `.../tokenizers`，
只读挂载。官方示例资源来源：

- `https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/pile_wikipedia_demo/`
- `https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/`

文件 SHA-256 记录位于 `/data/flagscale-ci/assets/SHA256SUMS`。
本次源码由公共 source catalog 解析并固定：

- Megatron-LM-FL：`bbc208e00f2a201783938ef549f5058d6047e4c0`
- TransformerEngine-FL：`6df49429794d18109f0d04d924a14e032cdc220b`

服务器独立验证目录为 `/data/flagscale-ci/iluvatar-b399420/FlagScale`。
上一级 `logs/` 保留构建、smoke、公共单测、功能测试及 Harbor 推送日志；
`results/qwen3-valid-first/` 保存首轮有效训练输出。
没有改动已有 Runner 工作区，不把数据、凭据或大日志提交到 Git。

公共 CLI 的外层命令曾在训练子进程失败时返回 0。
后续验收必须同时检查各 rank 错误日志、完整步数、有限 loss、checkpoint 完成标志
及公共 golden 比较结果，不能只看外层退出码。

## 后续更新步骤

1. 修改天数 requirements、安装器或配置，并分步提交；保留厂商版本约束。
2. 使用公共 `build_image.yml` 选择 `platform=iluvatar`、`tasks=train`，
   首先构建和验证候选镜像，不直接覆盖已验收的正式镜像引用。
3. 通过公共单测和 Qwen3 功能回归；若更新源码或配置导致 loss 改变，
   先解释变化并重复实测，再审阅 golden，不能直接放宽阈值。
4. 推送验收通过的镜像，验证 registry digest，再更新 `ci_train_image`。
5. 将分支提交推送 GitHub 后，在现有公共 CI 中确认 Runner 调度、依赖准备和测试结果。
   本次本地与服务器验证不等同于已经执行 GitHub Actions。

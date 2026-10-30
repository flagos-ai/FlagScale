# 天数 BI 训练镜像接入（COREX 4.5.0）

本地初始代码已获用户同意，进入分步提交和 tianshu-n32 验证阶段；镜像尚未验收或推送。
不提供推理镜像，不新增天数专用 workflow。

## 公共框架中的接入位置

- `.github/configs/iluvatar.yml`：Runner、设备映射、依赖和 `image_build.tasks.train`。
- `docker/iluvatar/Dockerfile.train`：继承 SDK 基础镜像，创建 Python 3.12 venv。
- `tools/install/iluvatar/`：公共 `tools/install/install.sh --platform iluvatar --task train`
  的平台安装实现，以及公共镜像 workflow 的 pre/post 验证钩子。
- `requirements/iluvatar/`：训练包与厂商运行时约束，不引用 CUDA 平台依赖。
- `.github/scripts/set_env_iluvatar.sh`：公共测试 workflow 的环境入口。
- `tests/test_utils/config/platforms/iluvatar.yaml`：两进程单测配置；训练功能矩阵待实测基线后登记。
- `tests/functional_tests/train/qwen3/conf/0_6b_iluvatar.yaml`：参考 PPU 的两卡 Qwen3
  0.6B、10 步训练配置，使用 TE-FL reference / unfused attention。

平台标识使用官方拼写 `iluvatar`；分支、Runner 和预定镜像标签保留用户提供的
`lluvatar` 拼写。Runner 为 `flagscale-lluvatar-bi-gpu8`，不使用 PPU 的
`flagcicd-810e`。

`ci_enabled: false` 使未验收平台暂不进入 `CI`、nightly、评论请求的公共测试计划。
已有平台默认启用，保持原有行为。此开关不关闭独立的公共镜像构建入口。
基础镜像只写在 `image_build.tasks.train.base_image`，不能写成 `ci_image`：
公共加载器把 `ci_image` 视为训练镜像的回退值，SDK-only 镜像不能直接跑训练测试。
公共加载器允许构建流程传入候选训练镜像，先验证它，再提升到正式 `ci_train_image`。
原来的 `all_tests.yml` 是静态的手动入口；本次不向它添加尚未验收的平台。
验收后的自动 CI 通过现有 `ci.yml` / `_platform_pipeline.yml` 按配置发现平台。

## 两个设备参数的区别

`runtime_device_count: '8'` 检查容器至少能看到 8 张卡；
`runtime_smoke_nproc: '2'` 启动两个分布式进程，各使用一张逻辑 GPU（当前为 0、1）。
这是双卡通信与训练算子的快速验证，不表示已经验证 8 卡并行。
设备映射仍暴露 8 张卡，SDK 的设备查询和初始化也可能访问其他设备；
因此不能把参数设为 2 理解成资源独占或绝不影响其他任务。
运行前检查占用情况；若 0、1 正忙，先选择空闲卡并一致调整可见设备配置，不能直接抢占。
后续确需整机验收时，单次覆盖进程数为 8，日常默认仍为 2。

## 基础镜像与依赖

基础镜像为：

```text
harbor.baai.ac.cn/flagos-base/flagos-base-iluvatar-corex4.5.0:2.3.0
sha256:187f7443fc0a6bf53f9f55cc599e7f95b7d6a9260b0281b927c8acff86a7bfce
```

Dockerfile 和平台配置固定以上 digest。厂商依赖版本参考
[build-infra 的配置](https://github.com/flagos-ai/build-infra/blob/main/configs.yaml)
和 [COREX 4.5.0 运行环境说明](https://github.com/flagos-ai/build-infra/blob/main/runtime/iluvatar-corex4.5.0.md)，
厂商包来自 `https://resource.flagos.net/repository/flagos-pypi-iluvatar/simple`；
普通 Python 包通过天数镜像专属的 `PIP_INDEX_URL` 构建参数使用阿里云镜像，
不修改其他平台或宿主机的 pip 配置，也不把构建参数写成运行时强制配置：

- Python 3.12 / NumPy 1.26.4。
- PyTorch、torchaudio 2.10.0+corex.4.5.0.20260804。
- torchvision 0.25.0+corex.4.5.0.20260804。
- Triton 3.2.0+corex.4.5.0.20260804。

初始正确性基线不启用 Apex、FlashAttention、原生 TE CUDA 扩展或 FlagGems 优化。
训练栈通过公共 source catalog 解析 Megatron-LM-FL / TE-FL 为完整 commit SHA，
源码安装使用 `--no-deps --no-build-isolation`，Python 依赖在训练 requirements 显式声明。
运行时约束保存在镜像内，后续公共 CI 安装包也不能静默替换厂商 torch/triton。
当前版本与 n32 的实际驱动兼容性仍需实测。

## 2026-10-30 n32 实际预检查记录

- 已经通过堡垒机连接 `ub32` / `10.31.28.32`。
- `ixsmi` 枚举 **16 个 BI-V150 逻辑设备，每个 32 GiB**，宿主机驱动和 IX-ML 均为 5.0.0；
  当前平台配置映射前 8 张卡，smoke 使用其中的逻辑卡 0、1，不要求使用全部 16 张卡。
- 预检查时 `ixsmi` 未发现运行中的 GPU 进程，`docker ps` 为空；Docker 为 24.0.7。
  这些是当时的状态，后续每次测试仍应重新检查占用。
- 固定 digest 的基础镜像已经拉取，基础容器内 `ixsmi` 和
  `cmake 3.31.8+corex.4.5.0.20260804` 检查通过；这不等于厂商 PyTorch/通信/训练已经通过。
- 服务器独立验证目录：`/data/flagscale-ci/iluvatar-b399420/FlagScale`。
  通过 Git bundle 同步本地三笔提交，没有改动已有 Runner 工作区，没有执行 Git push。
- 本次公共 source catalog 解析得到：
  Megatron-LM-FL `bbc208e00f2a201783938ef549f5058d6047e4c0`，
  TE-FL `6df49429794d18109f0d04d924a14e032cdc220b`。
  构建日志和解析结果保存在该目录的上一级 `logs/`、`resolved-sources.json`。
- 首次构建已完成系统依赖及 Python 3.12 环境；COREX 对应的厂商 torch、audio、vision、
  Triton 包均已下载。普通 PyPI 元数据解析较慢，因此只为天数构建增加镜像源参数后重试；
  尚未完成训练镜像构建或双卡 smoke。
- 初始预设的 `/home/gitlab-runner/data` 与 `/home/gitlab-runner/tokenizers` 不存在。
  已从 FlagScale 官方示例资源下载 Wikipedia `.bin`、`.idx` 以及四个 Qwen tokenizer 文件，
  放在 `/data/flagscale-ci/assets/data` 与 `/data/flagscale-ci/assets/tokenizers`。
  下载全部成功，SHA-256 记录在 `/data/flagscale-ci/assets/SHA256SUMS`。
  平台配置只调整宿主机挂载来源，容器内仍保留公共 `/home/gitlab-runner/...` 路径。
  下载成功仍不等于训练数据格式和 tokenizer 兼容性已经验证。

资源来源与仓库 `docs/getting-started.md` 使用相同的官方对象存储：
`https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/pile_wikipedia_demo/`
以及 `https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/`。
保留原始文件名及校验记录，不将大数据文件、账号或密码提交到 Git。

## n32 后续验证步骤

1. 先检查宿主机 `ixsmi`、驱动、Docker 版本、磁盘空间、现有占卡任务；确认待用的两张卡空闲。
   镜像为 linux/amd64，不在本地 Mac 上执行硬件验收。
   核实 `/dev/itrctl` 和 `/dev/iluvatar0` 到 `/dev/iluvatar7` 存在；这是官方 plain Docker
   映射，若 n32 的实际布局不同，只修改天数平台配置。先使用单机通信，不假设 RDMA 设备。
2. 核实 `/home/gitlab-runner/data/pile_wikipedia_demo/pile_wikipedia_demo.{bin,idx}`
   和 `/home/gitlab-runner/tokenizers/qwentokenizer`；以上是容器内路径。
   n32 上已下载的宿主机资源位于 `/data/flagscale-ci/assets/`，按平台配置只读挂载。
3. 在服务器放置审阅后的仓库代码。用交互式 `docker login harbor.baai.ac.cn` 登录；
   不把密码写进脚本、Git、Dockerfile 或构建参数。
4. 优先复用 `.github/workflows/build_image.yml`，输入 `platform=iluvatar`、`tasks=train`、
   `build_images=true`、`run_tests=false`、`push=false`，在天数 Runner 上进行首次候选构建。
   公共流程自动解析 source SHA、执行 pre/post 钩子。首次不运行完整测试，是因为还没有
   n32 的 Qwen3 golden 数据。共享 workflow 后续开启 push 时读取 `REGISTRY_USERNAME`
   和 `CONTAINER_REGISTRY` secrets；本次未设置任何 secrets。
5. 如需先在 n32 终端调试，仍使用同一个 Dockerfile 与验证钩子：

```bash
# 以下命令只在审阅确认后的 n32 仓库根目录执行，要求 Bash 4+、Docker、mikefarah/yq。
export BASE_IMAGE=$(yq -r '.image_build.tasks.train.base_image' .github/configs/iluvatar.yml)
export TRAIN_IMAGE=harbor.baai.ac.cn/flagos-dev/flagscale-train:manual-20261030-lluvatar-bi-gpu8-dev
export MEGATRON_SHA=$(git ls-remote https://github.com/flagos-ai/Megatron-LM-FL.git refs/heads/main | cut -f1)
export TE_SHA=$(git ls-remote https://github.com/flagos-ai/TransformerEngine-FL.git refs/heads/main | cut -f1)
# 保存上述两个 SHA；重试及 golden 复现必须使用同一对 SHA。
export IMAGE_BUILD_TASK=train IMAGE_BUILD_BASE_IMAGE="$BASE_IMAGE"
export IMAGE_BUILD_CANDIDATE_IMAGE="$TRAIN_IMAGE"
export IMAGE_BUILD_RUNTIME_DEVICE_COUNT=8 IMAGE_BUILD_RUNTIME_SMOKE_NPROC=2
IMAGE_BUILD_PHASE=pre bash tools/install/iluvatar/validate_image_build.sh

docker build --target dev -f docker/iluvatar/Dockerfile.train \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg FLAGSCALE_MEGATRON_REF="$MEGATRON_SHA" \
  --build-arg FLAGSCALE_TE_REF="$TE_SHA" \
  --build-arg FLAGSCALE_REVISION="$(git rev-parse HEAD)" \
  -t "$TRAIN_IMAGE" .
IMAGE_BUILD_PHASE=post bash tools/install/iluvatar/validate_image_build.sh
```

pre 钩子从基础镜像起临时容器，检查 SDK、CMake 与 `ixsmi`；不要求基础镜像预装 torch。
post 钩子从候选训练镜像起容器，验证厂商 torch 版本、8 卡可见、2 进程 all-reduce、
矩阵乘法反向传播、TE reference BF16 线性层和 AdamW 权重更新；失败不会继续推送。
不使用 `docker commit`：调试中发现的缺包要回写 requirements 或 Dockerfile 并重新构建。

6. 验证训练：从该候选镜像起容器，挂载仓库到 `/workspace/FlagScale` 及配置中的数据目录，
   使用相同 `container_options`，进入仓库后：

```bash
source tools/install/iluvatar/env.sh
python -m pip install . --no-build-isolation
# 手动验收也提供公共单测 runner 要求的 prepared Megatron 目录。
# Actions 模式下由 prepare_dependencies.yml 自动提供，不需要手动执行这步。
export GITHUB_WORKSPACE=/tmp/flagscale-iluvatar-acceptance
mkdir -p "$GITHUB_WORKSPACE"
python -m pip install --no-deps --no-build-isolation \
  --target "$GITHUB_WORKSPACE/megatron-lm-fl-install" \
  "$FLAGSCALE_HOME/deps/Megatron-LM-FL"
bash tests/test_utils/runners/run_unit_tests.sh --platform iluvatar --device iluvatar
flagscale train qwen3 --config tests/functional_tests/train/qwen3/conf/0_6b_iluvatar.yaml --test
```

保留完整训练日志、loss、设备/驱动信息和两个源码 SHA。检查 loss 有限、训练完整退出，
在相同条件下重复运行，再根据实际稳定性审阅误差容限，生成
`tests/functional_tests/train/qwen3/gold_values/0_6b_iluvatar.json`。
将这两份已验收的 example 配置移动到 `tests/functional_tests/train/qwen3/conf/`
及其 `train/` 子目录，把入口配置的 `exp_dir` 改为
`tests/functional_tests/train/qwen3/test_results/0_6b_iluvatar`。
随后在天数平台配置的 `functional.train` 登记 `qwen3: [0_6b_iluvatar]`，
通过公共 `run_functional_tests.sh` 再次运行并与 golden 比较；不能复制 PPU 的 loss。

7. 单测、双卡 smoke、Qwen3 功能回归通过后，再执行 `docker push "$TRAIN_IMAGE"`，
   记录返回 digest 并确认 Harbor 可拉取。最后将 `ci_train_image` 填成该镜像的已推送
   digest 引用，并删除临时 `ci_enabled: false`，按默认启用行为接入现有自动 CI。构建成功不等于训练验收。
   公共构建入口会生成自己的 SHA 标签；若走该入口，验收后再给同一 digest 添加预定的
   `manual-20261030-lluvatar-bi-gpu8-dev` 标签，不修改公共标签生成逻辑。

## 分步提交安排

1. `ci: support unvalidated platform bootstrap and candidate image overrides`
   ——公共开关、候选镜像加载修复及对应的通用回归测试（按测试函数分组暂存）。
2. `feat(iluvatar): add COREX 4.5 training image and shared CI configuration`
   ——天数配置、Dockerfile、依赖、安装和验证脚本及平台专属测试。
3. `test(iluvatar): prepare Qwen3 training acceptance and document node validation`
   ——待实测训练配置与本文档。
4. 服务器验收后单独提交 golden、已推送训练镜像 digest 和自动 CI 启用变更。

用户已同意提交初始实现并连接服务器继续验证。镜像验收和推送完成前，
`ci_train_image` 保持为空，`ci_enabled` 保持关闭。

## 后续构建发现：厂商 PyTorch 版本字段

第二次构建完成了 Python 依赖、TE-FL 和 Megatron-LM-FL 安装，但在最终版本检查失败。
从失败容器提取并核对，`torch/version.py` 的 `__version__` 为 `2.10.0`、CUDA 为 `10.2`，
而安装包 METADATA 的 Version 为 `2.10.0+corex.4.5.0.20260804`。
安装器和 smoke 因此改为同时严格检查发行包的完整厂商版本，以及运行时基础版本，
不会接受仅有上游 `torch==2.10.0` 的环境。后续仍须通过设备、通信和反向传播验证。
Dockerfile 将公共安装器的 pip 阶段和源码阶段分开缓存，并在末尾设置源码 revision 标签，
避免网络重试时无必要地重复下载已经安装成功的 Python 包。

第三次构建保存了 pip 依赖层，但 CUTLASS 子模块的临时 Git 包连续六分钟未增长。
天数 Dockerfile 的源码 RUN 阶段因此统一使用 HTTP/1.1，并设置
`http.lowSpeedLimit=1024`、`http.lowSpeedTime=120`，将停滞交给现有重试函数处理。
设置通过该 RUN 的局部环境传给 Git 子进程，不写宿主机 Git 配置或镜像运行时 ENV，
不影响其他芯片；重试复用 pip 层，不引入新的源码来源或工作流。

第四次构建与宿主机完整 fetch 均出现 GitHub 下载停滞，尚无证据支持更换构建网络。
reference 模式已设置 `TE_FL_SKIP_CUDA=1`，不构建 NVIDIA CUDA 扩展；
天数安装器因此不递归下载对应子模块，并在 TE 安装调用内使用上游支持的
`NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1`，避免 setup 隐式拉取这些子模块。
该变量只作用于天数 TE 安装，源码 SHA、公共安装器和其他芯片保持不变。

## 第五次候选镜像与公共单测补包

以 `d916227` 构建候选镜像成功，pre/post 验证脚本退出码为 0。
post 验证的 rank 0、1 均输出 `train image validation PASS`，已覆盖厂商
PyTorch、矩阵反向传播、NCCL-compatible all-reduce、TE reference BF16 和 AdamW 更新。
这只是双卡 smoke 结果，尚不能据此启用正式 CI 或推送已验收镜像。

首次公共 `run_unit_tests.sh` 在收集时缺少 pandas；继续核查发现公共数据工具还导入
Hugging Face datasets 和 PyArrow。天数 requirements 补充仓库使用的 pandas 2.2.3、
datasets 4.5.0，并固定 s3fs 2025.10.0 以匹配 datasets 的 fsspec 约束。
补包后的 `pip check` 无冲突，但第二次单测实测发现 PyArrow 26 要求 NumPy 2，
不兼容厂商固定的 NumPy 1.26.4，因此显式固定 PyArrow 21.0.0。
这些修复先在独立验收容器验证，再由同一 Dockerfile 重建最终候选镜像；
不能将手工补包的容器状态直接作为正式训练镜像发布。

## Subsequent n32 acceptance evidence

- Sixth candidate build and both smoke ranks passed with the data dependency fixes.
- Public unit tests: 682 passed, 1 failed, 1 deselected per rank. The remaining
  failure is dynamic-prefix KERV tree attention on COREX's 64-lane CUDA target.
  A seeded comparison found max error 13.828125 for the optimized kernel and
  approximately 3.05e-5 for the existing GPU dense reference against CPU.
  Keep the original test and tolerance; use the existing exact reference for
  unsupported dynamic-prefix warp layouts and verify both 32/64-lane dispatch.
- Qwen3 initially failed because python3.12-dev supplies python3.12-config,
  while Megatron invokes python3-config. Add Ubuntu python3-dev in a final
  small OS layer to preserve completed vendor/source cache layers.
- After installing that package, two fresh two-GPU Qwen3 runs completed all
  10 steps and saved iteration-10 checkpoints. Loss comparison is pending.
- The public CLI can return status 0 while training children failed. Acceptance
  must also inspect rank logs, all ten finite losses and completion markers.

Public unit regression after the warp guard: 685 passed, 1 deselected per
rank, exit 0. Two fresh Qwen3 runs produced identical ten-value loss logs
(max difference 0), verified using the public metric extractor. Register
the measured values at rtol=1e-5, atol=1e-8 and move the accepted configs
into the standard functional train tree. Seventh image build and both
smoke ranks passed; final-image unit and registered functional tests follow.
CI remains disabled until the tested image is pushed and its digest verified.

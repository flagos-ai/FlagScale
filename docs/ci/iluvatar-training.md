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
- `examples/qwen3/conf/0_6b_iluvatar.yaml`：参考 PPU 的两卡 Qwen3
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
包来自 `https://resource.flagos.net/repository/flagos-pypi-iluvatar/simple`：

- Python 3.12 / NumPy 1.26.4。
- PyTorch、torchaudio 2.10.0+corex.4.5.0.20260804。
- torchvision 0.25.0+corex.4.5.0.20260804。
- Triton 3.2.0+corex.4.5.0.20260804。

初始正确性基线不启用 Apex、FlashAttention、原生 TE CUDA 扩展或 FlagGems 优化。
训练栈通过公共 source catalog 解析 Megatron-LM-FL / TE-FL 为完整 commit SHA，
源码安装使用 `--no-deps --no-build-isolation`，Python 依赖在训练 requirements 显式声明。
运行时约束保存在镜像内，后续公共 CI 安装包也不能静默替换厂商 torch/triton。
当前版本与 n32 的实际驱动兼容性仍需实测。

## 审阅确认后的 n32 验证步骤

1. 先检查宿主机 `ixsmi`、驱动、Docker 版本、磁盘空间、现有占卡任务；确认待用的两张卡空闲。
   镜像为 linux/amd64，不在本地 Mac 上执行硬件验收。
   核实 `/dev/itrctl` 和 `/dev/iluvatar0` 到 `/dev/iluvatar7` 存在；这是官方 plain Docker
   映射，若 n32 的实际布局不同，只修改天数平台配置。先使用单机通信，不假设 RDMA 设备。
2. 核实 `/home/gitlab-runner/data/pile_wikipedia_demo/pile_wikipedia_demo.{bin,idx}`
   和 `/home/gitlab-runner/tokenizers/qwentokenizer`，数据与 tokenizer 路径目前是待确认值。
   按服务器实际位置调整天数配置的挂载来源，保持容器内公共路径一致。
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
flagscale train qwen3 --config examples/qwen3/conf/0_6b_iluvatar.yaml --test
```

保留完整训练日志、loss、设备/驱动信息和两个源码 SHA。检查 loss 有限、训练完整退出，
在相同条件下重复运行，再根据实际稳定性审阅误差容限，生成
`tests/functional_tests/train/qwen3/gold_values/0_6b_iluvatar.json`。
将这两份已验收的 example 配置复制到 `tests/functional_tests/train/qwen3/conf/`
及其 `train/` 子目录，把入口配置的 `exp_dir` 改为
`tests/functional_tests/train/qwen3/test_results/0_6b_iluvatar`。
随后在天数平台配置的 `functional.train` 登记 `qwen3: [0_6b_iluvatar]`，
通过公共 `run_functional_tests.sh` 再次运行并与 golden 比较；不能复制 PPU 的 loss。

7. 单测、双卡 smoke、Qwen3 功能回归通过后，再执行 `docker push "$TRAIN_IMAGE"`，
   记录返回 digest 并确认 Harbor 可拉取。最后将 `ci_train_image` 填成该镜像的已推送
   digest 引用，并把 `ci_enabled` 改为 `true`，接入现有自动 CI。构建成功不等于训练验收。
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

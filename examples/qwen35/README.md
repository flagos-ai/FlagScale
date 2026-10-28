# Qwen3.5-4B：从下载数据到图文训练的逐步教程

本教程使用完整的 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)，开启 1 层辅助 MTP 和 mRoPE，在 2 张 H100 80GB 上运行 TP=2、PP=1、BF16、512 token、global batch size 2 的短程图文训练。流程为：准备环境 → 下载模型和数据 → 提取 5000 条图文样本 → 制作 WebDataset 和 Energon 索引 → 检查数据 → 转换权重 → 训练 4 步 → 恢复到第 6 步 → 导出 HF 权重。

所有命令从 FlagScale 仓库根目录执行，后续步骤在同一个 shell 中运行。完整模型短程验证结果见 [验证报告](validation_report.md)。

## 1. 准备训练环境

先进入你自己的 FlagScale 仓库根目录，并激活已安装训练后端的 Python 环境。本文验证环境为 Python 3.12、Torch 2.7.1+cu128，使用 NVIDIA GPU；其他芯片厂商尚未经过本例验证。

本教程所需的 Qwen3.5 适配已随 PR #1284（`Upgrade/megatron v0.18.2`）合入 FlagScale `main`，对应提交为 `b5741d04760a3fcdd33c70524697540b1322a3fe`。本文以该主干提交为代码基准；使用后续版本时，请确认仍包含该适配。还需准备兼容 Megatron-LM-FL v0.18.2 的 CUDA/TransformerEngine-FL 训练环境；不能仅安装 FlagScale CLI 就开始训练。基础环境入口见 [环境说明](../../docs/getting-started.md)，其中旧版本 Megatron 示例不能直接替代本教程验证的后端。在已经可用的训练环境中，补齐本教程用到的 Python 依赖：

```bash
python -m pip install -e . --no-deps
python -m pip install \
  'transformers==5.3.0' 'huggingface-hub==1.6.0' 'tokenizers==0.22.2' \
  'flash-linear-attention==0.4.2' 'fla-core==0.4.2' \
  'megatron-energon==6.0.1' 'webdataset==1.0.2' \
  hydra-core pillow pyyaml tqdm
export PYTHONPATH="$PWD:$PWD/flagscale/train:${PYTHONPATH:-}"
export TE_FL_PREFER=vendor
export OMP_NUM_THREADS=4
```

环境检查（应显示至少 2 张 GPU，所有 import 成功）：

```bash
python - <<'PY'
import torch, transformers
import megatron.core, megatron.energon, transformer_engine, fla, causal_conv1d, flash_attn
print('Torch:', torch.__version__, 'Transformers:', transformers.__version__)
print('Megatron Core:', megatron.core.__file__)
print('Energon:', megatron.energon.__file__)
print('GPU count:', torch.cuda.device_count())
assert torch.cuda.device_count() >= 2
PY
```

## 2. 设置数据、模型和训练输出目录

下面所有目录都在你的环境中创建，不依赖预先准备好的数据或模型。默认以当前仓库下的 `outputs/qwen35-tutorial` 为工作目录。

`QWEN35_WORKDIR` 可修改为容量充足、GPU 节点也能访问的绝对路径。图片原始压缩包约 27.4 GB，即使只使用 5000 条样本，下面的下载方案仍需下载整个压缩包；模型权重约 9.3 GB，还需预留转换、优化器和多次 checkpoint 的空间。

```bash
export QWEN35_WORKDIR="$PWD/outputs/qwen35-tutorial"
export QWEN35_RAW_DATA="$QWEN35_WORKDIR/raw/LLaVA-Pretrain"
export QWEN35_HF_PATH="$QWEN35_WORKDIR/hf/Qwen3.5-4B"
export QWEN35_VISION_ROOT="$QWEN35_WORKDIR/data/llava-first-5k"
export QWEN35_SUBSET_JSON=dataset.json
export QWEN35_WDS_OUTPUT="$QWEN35_WORKDIR/data/energon"
export QWEN35_DATA_PATH="$QWEN35_WDS_OUTPUT/wds-1"
export QWEN35_LOAD="$QWEN35_WORKDIR/checkpoints/release-tp2"
export QWEN35_OUTPUT_DIR="$QWEN35_WORKDIR/train"
mkdir -p "$QWEN35_RAW_DATA" "$QWEN35_HF_PATH" "$QWEN35_VISION_ROOT"
```

训练配置中的 `data_path` 对应 `$QWEN35_DATA_PATH`（Energon 数据目录），`vision_root` 对应 `$QWEN35_VISION_ROOT`（图片根目录）。图片按「图片根目录 / 标注中的相对路径」读取，两者不能互换。

## 3. 下载官方模型和 LLaVA-Pretrain 数据

数据来自 [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main)。需要的文件是 `blip_laion_cc_sbu_558k.json` 和 `images.zip`，无需下载 `*_meta.json`；数据含义和使用条件见其 [数据卡](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)。

在**能访问 Hugging Face 的机器**上执行。如果训练机器不能联网，可以先在联网机器下载，再将模型和数据目录传到训练机器，并按第 2 步设置对应路径。下载前关闭此前可能设置的离线模式：

```bash
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3.5-4B',
    revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    local_dir=os.environ['QWEN35_HF_PATH'],
)
revision = HfApi().dataset_info('liuhaotian/LLaVA-Pretrain').sha
snapshot_download(
    repo_id='liuhaotian/LLaVA-Pretrain', repo_type='dataset', revision=revision,
    allow_patterns=['blip_laion_cc_sbu_558k.json', 'images.zip'],
    local_dir=os.environ['QWEN35_RAW_DATA'],
)
(Path(os.environ['QWEN35_RAW_DATA']) / 'download-revision.txt').write_text(revision + '\n')
print('Dataset revision:', revision)
PY
```

该步骤会下载完整模型分片、config、tokenizer、processor 和 chat template。网络中断后可重跑同一命令，使用已有缓存继续。不要把 Git LFS 指针或未完成的大文件当作模型权重。

## 4. 取前 5000 条标注，只解压所需图片

下面按原 JSON 顺序选取前 5000 条记录，将所需图片解压到统一的图片根目录。兼容压缩包直接包含 `00000/...jpg` 或带 `images/` 前缀的布局，转换后的标注继续保留相对图片路径。

```bash
python - <<'PY'
import json, os, shutil, zipfile
from pathlib import Path
raw = Path(os.environ['QWEN35_RAW_DATA'])
root = Path(os.environ['QWEN35_VISION_ROOT'])
entries = json.loads((raw / 'blip_laion_cc_sbu_558k.json').read_text())[:5000]
assert len(entries) == 5000
with zipfile.ZipFile(raw / 'images.zip') as archive:
    names = set(archive.namelist())
    for entry in entries:
        relative = entry['image']
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        member = next((name for name in (relative, 'images/' + relative) if name in names), None)
        if member is None:
            raise FileNotFoundError(f'Image not in ZIP: {relative}')
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open('wb') as output:
            shutil.copyfileobj(source, output)
(root / os.environ['QWEN35_SUBSET_JSON']).write_text(json.dumps(entries, ensure_ascii=False))
print(f'Ready: {len(entries)} entries in {root}')
print('First entry:', entries[0])
PY
```

标注格式示例：

```json
{
  "id": "004539375",
  "image": "00453/004539375.jpg",
  "conversations": [
    {"from": "human", "value": "Describe the image.\n<image>"},
    {"from": "gpt", "value": "An image caption."}
  ]
}
```

这一步只选取样本和提取图片，不执行 tokenizer，也不把图像转成 embedding。

## 5. 转换 WebDataset，并自动生成 Energon 索引

在训练 Python 环境和 FlagScale 仓库根目录中执行，保留上面设置的路径变量：

```bash
python tools/datasets/qwenvl/convert_custom_dataset_to_wds_chatml_str.py \
  --dataset-root "$QWEN35_VISION_ROOT" \
  --output-root "$QWEN35_WDS_OUTPUT" \
  --json "$QWEN35_SUBSET_JSON" \
  --images-key image \
  --videos-key video \
  --vision-root "$QWEN35_VISION_ROOT" \
  --dp-size 1 \
  --max-samples-per-tar 1000 \
  --train-split 1 --val-split 0 --test-split 0 \
  --num-workers 2
```

预期最后输出 `Dataset successfully converted to wds` 和 `Configurations Generated`。这是纯图像数据，出现 `video not found in the first entry` 提示是正常的。

脚本做两件事：

1. 将 ShareGPT 风格 JSON 转为 ChatML WebDataset。每个样本包含 `.json`（对话和 `second_per_grid_ts`）、`.jpgs`（图片路径列表）、`.videos`（视频路径列表，本例为空）。路径列表由脚本序列化为 pickle，不能手写成文本替代。
2. 调用 Energon 的 `BaseWebdatasetFactory.prepare_dataset`，建立索引和 split，并写入自定义 `ChatMLWebdataset` 的 `dataset.yaml`。**不需要再运行 `energon prepare`，也不需要手工写 YAML。**

2 GPU / TP2 / PP1 / CP1 对应 DP1，所以本教程使用 `--dp-size 1`，输出目录名为 `wds-1`。目录名本身不会设置训练并行度；训练并行度由 YAML 配置指定。

5000 条记录按每 tar 1000 条切分后应得到：

```text
$QWEN35_DATA_PATH/
├── pretrain-0.tar ... pretrain-4.tar
├── pretrain-0.tar.idx ... pretrain-4.tar.idx
└── .nv-meta/
    ├── .info.yaml
    ├── dataset.yaml
    ├── split.yaml
    ├── index.sqlite
    └── index.uuid
```

`dataset.yaml` 应包含：

```yaml
__class__: ChatMLWebdataset
__module__: tools.datasets.qwenvl.data.energon.chatml
field_map:
  conversation: json
  imgs: jpgs
  videos: videos
```

tar 中**不包含图片本体**。训练时仍会读取 `QWEN35_VISION_ROOT/标注中的相对路径`；迁移数据时，需要把图片和 wds 一起复制到训练节点可见的位置。本例只建立 train split，训练配置同时使用 `eval_iters: 0`。

## 6. 训练前检查：真实遍历 Energon 数据和图片

这一步不需要启动分布式训练。使用 Energon 的有限遍历接口读取刚生成的 `train` split，并校验所有图片：

```bash
python - <<'PY'
import os
from pathlib import Path
from PIL import Image
from megatron.energon import TaskEncoder, WorkerConfig, get_val_dataset, get_loader
worker = WorkerConfig(rank=0, world_size=1, num_workers=0)
dataset = get_val_dataset(
    os.environ['QWEN35_DATA_PATH'], worker_config=worker,
    batch_size=None, split_part='train', task_encoder=TaskEncoder(),
)
count = 0
for sample in get_loader(dataset):
    assert sample.conversation['conversations']
    for relative in sample.imgs:
        with Image.open(Path(os.environ['QWEN35_VISION_ROOT']) / relative) as image:
            image.verify()
    count += 1
assert count == 5000, count
print(f'PASS: Energon decoded {count} samples; all images verified')
PY
```

预期输出 `PASS: Energon decoded 5000 samples; all images verified`。这里仅用 `get_val_dataset(..., split_part='train')` 做一次有限遍历，未建立验证集或修改训练 split。

再检查模型分片完整且 processor 可加载：

```bash
python - <<'PY'
import json, os
from pathlib import Path
from safetensors import safe_open
from transformers import AutoProcessor
root = Path(os.environ['QWEN35_HF_PATH'])
index = json.loads((root / 'model.safetensors.index.json').read_text())
for name in sorted(set(index['weight_map'].values())):
    with safe_open(root / name, framework='pt', device='cpu') as weights:
        print(name, len(weights.keys()), 'tensors')
processor = AutoProcessor.from_pretrained(root, local_files_only=True)
print(type(processor).__name__)
PY
```

## 7. 将 HF 模型权重转换成 Megatron TP2 格式

使用第 3 步下载的官方模型权重执行：

```bash
python tools/checkpoint/qwen35/convert_qwen35.py \
  --direction hf2meg \
  --hf-path "$QWEN35_HF_PATH" \
  --meg-path "$QWEN35_LOAD" \
  --yaml examples/qwen35/conf/train/4b.yaml \
  --tp 2 --pp 1
```

预期出现 `release/mp_rank_00/model_optim_rng.pt`、`release/mp_rank_01/model_optim_rng.pt`，以及内容为 `release` 的 `latest_checkpointed_iteration.txt`。此处转换的是**模型权重**，与第 5 步的**训练数据转换**是两个独立步骤。

转换器按当前 Torch 版本选择与视觉模型一致的 Linear/Conv3D patch embedding 布局；跨环境转换可在转换 YAML 中显式指定 `vision_patch_embed_linear`。release checkpoint 使用版本 3.0，避免旧格式 QKV 重排。官方模型的 738 个权重张量包含 MTP，原始词表 embedding 为 248320 行。

## 8. 启动 4 步图文训练，记录 loss 和 TFLOP/s

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN35_TRAIN_ITERS=4
python -m flagscale.cli train qwen35 --config examples/qwen35/conf/train_smoke.yaml --dryrun
python -m flagscale.cli train qwen35 --config examples/qwen35/conf/train_smoke.yaml
```

[train_smoke.yaml](conf/train_smoke.yaml) 读取前面设置的路径，默认使用 GPU 0、1（可通过 `QWEN35_VISIBLE_DEVICES` 指定其他设备），并开启 `train.system.logging.log_throughput: true`。每步记录 LM loss、MTP loss、梯度范数、耗时和 `throughput per GPU (TFLOP/s/GPU)`，每 2 步保存模型、优化器及 dataloader checkpoint。

CLI 启动的训练是异步任务，CLI 返回不能作为训练成功的依据。查看实际日志：

```bash
tail -f "$QWEN35_OUTPUT_DIR/logs/host_0_localhost.output"
```

看到 `iteration 4/4`、第 4 步 `successfully saved checkpoint` 和 `[after training is done]` 后，按 Ctrl-C 退出 `tail`。检查保存进度：

```bash
cat "$QWEN35_OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt"
```

预期为 `4`。首次编译及 checkpoint 开销会影响日志耗时，短程结果不能当作稳定性能基准。TFLOP/s 是框架基于模型结构估算的计算量除以时间和 GPU 数；当前公式覆盖语言模型 GDN、full attention 和 MTP，不包含视觉编码器完整计算量，也不是硬件 profiler 实测 FLOPs。具体数字见 [验证报告](validation_report.md)。

## 9. 从第 4 步恢复到第 6 步

确认上一次任务已结束，再执行：

```bash
export QWEN35_LOAD="$QWEN35_OUTPUT_DIR/checkpoints"
export QWEN35_TRAIN_ITERS=6
python -m flagscale.cli train qwen35 --config examples/qwen35/conf/train_smoke.yaml
```

这里 `6` 是训练总步数，实际新增 2 步。保持相同的输出目录、数据路径、并行配置，配置已启用 `use_checkpoint_opt_param_scheduler: true`，使用保存的 scheduler；不要添加 `finetune`、`no_load_optim` 或 `no_load_rng`。

日志应出现：加载 iteration 4、`restored dataset state`、第 5/6 和 6/6 步、第 6 步保存成功及训练结束。`latest_checkpointed_iteration.txt` 此时应为 `6`。本次已验证状态恢复，但未比较与不中断训练的逐步数值等价性。

保存完成后，可以读取第 6 步 checkpoint 的累计 FLOPs，并使用同一框架公式重算单步计算量：

```bash
python - <<'PYCODE'
import os
from pathlib import Path
import torch
from megatron.training.training import num_floating_point_operations
path = Path(os.environ['QWEN35_OUTPUT_DIR']) / 'checkpoints/iter_0000006/mp_rank_00/model_optim_rng.pt'
checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
args = checkpoint['args']
per_step = num_floating_point_operations(args, args.global_batch_size)
print('Iteration:', checkpoint['iteration'])
print('Estimated FLOPs / global step:', per_step)
print('Cumulative FLOPs:', checkpoint['num_floating_point_operations_so_far'])
PYCODE
```

本次配置下，单步为 `30944347029504` FLOPs，6 步累计为 `185666082177024` FLOPs；均为上文说明的语言模型框架估算口径。

## 10. 导出训练后的 HF 权重

显式指定第 6 步目录，避免多 checkpoint 目录的选择歧义：

```bash
export QWEN35_EXPORT_DIR="$QWEN35_OUTPUT_DIR/exported-hf-step6"
python tools/checkpoint/qwen35/convert_qwen35.py \
  --direction meg2hf \
  --meg-path "$QWEN35_OUTPUT_DIR/checkpoints/iter_0000006" \
  --hf-path "$QWEN35_EXPORT_DIR" \
  --yaml examples/qwen35/conf/train/4b.yaml \
  --tp 2 --pp 1
```

导出器仅生成 `model.safetensors`。如需继续用 Transformers 加载，补齐原模型配置、tokenizer、processor、chat template，不复制原始权重分片及其索引：

```bash
python - <<'PY'
import os, shutil
from pathlib import Path
source = Path(os.environ['QWEN35_HF_PATH'])
target = Path(os.environ['QWEN35_EXPORT_DIR'])
for path in source.iterdir():
    if path.is_file() and path.suffix in {'.json', '.txt', '.jinja', '.model'}:
        if not path.name.endswith('.index.json'):
            shutil.copy2(path, target / path.name)
PY
```

本次完成权重导出和数值完整性检查，导出后推理与精度评测需单独验证。

## 11. 常见问题和扩大数据规模

| 现象 | 检查方式 |
| --- | --- |
| 找不到图片 | 用 `vision_root / JSON中的image` 拼接路径；不能将 `vision_root` 设置为 wds 目录 |
| Energon 找不到数据或 split | `data_path` 应直接指向含 `.nv-meta` 的目录；确认 `split.yaml` 的 train 非空 |
| `av_decode` 参数错误 | 使用 FlagScale `main` 中的 ChatML decoder 和 Energon 6.0.1，不能混用 Energon 7 的 API |
| `No module named tools` / 模型导入错误 | 从仓库根目录执行，并设置本教程的 `PYTHONPATH` |
| 下载失败但本地已有模型 | 检查模型全部分片和配置是否完整，再启用 HF/Transformers 离线模式 |
| MTP 报不支持 mRoPE | 使用已合入 Qwen3.5 适配的 FlagScale `main`，该组合已支持 |

扩大到全部数据时，移除第 4 步的 `[:5000]`，解压全部所需图片，第 6 步将固定数量断言改为实际标注数；使用新的输出目录重新转换。若要划分验证集，应准备足够数量的 tar 分片，调整 train/val 比例和 `eval_iters`，不能只打开评估却不建立 val split。正式训练还需独立设置训练步数、学习率计划、图像分辨率和序列长度；本教程只验证短程链路，不代表完整收敛。

## 12. 8 卡吞吐测试：取第 6–15 步日志

吞吐测试统一使用 **8 张 H100 80GB，TP=2、DP=4、PP=1**。完整模型和辅助 MTP 保持开启，输入固定 padding 到配置长度，关闭 `enable_variable_seq_lengths`，使日志 FLOPs 的长度假设与实际执行长度一致。之前动态长度测试打印的数值不能用来做这组性能比较。

保留一份吞吐配置 `conf/train_throughput.yaml`，小规模链路验证使用前面的 `conf/train_smoke.yaml`。

| 参数 | 吞吐配置 |
| --- | --- |
| GPU 数 / TP / DP / PP | 8 / 2 / 4 / 1 |
| 固定序列长度 | 2048 |
| micro batch size / global batch size | 4 / 64 |
| 每步梯度累积次数 | 4 |
| 语言模型 / 视觉重计算 | 关闭 |
| 交叉熵融合 | native |
| TP overlap / DP 梯度归约与参数收集 overlap | 开启 |
| 训练步数 / 统计窗口 | 15 / 第 6–15 步 |

TP overlap 要求配套 TransformerEngine-FL 正确转发 `use_cublasmp`、`comm_type` 等 overlap 构造参数，且普通 overlap 构造函数不存在位置参数错位；仅更新 FlagScale 不能替代 TE-FL 后端更新。非 CUDA 后端还需确认厂商原生 overlap 实现可用，接口兼容测试不能替代真机验证。

先按第 1–7 步准备环境、数据及 TP2 release 权重，确认 8 张卡空闲，然后执行：

```bash
export QWEN35_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export QWEN35_LOAD="$QWEN35_WORKDIR/checkpoints/release-tp2"
export QWEN35_TRAIN_ITERS=15
unset QWEN35_MICRO_BATCH_SIZE QWEN35_GLOBAL_BATCH_SIZE QWEN35_SEQ_LENGTH
export QWEN35_OUTPUT_DIR="$QWEN35_WORKDIR/throughput"
python -m flagscale.cli train qwen35 --config examples/qwen35/conf/train_throughput.yaml
```

重新测试时使用新的输出目录，避免混合多次运行的日志。

每轮运行期间，可在另一个终端设置同一个 `QWEN35_OUTPUT_DIR` 并采样显存，训练结束后按 Ctrl-C 停止采样：

```bash
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -l 1 \
  > "$QWEN35_OUTPUT_DIR/gpu-memory.csv"
```

`memory.used` 单位为 MiB，除以 1024 得到 GiB。它包含进程整体占用，与 PyTorch 的 allocated/reserved 不同。当前配置在 H100 上稳定窗口实测最高约 **67.15 GiB**，不满足 64 GiB 显存预算。移除逐 microbatch 清空缓存后，缓存复用提高了吞吐，也提高了常驻显存。64G 设备需降低 micro batch size 或序列长度并重新验证；不能直接沿用此前优化前约 61 GiB 的测量。

每轮完成后，直接汇总日志已经打印的吞吐，不重新计算 FLOPs。下面固定取第 6–15 步，缺少任意一步会报错：

```bash
python - <<'PY'
import os, re, statistics
from pathlib import Path
path = Path(os.environ['QWEN35_OUTPUT_DIR']) / 'logs/host_0_localhost.output'
pattern = r'iteration\s+(\d+)/\s+\d+\s*\|[^\n]*?throughput per GPU \(TFLOP/s/GPU\): ([\d.]+)'
values = {int(step): float(value) for step, value in re.findall(pattern, path.read_text())}
assert all(step in values for step in range(6, 16)), sorted(values)
window = [values[step] for step in range(6, 16)]
for step in range(6, 16):
    print(f'iter {step}: {values[step]} TFLOP/s/GPU')
print('mean:', statistics.mean(window))
print('median:', statistics.median(window))
print('min:', min(window), 'max:', max(window))
PY
```

稳定窗口内不保存 checkpoint（`save_interval: 1000`），第 15 步结束后仍会保存最终训练状态。比较结果见 [8 卡吞吐报告](throughput_report.md)。日志吞吐仍是框架语言模型计算量估算除以实际耗时，并非硬件计数器读数；固定长度消除了动态长度造成的口径不一致，但该指标不包含视觉编码器完整 FLOPs，也不等同于有效非 padding token 吞吐。

## 13. 基础多模态 packing

Qwen3.5 的 Energon 数据路径支持把多条完整样本拼接为一个训练序列。配置
`train.system.packing_buffer_size` 为正整数即可开启，默认 `0` 关闭。例如：

```bash
export QWEN35_DATA_PATH=/path/to/energon/wds
export QWEN35_VISION_ROOT=/path/to/images
export QWEN35_HF_PATH=/path/to/tokenizer
python run.py --config-path examples/qwen35/conf --config-name train_packing
```

`conf/train_packing.yaml` 继承完整 4B 模型配置（包括视觉编码器和 MTP），示例使用
8 卡、TP4、PP1、CP1。模型权重、训练步数、输出目录按前文设置；默认配置不指定预训练权重。

- `packing_buffer_size` 的单位是**已编码样本数**，表示每个 Energon worker 一次用于打包的
  缓冲样本数量，不是 token 数、batch size 或全局 DP 缓冲大小。调大它会增加主机内存需求。
- 数据先按 DP rank 和 worker 划分，再在各自的数据流中独立 packing，不跨 DP 收集或重分配。
- 默认采用顺序 **Next Fit**：保持输入顺序；下一条样本放不下时结束当前包、另起一包；
  缓冲区末尾的不足一包也会输出。没有长度排序、BFD、代价模型、solver 或负载均衡。
- 每包上限为 `max_padding_length`，须与 `seq_length` 相等。超过上限或空样本会记录警告并跳过；
  不切断视觉 token 区间。请按数据长度分布设置上限，检查日志中的跳过情况。
  这是与原非 packing 路径有意保留的差异：原路径在 batch 阶段截断超长样本，packing 路径
  选择整条跳过，以避免截断视觉 span 后 token、patch 和 grid 不再对齐。
- 当前支持 `micro_batch_size=1`、`context_parallel_size=1`、`pipeline_model_parallel_size=1`；
  TP、DP 可以大于 1。一个 microbatch 包含一个打包后的序列，`global_batch_size` 统计包数，
  原始样本数随长度变化，不能直接沿用未 packing 的样本数口径。

拼接时保留图像/视频 patch、grid、视觉 mask 的对应顺序，通过 `cu_seqlens` 将原始样本边界
传给语言模型的 attention、GDN 和 MTP；mRoPE 在各样本内部独立计算。已有标签已完成 next-token
shift，因此只屏蔽每条样本末尾的标签，不再次 shift，也不屏蔽下一条样本的首个有效标签。

保留原有首批容量 padding 和 TP 对齐 padding；尾部 padding 属于最后一段、loss 为零。
`cu_seqlens` 描述整个物理 THD 缓冲区，最后一个边界等于包含尾部 padding 的序列长度，
中间没有另行插入的对齐空洞。因此当前 CP1 路径不设置 `cu_seqlens_q_padded` /
`cu_seqlens_kv_padded`：attention 和 GDN 在这些字段为 `None` 时使用 `cu_seqlens_q/kv`，
MTP 也在同一组边界内执行位移。最后一段的真实 token 位于 padding 之前，因果计算不读取
其后的 padding；对应 padding 标签和 MTP loss mask 均被忽略。CP>1 仍被参数检查拒绝。

打包分支的 mRoPE delta 形状为 `[1, 1]`，含义是整个打包行的最大位置加一，减去该行的
物理长度（包括尾部 padding），并非每条原始样本各有一个 delta。当前训练入口只使用
显式 `position_ids`，丢弃 delta；此接口未支持打包样本的独立生成或 KV-cache 续接。

`packing_buffer_size=0` 使用原来的非 packing 路径。验证集继续逐样本评估，并根据实际 batch
元数据选择普通路径。不同 attention 后端可能内部补齐后计算，启用该功能不等于启用了原生
varlen kernel，也不保证具体吞吐收益。

在已安装训练依赖的环境中，从 FlagScale 根目录运行回归测试：

```bash
PYTHONPATH="$PWD:$PWD/flagscale/train:$PWD/flagscale/train/megatron:${PYTHONPATH:-}" \
  python -m pytest -q tests/unit_tests/train/megatron/test_qwen35_packing.py
```

测试覆盖顺序打包、尾包/超长样本、图文与视频对齐、标签边界、padding、分段 mRoPE、MTP 位移、
非法配置，以及启用训练 packing 时未打包验证 batch 的回退路径。接收端用例明确使用非 TP0
rank 和 `data=None`；另有整包 delta 的数值断言，以及带尾部 padding 的 MTP 标签验证。
在 TE-FL 环境中，还会直接比较参考 attention 的打包结果与逐样本计算的输出和梯度，确认
尾部 padding 不影响真实 token、且其梯度为零；没有此可选后端的环境会明确跳过该项。

提供可复现的设备端冒烟入口：
`tests/functional_tests/train/qwen35/packing_smoke.py`，配置为 `conf/train_packing_smoke.yaml`。
它复用完整 4B 模型的正式训练流程，保留视觉编码器和一层 MTP，使用随机初始化，不读取或
写入 checkpoint。入口增加断言检查真实 TP 广播、packed 参数、有限 loss、有限梯度范数及
每步优化器成功更新，不修改模型计算或打包策略。

先按当前平台配置训练运行时，并设置上面的数据、图片根目录和 tokenizer 路径。数据中应包含
能够共同装入一个包的较短样本，否则冒烟会因未观察到多样本打包而失败。从 FlagScale 根目录
运行单节点 8 卡冒烟：

```bash
export QWEN35_OUTPUT_DIR=/path/to/new/packing-smoke-output
python run.py --config-path examples/qwen35/conf --config-name train_packing_smoke
```

四节点、每节点 8 卡时，准备各节点可访问的 hostfile 和一致的训练环境，然后执行：

```bash
export PACKING_HOSTFILE=/path/to/hostfile
export PACKING_MASTER_ADDR=192.0.2.1  # replace with the actual master address
python run.py --config-path examples/qwen35/conf --config-name train_packing_smoke \
  experiment.runner.nnodes=4 \
  +experiment.runner.hostfile="$PACKING_HOSTFILE" \
  +experiment.runner.master_addr="$PACKING_MASTER_ADDR" \
  +experiment.runner.master_port=29644 \
  train.model.global_batch_size=32
```

如使用多节点容器，应通过已有的容器启动方式执行 `action=dryrun` 生成的各节点命令。
设备和网络相关变量按平台设置，示例配置不硬编码某一厂商环境。

每个 rank 应输出 `PACKING_TP_BROADCAST_OK`，表示真实通信已覆盖 packed 和 unpacked 两种
batch，后者用于验证“训练 packing 已开启、验证 batch 未打包”的接收端路径。训练完成时
应输出 `PACKING_SMOKE_COMPLETE`，其中 `optimizer_steps` 等于 `train_iters`，
`multi_sample_forwards` 大于零；同时检查所有节点退出码为 0。断言失败会使进程失败，
仅启动训练进程不算通过。

2026-10-15 使用上述功能测试入口完成四节点 MetaX C550（32 卡）图文冒烟：
TP2 / PP1 / CP1 / DP16、MBS1、GBS32、序列长度 4096、packing buffer size 32，
连续完成 5 步训练。32 个 rank 均通过 packed/unpacked TP 广播检查，且每个 rank 完成
10 次模型前向和 5 次优化器更新，梯度范数均有限。四节点退出码均为 0，无 NaN 或跳过
更新；LM loss 从 12.8767 降至 10.9768。32 项回归测试全部通过，无跳过项。
该结果使用 FineVision 图文数据；视频仅有单元测试覆盖，尚未验证真实视频的分布式训练。

该次 MetaX 环境的版本如下，不能将它等同于其他依赖版本或平台的验证结果：

| Component | Tested version / source revision |
| --- | --- |
| PyTorch | `2.10.0+metax3.8.1.0` |
| Megatron-LM-FL | package `0.18.2`, source `c8fa61f2e403f490baf4cf43fbad24a122f7225a` |
| Energon | `6.0.1` |
| TransformerEngine-FL | package `2.17.0`, source `247bdb33d8047630aae9afc249522ef31886393e` |
| FlagGems | `4af1e434eff5adbf448890089cc3e75ba15a9ff5` |
| FLA | `0.5.0+metax3.8.1.0torch2.10` |
| FlashAttention package | `2.6.3+metax3.8.1.0torch2.10` |

MetaX 运行时沿用 packing 之前已有的本地 JIT 策略、chunked cross entropy 和 TE vendor
加载路径调整，并设置 `CE_TOKEN_CHUNK_SIZE=256`；因此上述 revision 是基础版本而非干净
checkout 的声明。这些调整不是本 PR 新增的 packing 依赖改动。测试实际使用 TE 的
`reference.torch` attention 回退路径，不代表原生 varlen 内核性能或其他后端也已验证。
packing 接口继续复用依赖已有能力，本 PR 不修改 Megatron-LM-FL、TransformerEngine-FL 或 FlagGems。

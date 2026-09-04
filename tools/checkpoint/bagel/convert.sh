#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/share/project/zhaoyingli/bagel-fs/FlagScale/
MEGATRON_ROOT=/share/project/zhaoyingli/bagel-fs/FlagScale/flagscale/train
# export PYTHONPATH=/share/project/zhaoyingli/bagel-fs/FlagScale:/share/project/zhaoyingli/bagel-fs/FlagScale/flagscale/train:${PYTHONPATH}
CONDA_ENV=/share/project/zhaoyingli/envs/bagel-fs-train

BAGEL_SOURCE_CHECKPOINT=${BAGEL_SOURCE_CHECKPOINT:-/share/project/zhaoyingli/checkpoints/ByteDance-Seed/BAGEL-7B-MoT}
BAGEL_OUTPUT_CHECKPOINT=${BAGEL_OUTPUT_CHECKPOINT:-/share/project/zhaoyingli/checkpoints/ByteDance-Seed/BAGEL-7B-MoT-fsdp-dtensor}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

cd "${REPO_ROOT}"
source /root/miniconda3/bin/activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"


# --standalone \
torchrun \
  --nnodes 1 \
  --node-rank 0 \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --master_addr "job-095e0f64-a4ad-47fc-9fc5-10460ccf24f1-master-0" \
  --master_port 23456 \
  tools/checkpoint/bagel/convert_to_fsdp_dtensor.py \
  --bagel-source-checkpoint "${BAGEL_SOURCE_CHECKPOINT}" \
  --save "${BAGEL_OUTPUT_CHECKPOINT}"

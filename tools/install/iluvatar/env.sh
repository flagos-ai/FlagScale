#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

# Python 3.12 is created in the training image, never in the SDK-only base.
export COREX_ROOT=/usr/local/corex
export PATH="${FLAGSCALE_HOME:-/opt/flagscale}/venv/bin:$COREX_ROOT/bin:/opt/cmake-corex/bin:$PATH"
export LD_LIBRARY_PATH="$COREX_ROOT/lib:${LD_LIBRARY_PATH:-}"
export FLAGSCALE_TEST_DIST_BACKEND=nccl
export FLAGSCALE_TEST_TORCH_DEVICE_TYPE=cuda
export TE_FL_SKIP_CUDA=1
export TE_FL_PREFER=reference
export NVTE_FRAMEWORK=pytorch
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=1
export NVTE_WITH_NCCL_EP=0

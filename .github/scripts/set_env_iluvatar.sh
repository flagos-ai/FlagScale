#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail
CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CI_SETUP_DIR/set_env_common.sh"
source "$CI_SETUP_DIR/../../tools/install/iluvatar/env.sh"
ci_activate_python_environment
# Persist for subsequent Actions steps (profile scripts are not always sourced).
for name in PATH LD_LIBRARY_PATH COREX_ROOT FLAGSCALE_TEST_DIST_BACKEND \
    FLAGSCALE_TEST_TORCH_DEVICE_TYPE TE_FL_SKIP_CUDA TE_FL_PREFER NVTE_FRAMEWORK \
    NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN NVTE_WITH_NCCL_EP; do
    ci_export_env "$name" "${!name}"
done
echo "Iluvatar training environment setup complete"

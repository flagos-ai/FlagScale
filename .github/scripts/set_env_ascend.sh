#!/usr/bin/env bash

set -euo pipefail

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CI_SETUP_DIR/set_env_common.sh"

echo "Setting up Ascend NPU environment"

ci_activate_python_environment

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "::warning::npu-smi not found, skipping Ascend validation"
else
  npu-smi info || echo "::warning::npu-smi info returned non-zero"
fi

if [ -n "${CI_NPROC_PER_NODE:-}" ]; then
  "$CI_PYTHON_BIN" - <<'PY'
import os
import sys

try:
    import torch
    import torch_npu
    available = torch.npu.device_count()
    required = int(os.environ.get("CI_NPROC_PER_NODE", "0"))
    print(f"Ascend NPU devices available: {available}, required: {required}")
    if available < required:
        print(f"::error::Not enough devices: available={available}, required={required}", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"::warning::Could not verify device count: {e}")
PY
fi

echo "Ascend NPU environment setup complete"

#!/usr/bin/env bash

set -euo pipefail

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CI_SETUP_DIR/set_env_common.sh"

echo "Setting up MThreads MUSA environment"

ci_activate_python_environment

if ! command -v musa-smi >/dev/null 2>&1; then
  echo "::warning::musa-smi not found, skipping MUSA validation"
else
  musa-smi || echo "::warning::musa-smi returned non-zero"
fi

if [ -n "${CI_NPROC_PER_NODE:-}" ]; then
  "$CI_PYTHON_BIN" - <<'PY'
import os
import sys

try:
    import torch
    import torch_musa
    available = torch.musa.device_count()
    required = int(os.environ.get("CI_NPROC_PER_NODE", "0"))
    print(f"MUSA devices available: {available}, required: {required}")
    if available < required:
        print(f"::error::Not enough devices: available={available}, required={required}", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"::warning::Could not verify device count: {e}")
PY
fi

echo "MThreads MUSA environment setup complete"

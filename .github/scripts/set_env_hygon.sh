#!/usr/bin/env bash

set -euo pipefail

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CI_SETUP_DIR/set_env_common.sh"

echo "Setting up Hygon DCU environment"

ci_activate_python_environment

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "::warning::rocm-smi not found, skipping Hygon validation"
else
  rocm-smi || echo "::warning::rocm-smi returned non-zero"
fi

if [ -n "${CI_NPROC_PER_NODE:-}" ]; then
  "$CI_PYTHON_BIN" - <<'PY'
import os
import sys

try:
    import torch
    available = torch.cuda.device_count() if hasattr(torch.cuda, 'device_count') else 0
    required = int(os.environ.get("CI_NPROC_PER_NODE", "0"))
    print(f"Hygon DCU devices available: {available}, required: {required}")
    if available < required:
        print(f"::error::Not enough devices: available={available}, required={required}", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"::warning::Could not verify device count: {e}")
PY
fi

echo "Hygon DCU environment setup complete"

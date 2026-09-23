#!/usr/bin/env bash

set -euo pipefail

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CI_SETUP_DIR/set_env_common.sh"

echo "Setting up MetaX C550 environment"

ci_activate_python_environment

if ! command -v maca-check >/dev/null 2>&1; then
  echo "::warning::maca-check not found, skipping MACA validation"
else
  maca-check || echo "::warning::maca-check returned non-zero"
fi

if [ -n "${CI_NPROC_PER_NODE:-}" ]; then
  "$CI_PYTHON_BIN" - <<'PY'
import os
import sys

try:
    import torch_maca
    available = torch_maca.maca.device_count()
    required = int(os.environ.get("CI_NPROC_PER_NODE", "0"))
    print(f"MetaX devices available: {available}, required: {required}")
    if available < required:
        print(f"::error::Not enough devices: available={available}, required={required}", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"::warning::Could not verify device count: {e}")
PY
fi

echo "MetaX C550 environment setup complete"

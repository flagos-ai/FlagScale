#!/usr/bin/env bash

set -euo pipefail

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CI_SETUP_DIR/set_env_common.sh"

echo "Setting up Enflame GCU environment"

ci_activate_python_environment

if ! command -v efml-smi >/dev/null 2>&1; then
  echo "::warning::efml-smi not found, skipping Enflame validation"
else
  efml-smi || echo "::warning::efml-smi returned non-zero"
fi

echo "Enflame GCU environment setup complete"

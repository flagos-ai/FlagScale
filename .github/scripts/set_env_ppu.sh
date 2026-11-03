#!/usr/bin/env bash

set -euo pipefail

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CI_SETUP_DIR/set_env_common.sh"

echo "Setting up PPU environment"
ci_activate_python_environment
echo "PPU environment setup complete"

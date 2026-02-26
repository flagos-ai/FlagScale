#!/bin/bash
# Source dependencies for hetero_train task (same as train)
# Delegates to install_train.sh since both tasks use Megatron-LM-FL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/install_train.sh" "$@"

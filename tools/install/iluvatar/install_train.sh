#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/retry_utils.sh"
source "$SCRIPT_DIR/env.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
deps="${FLAGSCALE_DEPS:-${FLAGSCALE_HOME:-/opt/flagscale}/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/iluvatar/train.txt"
pip_cmd=$(get_pip_cmd)

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

# Enforce the same pins when the common installer skips the base phase.
# The Docker image also persists these constraints for shared CI installs.
export PIP_CONSTRAINT="$PROJECT_ROOT/requirements/iluvatar/constraints.txt${PIP_CONSTRAINT:+ $PIP_CONSTRAINT}"
packages=$(get_pip_deps_for_requirements "$REQ_FILE")
if is_phase_enabled task; then
    set_step "Installing Iluvatar train requirements"
    retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT"
elif [ -n "$packages" ]; then
    run_cmd -d "$DEBUG" "$pip_cmd" install --root-user-action=ignore $packages
fi

install_source() {
    local package=$1 revision=$2
    shift 2
    [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || die "Expected resolved SHA for $package: $revision"
    run_cmd -d "$DEBUG" mkdir -p "$deps"
    retry_git_checkout_ref -d "$DEBUG" "$@" \
        "https://github.com/flagos-ai/$package.git" "$revision" "$deps/$package" "$RETRY_COUNT"
    retry -d "$DEBUG" "$RETRY_COUNT" \
        "$pip_cmd install --no-build-isolation --no-deps '$deps/$package'"
}

installed_te=false
installed_megatron=false
if should_install_src task transformer-engine; then
    install_source TransformerEngine-FL "${FLAGSCALE_TE_REF:-}" --recursive
    installed_te=true
fi
if should_install_src task megatron-lm; then
    install_source Megatron-LM-FL "${FLAGSCALE_MEGATRON_REF:-}"
    installed_megatron=true
fi
[ "$DEBUG" = true ] && exit 0
# COREX torch exposes CUDA-compatible device and distributed APIs.
# Native driver/collective compatibility is checked on-node after the build.
if [ "$installed_te" = true ] && [ "$installed_megatron" = true ]; then
python -c '
import torch
import torch.distributed as dist
import transformer_engine.pytorch
from megatron.core.models.gpt import GPTModel

assert dist.is_nccl_available()
assert torch.__version__ == "2.10.0+corex.4.5.0.20260804", torch.__version__
print("COREX torch:", torch.__version__)
'
fi
log_success "Iluvatar training runtime ready"

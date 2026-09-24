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
REQ_FILE="$PROJECT_ROOT/requirements/ppu/train.txt"
pip_cmd=$(get_pip_cmd)

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

# Pin the installed vendor runtime while resolving the training requirements.
# This phase must also work when the common installer skips the base phase.
packages=$(get_pip_deps_for_requirements "$REQ_FILE")
if [ "$DEBUG" != true ] && { is_phase_enabled task || [ -n "$packages" ]; }; then
    python -c 'import torch; print("Vendor torch:", torch.__version__, torch.__file__)'
    constraints=$(mktemp)
    trap 'rm -f "$constraints"' EXIT
    python - <<'PY' > "$constraints"
import importlib.metadata as md
for dist in md.distributions():
    name = dist.metadata.get("Name", "")
    if name.lower().replace("_", "-").startswith(("torch", "triton", "flagcx")):
        print(f"{name}=={dist.version}")
PY
    export PIP_CONSTRAINT="$constraints${PIP_CONSTRAINT:+ $PIP_CONSTRAINT}"
fi
if is_phase_enabled task; then
    set_step "Installing PPU train requirements"
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
# The vendor torch exposes PPU through CUDA-compatible APIs and routes the
# PyTorch NCCL backend to PCCL. Never replace it with a public torch wheel.
if [ "$installed_te" = true ] && [ "$installed_megatron" = true ]; then
python -c '
import torch
import torch.distributed as dist
import transformer_engine.pytorch
from megatron.core.models.gpt import GPTModel

assert dist.is_nccl_available()
print("Vendor NCCL/PCCL:", torch.cuda.nccl.version())
'
fi
log_success "PPU training runtime ready"

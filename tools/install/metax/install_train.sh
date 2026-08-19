#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

# The validated MetaX base image owns the vendor Torch and native MetaX TE
# libraries. Install the pinned FlagOS Megatron and TE Python layers without
# replacing the vendor runtime or compiling CUDA extensions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/metax/train.txt"
MEGATRON_REPO="${FLAGSCALE_MEGATRON_REPO:-https://github.com/flagos-ai/Megatron-LM-FL.git}"
MEGATRON_REF="${FLAGSCALE_MEGATRON_REF:-main}"
TE_REPO="${FLAGSCALE_TE_REPO:-https://github.com/flagos-ai/TransformerEngine-FL.git}"
TE_REF="${FLAGSCALE_TE_REF:-main}"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

checkout_pinned_ref() {
    local repo=$1
    local ref=$2
    local target=$3

    [ -z "$ref" ] && { log_error "A pinned git ref is required"; return 1; }
    retry -d "$DEBUG" "$RETRY_COUNT" "rm -rf '$target' && \
        git init -q '$target' && \
        git -C '$target' remote add origin '$repo' && \
        git -c http.version=HTTP/1.1 -C '$target' fetch --depth 1 origin '$ref' && \
        git -C '$target' checkout -q --detach FETCH_HEAD"
}

install_training_runtime() {
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    mkdir -p "$FLAGSCALE_DEPS"

    set_step "Installing pinned Megatron-LM-FL"
    checkout_pinned_ref "$MEGATRON_REPO" "$MEGATRON_REF" \
        "$FLAGSCALE_DEPS/Megatron-LM-FL" || return 1
    run_cmd -d "$DEBUG" bash -c \
        "cd '$FLAGSCALE_DEPS/Megatron-LM-FL' && $pip_cmd install \
        --root-user-action=ignore --no-build-isolation --no-deps ." || return 1

    set_step "Installing pinned TransformerEngine-FL"
    checkout_pinned_ref "$TE_REPO" "$TE_REF" \
        "$FLAGSCALE_DEPS/TransformerEngine-FL" || return 1
    retry -d "$DEBUG" "$RETRY_COUNT" \
        "git -c http.version=HTTP/1.1 -C '$FLAGSCALE_DEPS/TransformerEngine-FL' \
        submodule sync --recursive && \
        git -c http.version=HTTP/1.1 -C '$FLAGSCALE_DEPS/TransformerEngine-FL' \
        submodule update --init --recursive --depth 1 \
        --recommend-shallow --jobs 1" || return 1
    run_cmd -d "$DEBUG" bash -c \
        "cd '$FLAGSCALE_DEPS/TransformerEngine-FL' && \
        TE_FL_SKIP_CUDA='${TE_FL_SKIP_CUDA:-1}' \
        NVTE_WITH_MACA='${NVTE_WITH_MACA:-1}' \
        $pip_cmd install --root-user-action=ignore \
        --no-build-isolation --no-deps ." || return 1
}

validate_runtime() {
    [ "$DEBUG" = true ] && return 0
    python - <<'PY'
import torch
import transformer_engine
from transformer_engine.pytorch import DotProductAttention, LayerNormLinear
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt import GPTModel

assert HAVE_TE, "Megatron-LM-FL did not detect TransformerEngine-FL"
assert TESpecProvider is not None, "TransformerEngine spec provider is unavailable"
print("torch:", torch.__version__)
print("transformer_engine:", transformer_engine.__file__)
print("TE modules:", DotProductAttention, LayerNormLinear)
print("megatron GPTModel:", GPTModel)
PY
}

main() {
    set_step "Installing MetaX train requirements"
    retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || die "MetaX train pip failed"
    install_training_runtime || die "MetaX training runtime install failed"
    validate_runtime || die "MetaX train runtime validation failed"
    log_success "MetaX train runtime ready"
}

main

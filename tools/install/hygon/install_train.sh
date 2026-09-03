#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

# The DTK base image owns Torch and the device runtime. This installer adds
# only the pinned FlagOS training stack and must not replace vendor Torch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/hygon/train.txt"
MEGATRON_REF="${FLAGSCALE_MEGATRON_REF:?FLAGSCALE_MEGATRON_REF is required}"
FLAGGEMS_REF="${FLAGSCALE_FLAGGEMS_REF:?FLAGSCALE_FLAGGEMS_REF is required}"
TRANSFORMER_ENGINE_REF="${FLAGSCALE_TRANSFORMER_ENGINE_REF:?FLAGSCALE_TRANSFORMER_ENGINE_REF is required}"

fetch_source() {
    local repository=$1 ref=$2 target=$3 archive=$4

    retry -d "$DEBUG" "$RETRY_COUNT" "rm -rf '$target' '$archive' && \
        mkdir -p '$target' && \
        curl --http1.1 --fail --location --retry 5 --retry-all-errors \
          --connect-timeout 30 --output '$archive' \
          'https://codeload.github.com/flagos-ai/$repository/tar.gz/$ref' && \
        tar -xzf '$archive' --strip-components=1 -C '$target' && \
        rm -f '$archive'"
}

install_requirements() {
    local pip_cmd
    pip_cmd=$(get_pip_cmd)

    set_step "Installing Hygon training build requirements"
    retry -d "$DEBUG" "$RETRY_COUNT" "$pip_cmd install --root-user-action=ignore \
        'setuptools>=64.0,<77.0' 'setuptools-scm>=8.0,<10.0' \
        'wheel==0.46.2' 'cmake>=3.21,<4.0' 'ninja==1.13.0' \
        'scikit-build-core==0.12.2' 'pybind11[global]==3.0.3'" || return 1

    set_step "Installing Hygon train requirements"
    retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT"
}

install_source() {
    local repository=$1 ref=$2 target=$3 archive=$4 env_prefix=${5:-}
    local pip_options=${6:-}
    local pip_cmd
    pip_cmd=$(get_pip_cmd)

    set_step "Installing $repository at $ref"
    fetch_source "$repository" "$ref" "$target" "$archive" || return 1
    run_cmd -d "$DEBUG" bash -c "cd '$target' && $env_prefix \
        $pip_cmd install --root-user-action=ignore \
        --no-build-isolation --no-deps $pip_options -e ."
}

validate_runtime() {
    [ "$DEBUG" = true ] && return 0
    python - <<'PY'
from importlib.metadata import distribution

for package in ("flag_gems", "megatron-core", "transformer-engine"):
    distribution(package)
PY
}

main() {
    install_requirements || die "Hygon train requirements installation failed"
    mkdir -p "$FLAGSCALE_DEPS"
    install_source Megatron-LM-FL "$MEGATRON_REF" \
        "$FLAGSCALE_DEPS/Megatron-LM-FL" /tmp/hygon-megatron-lm-fl.tar.gz \
        "" "--ignore-requires-python" || \
        die "Hygon Megatron-LM-FL installation failed"
    install_source FlagGems "$FLAGGEMS_REF" \
        "$FLAGSCALE_DEPS/FlagGems" /tmp/hygon-flaggems.tar.gz || \
        die "Hygon FlagGems installation failed"
    install_source TransformerEngine-FL "$TRANSFORMER_ENGINE_REF" \
        "$FLAGSCALE_DEPS/TransformerEngine-FL" \
        /tmp/hygon-transformer-engine-fl.tar.gz "TE_FL_SKIP_CUDA=1" || \
        die "Hygon TransformerEngine-FL installation failed"
    validate_runtime || die "Hygon training runtime validation failed"
    log_success "Hygon training runtime ready"
}

main

#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

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
REQ_FILE="$PROJECT_ROOT/requirements/hygon/inference.txt"
FLAGGEMS_REF="${FLAGSCALE_FLAGGEMS_REF:-62d70b9e858ec407572153ee8cdf65cc24a637d5}"
VLLM_PLUGIN_FL_REF="${FLAGSCALE_VLLM_PLUGIN_FL_REF:-ffa2ee3eb3831f3873dd0966d12fc8e0b4e6e3d4}"

fetch_source() {
    local repository=$1
    local ref=$2
    local target=$3
    local archive=$4

    retry -d "$DEBUG" "$RETRY_COUNT" "rm -rf '$target' '$archive' && \
        mkdir -p '$target' && \
        curl --http1.1 --fail --location --retry 5 --retry-all-errors \
          --connect-timeout 30 --output '$archive' \
          'https://codeload.github.com/flagos-ai/$repository/tar.gz/$ref' && \
        tar -xzf '$archive' --strip-components=1 -C '$target' && \
        rm -f '$archive'"
}

install_requirements() {
    set_step "Installing Hygon inference requirements"
    [ -f "$REQ_FILE" ] || die "Hygon inference requirements not found"
    retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT"
}

install_flaggems() {
    set_step "Installing resolved FlagGems for Hygon inference"
    fetch_source FlagGems "$FLAGGEMS_REF" "$FLAGSCALE_DEPS/FlagGems" \
        /tmp/hygon-flaggems.tar.gz || return 1
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c "cd '$FLAGSCALE_DEPS/FlagGems' && \
        $pip_cmd install --root-user-action=ignore --no-build-isolation --no-deps ."
}

install_vllm_plugin() {
    set_step "Installing resolved vllm-plugin-FL for Hygon inference"
    fetch_source vllm-plugin-FL "$VLLM_PLUGIN_FL_REF" \
        "$FLAGSCALE_DEPS/vllm-plugin-FL" /tmp/hygon-vllm-plugin-fl.tar.gz || return 1
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c "cd '$FLAGSCALE_DEPS/vllm-plugin-FL' && \
        CMAKE_ARGS='-DCMAKE_PREFIX_PATH=/opt/dtk;/opt/dtk/hip;/opt/dtk/hsa;/opt/hyhal;/usr/local/hyhal' \
        VLLM_VENDOR=cuda $pip_cmd install --root-user-action=ignore \
        --no-build-isolation --no-deps ."
}

validate_stack() {
    [ "$DEBUG" = true ] && return 0
    GEMS_VENDOR=hygon VLLM_PLUGINS=fl VLLM_FL_PLATFORM=hygon python - <<'PY'
import importlib.metadata as metadata

assert metadata.version("vllm").startswith("0.20.2")
assert metadata.version("vllm-plugin-fl")
assert metadata.version("flag_gems")
plugins = {
    entry.name: entry.value
    for entry in metadata.entry_points(group="vllm.platform_plugins")
}
assert plugins.get("fl") == "vllm_fl:register", plugins
PY
}

install_requirements || die "Hygon inference requirements installation failed"
install_flaggems || die "Hygon FlagGems installation failed"
install_vllm_plugin || die "Hygon vllm-plugin-FL installation failed"
validate_stack || die "Hygon inference package validation failed"

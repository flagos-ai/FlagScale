#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/ascend/inference.txt"
FLAGGEMS_REPO="${FLAGSCALE_FLAGGEMS_REPO:-https://github.com/flagos-ai/FlagGems.git}"
FLAGGEMS_REF="${FLAGSCALE_FLAGGEMS_REF:-v5.3.0}"
VLLM_PLUGIN_REPO="${FLAGSCALE_VLLM_PLUGIN_REPO:-https://github.com/flagos-ai/vllm-plugin-FL.git}"
VLLM_PLUGIN_REF="${FLAGSCALE_VLLM_PLUGIN_REF:-main}"

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

install_pip() {
    [ ! -f "$REQ_FILE" ] && { log_info "inference.txt not found"; return 0; }
    set_step "Installing Ascend inference requirements"
    retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || return 1
    log_success "Ascend inference requirements installed"
}

install_vllm_plugin() {
    set_step "Installing resolved vllm-plugin-FL"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_pinned_ref "$VLLM_PLUGIN_REPO" "$VLLM_PLUGIN_REF" \
        "$FLAGSCALE_DEPS/vllm-plugin-FL" || return 1
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c \
        "cd '$FLAGSCALE_DEPS/vllm-plugin-FL' && $pip_cmd install \
        --root-user-action=ignore --no-build-isolation --no-deps ." || return 1
    log_success "vllm-plugin-FL ready at $VLLM_PLUGIN_REF"
}

install_flaggems() {
    set_step "Installing resolved FlagGems"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_pinned_ref "$FLAGGEMS_REPO" "$FLAGGEMS_REF" \
        "$FLAGSCALE_DEPS/FlagGems" || return 1
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c \
        "cd '$FLAGSCALE_DEPS/FlagGems' && $pip_cmd install \
        --root-user-action=ignore --no-deps ." || return 1
    log_success "FlagGems ready at $FLAGGEMS_REF"
}

validate_inference_stack() {
    set_step "Validating Ascend inference package contract"
    [ "$DEBUG" = true ] && return 0
    python - <<'PY'
import importlib.metadata as metadata

vllm_version = metadata.version("vllm")
plugin_version = metadata.version("vllm-plugin-fl")
flaggems_version = metadata.version("flag_gems")
platform_plugins = {
    entry.name: entry.value
    for entry in metadata.entry_points(group="vllm.platform_plugins")
}
print("vllm:", vllm_version)
print("plugin:", plugin_version)
print("FlagGems:", flaggems_version)
print("platform_plugins:", platform_plugins)
assert vllm_version.startswith("0.20.2"), vllm_version
assert platform_plugins.get("fl") == "vllm_fl:register", platform_plugins
PY
    log_success "Ascend inference stack ready"
}

main() {
    install_pip || die "Ascend inference pip failed"
    install_flaggems || die "FlagGems failed"
    install_vllm_plugin || die "vllm-plugin-FL failed"
    validate_inference_stack || die "Ascend inference validation failed"
}

main

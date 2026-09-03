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
REQ_FILE="$PROJECT_ROOT/requirements/metax/inference.txt"
PLUGIN_REPO="${FLAGSCALE_VLLM_PLUGIN_REPO:-https://github.com/flagos-ai/vllm-plugin-FL.git}"
PLUGIN_REF="${FLAGSCALE_VLLM_PLUGIN_REF:-}"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

checkout_pinned_ref() {
    local repo=$1
    local ref=$2
    local target=$3

    [ -z "$ref" ] && { log_error "A pinned source ref is required for $repo"; return 1; }
    retry -d "$DEBUG" "$RETRY_COUNT" "rm -rf '$target' && \
        git init -q '$target' && \
        git -C '$target' remote add origin '$repo' && \
        git -c http.version=HTTP/1.1 -C '$target' fetch --depth 1 origin '$ref' && \
        git -C '$target' checkout -q --detach FETCH_HEAD"
}

install_requirements() {
    set_step "Installing MetaX inference requirements"
    retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || return 1
}

install_plugin() {
    set_step "Installing resolved vllm-plugin-FL for MetaX"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_pinned_ref "$PLUGIN_REPO" "$PLUGIN_REF" \
        "$FLAGSCALE_DEPS/vllm-plugin-FL" || return 1

    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c \
        "cd '$FLAGSCALE_DEPS/vllm-plugin-FL' && $pip_cmd install \
        --root-user-action=ignore --no-deps --no-build-isolation ." || return 1
}

validate_runtime() {
    [ "$DEBUG" = true ] && return 0
    VLLM_PLUGINS=fl VLLM_FL_PLATFORM=metax python - <<'PY'
import importlib.metadata as metadata

assert metadata.version("vllm").startswith("0.20.2")
entrypoints = {
    entry.name: entry.value
    for entry in metadata.entry_points(group="vllm.platform_plugins")
}
assert entrypoints.get("fl") == "vllm_fl:register", entrypoints
print("vllm:", metadata.version("vllm"))
print("vllm-plugin-fl:", metadata.version("vllm-plugin-fl"))
print("flag-gems:", metadata.version("flag-gems"))
print("platform entrypoints:", entrypoints)
PY
}

main() {
    install_requirements || die "MetaX inference requirements failed"
    install_plugin || die "vllm-plugin-FL installation failed"
    validate_runtime || die "MetaX inference runtime validation failed"
    log_success "MetaX inference runtime ready"
}

main

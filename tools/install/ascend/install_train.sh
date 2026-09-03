#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Train task (Ascend): Python requirements plus training source dependencies.
# The Ascend CANN runtime and torch_npu are supplied by the base image/host.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
TE_REPO="${FLAGSCALE_TE_REPO:-https://github.com/flagos-ai/TransformerEngine-FL.git}"
TE_REF="${FLAGSCALE_TE_REF:-}"
MEGATRON_REPO="${FLAGSCALE_MEGATRON_REPO:-https://github.com/flagos-ai/Megatron-LM-FL.git}"
MEGATRON_REF="${FLAGSCALE_MEGATRON_REF:-}"
REQ_FILE="$PROJECT_ROOT/requirements/ascend/train.txt"

SRC_DEPS_LIST="transformer-engine megatron-lm"

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
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "train.txt not found"; return 0; }
        set_step "Installing Ascend train requirements"
        retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Ascend train requirements installed"
    else
        local pkgs
        pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing Ascend train pip packages (override)"
        run_cmd -d "$DEBUG" "$(get_pip_cmd)" install \
            --root-user-action=ignore $pkgs || return 1
        log_success "Ascend train pip packages installed"
    fi
}

ensure_python_config() {
    command -v python3-config &>/dev/null && return 0

    local python_bin
    python_bin=$(command -v python || command -v python3) || return 1
    local python_config
    python_config="$(dirname "$python_bin")/python3-config"

    set_step "Installing python3-config for Ascend dataset helpers"
    if [ "$DEBUG" = true ]; then
        log_info "Would install $python_config"
        return 0
    fi

    cat >"$python_config" <<EOF
#!$python_bin
import sys
import sysconfig

if sys.argv[1:] != ["--extension-suffix"]:
    raise SystemExit("python3-config shim only supports --extension-suffix")
suffix = sysconfig.get_config_var("EXT_SUFFIX")
if not suffix:
    raise SystemExit("Python EXT_SUFFIX is unavailable")
print(suffix)
EOF
    chmod 0755 "$python_config"
    "$python_config" --extension-suffix >/dev/null || return 1
    log_success "python3-config ready"
}

transformer_engine_ready() {
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 TE_FL_SKIP_CUDA=1 python -c '
from transformer_engine.pytorch import DotProductAttention, LayerNormLinear
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
' &>/dev/null
}

megatron_lm_ready() {
    TE_FL_SKIP_CUDA=1 python -c '
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt import GPTModel

assert HAVE_TE
assert TESpecProvider is not None
' &>/dev/null
}

install_transformer_engine() {
    if [ "${FLAGSCALE_FORCE_BUILD:-false}" != true ] && transformer_engine_ready; then
        local version
        version=$(get_package_version "transformer-engine")
        log_info "TransformerEngine-FL is ready (version: ${version:-unknown}), skipping"
        return 0
    fi

    set_step "Installing TransformerEngine-FL for Ascend"
    mkdir -p "$FLAGSCALE_DEPS"
    if [ -n "$TE_REF" ]; then
        checkout_pinned_ref "$TE_REPO" "$TE_REF" \
            "$FLAGSCALE_DEPS/TransformerEngine-FL" || return 1
    else
        retry_git_clone -d "$DEBUG" --depth 1 "$TE_REPO" \
            "$FLAGSCALE_DEPS/TransformerEngine-FL" "$RETRY_COUNT" || return 1
    fi
    retry -d "$DEBUG" "$RETRY_COUNT" \
        "git -c http.version=HTTP/1.1 -C '$FLAGSCALE_DEPS/TransformerEngine-FL' \
        submodule update --init --recursive --depth 1 --recommend-shallow --jobs 1" || return 1

    local npu_init="$FLAGSCALE_DEPS/TransformerEngine-FL/transformer_engine/plugin/core/backends/vendor/npu/__init__.py"
    if [ "$DEBUG" != true ] && [ ! -f "$npu_init" ]; then
        log_error "TransformerEngine-FL ref ${TE_REF:-default} is missing the NPU backend package"
        return 1
    fi
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c "cd '$FLAGSCALE_DEPS/TransformerEngine-FL' && \
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 TE_FL_SKIP_CUDA=1 \
        $pip_cmd install --root-user-action=ignore \
        --no-build-isolation ." || return 1
    log_success "TransformerEngine-FL ready"
}

install_megatron_lm() {
    if [ "${FLAGSCALE_FORCE_BUILD:-false}" != true ] && \
        megatron_lm_ready; then
        local version
        version=$(get_package_version "megatron-core")
        log_info "megatron-core is importable (version: ${version:-unknown}), skipping"
        return 0
    fi

    set_step "Installing Megatron-LM-FL for Ascend"
    mkdir -p "$FLAGSCALE_DEPS"
    if [ -n "$MEGATRON_REF" ]; then
        checkout_pinned_ref "$MEGATRON_REPO" "$MEGATRON_REF" \
            "$FLAGSCALE_DEPS/Megatron-LM-FL" || return 1
    else
        retry_git_clone -d "$DEBUG" --depth 1 "$MEGATRON_REPO" \
            "$FLAGSCALE_DEPS/Megatron-LM-FL" "$RETRY_COUNT" || return 1
    fi
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c "cd '$FLAGSCALE_DEPS/Megatron-LM-FL' && \
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
        $pip_cmd install --root-user-action=ignore --no-build-isolation . -v" || return 1
    log_success "Megatron-LM-FL ready"
}

install_src() {
    if is_only_pip && ! has_src_deps_for_phase "$SRC_DEPS_LIST"; then
        log_info "Skipping source deps (only-pip mode)"
        return 0
    fi
    is_phase_enabled task || has_src_deps_for_phase "$SRC_DEPS_LIST" || return 0

    should_install_src task "transformer-engine" && {
        install_transformer_engine || die "TransformerEngine-FL failed"
    }
    should_install_src task "megatron-lm" && {
        install_megatron_lm || die "Megatron-LM-FL failed"
    }
}

validate_training_stack() {
    set_step "Validating Ascend training stack"
    if [ "$DEBUG" = true ]; then
        log_info "Would validate Megatron-LM-FL and TransformerEngine-FL integration"
        return 0
    fi

    TE_FL_SKIP_CUDA=1 python -c '
import transformer_engine
from transformer_engine.pytorch import DotProductAttention, LayerNormLinear
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

assert HAVE_TE, "Megatron-LM-FL did not detect TransformerEngine-FL"
assert TESpecProvider is not None, "TransformerEngine spec provider is unavailable"
' || return 1
    log_success "Ascend training stack ready"
}

main() {
    install_pip || die "Ascend train pip failed"
    ensure_python_config || die "python3-config setup failed"
    install_src
    validate_training_stack || die "Ascend training stack validation failed"
}

main

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

# Train task (MUSA): requirements/musa/train.txt + native MUSA TransformerEngine

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
MEGATRON_REPO="${FLAGSCALE_MEGATRON_REPO:-https://github.com/flagos-ai/Megatron-LM-FL.git}"
MEGATRON_REF="${FLAGSCALE_MEGATRON_REF:-175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82}"
TE_VERSION="${FLAGSCALE_TE_VERSION:-2.0.0+e73781e}"
REQ_FILE="$PROJECT_ROOT/requirements/musa/train.txt"
SRC_DEPS_LIST="megatron-lm"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

checkout_megatron_ref() {
    local target_dir=$1
    retry -d "$DEBUG" "$RETRY_COUNT" "rm -rf '$target_dir' && \
        git init -q '$target_dir' && \
        git -C '$target_dir' remote add origin '$MEGATRON_REPO' && \
        git -C '$target_dir' fetch --depth 1 origin '$MEGATRON_REF' && \
        git -C '$target_dir' checkout -q --detach FETCH_HEAD"
}

install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "train.txt not found"; return 0; }
        set_step "Installing MUSA train requirements"
        retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "MUSA train requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing MUSA train pip packages (override)"
        run_cmd -d "$DEBUG" "$(get_pip_cmd)" install --root-user-action=ignore $pkgs || return 1
        log_success "MUSA train pip packages installed"
    fi
}

megatron_lm_ready() {
    # A device-less Docker build must install the pinned source instead of
    # importing Megatron through an incomplete driver placeholder. At runtime,
    # validate with torch_musa auto-loading enabled.
    [ "${FLAGSCALE_MUSA_BUILD_NO_DEVICE:-false}" = true ] && return 1
    python -c '
import megatron.core
from megatron.plugin.platform import get_platform
' &>/dev/null
}

validate_megatron_lm() {
    if [ "${FLAGSCALE_MUSA_BUILD_NO_DEVICE:-false}" = true ]; then
        "$(get_pip_cmd)" show megatron-core >/dev/null 2>&1 || return 1
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c '
import importlib.util
assert importlib.util.find_spec("megatron") is not None
' || return 1
        log_success "Megatron-LM-FL package is installed; import validation deferred to runtime"
        return 0
    fi
    python -c '
import megatron.core
from megatron.plugin.platform import get_platform
print("Megatron-LM-FL import validation passed")
'
}

install_megatron_lm() {
    if [ "${FLAGSCALE_FORCE_BUILD:-false}" != true ] && megatron_lm_ready; then
        log_info "Megatron-LM-FL is importable, skipping"
        return 0
    fi

    set_step "Installing Megatron-LM-FL for MUSA"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_megatron_ref "$FLAGSCALE_DEPS/Megatron-LM-FL" || return 1

    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    # The pinned source is verified with the vendor's Python 3.10 runtime, but
    # currently declares Python >=3.12 in package metadata. Keep this exception
    # explicit and fail below if the source stops being Python 3.10 compatible.
    run_cmd -d "$DEBUG" bash -c "cd '$FLAGSCALE_DEPS/Megatron-LM-FL' && \
        $pip_cmd install --ignore-requires-python --root-user-action=ignore \
        --no-build-isolation . -v" || return 1
    validate_megatron_lm || return 1
    log_success "Megatron-LM-FL ready"
}

validate_transformer_engine() {
    set_step "Validating native TransformerEngine for MUSA"
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python - "$TE_VERSION" <<'PY' || return 1
import importlib.metadata as metadata
import sys

expected = sys.argv[1]
actual = metadata.version("transformer-engine")
assert actual == expected, (actual, expected)
files = [str(path) for path in metadata.files("transformer-engine") or ()]
assert any("transformer_engine_torch" in path and path.endswith(".so") for path in files), files
PY
    if [ "${FLAGSCALE_MUSA_BUILD_NO_DEVICE:-false}" != true ]; then
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c '
import transformer_engine
import transformer_engine_torch as tex
for symbol in ("generic_gemm", "layernorm_fwd", "layernorm_bwd", "rmsnorm_fwd", "rmsnorm_bwd"):
    assert hasattr(tex, symbol), symbol
print(transformer_engine.__version__)
' || return 1
    fi
    log_success "Native MUSA TransformerEngine ready"
}

install_src() {
    if is_only_pip && ! has_src_deps_for_phase $SRC_DEPS_LIST; then
        log_info "Skipping source deps (only-pip mode)"
        return 0
    fi
    is_phase_enabled task || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src task "megatron-lm" && {
        install_megatron_lm || die "Megatron-LM-FL failed"
    }
}

verify_musa_runtime() {
    set_step "Validating torch_musa runtime"
    "$(get_pip_cmd)" show torch-musa >/dev/null 2>&1 || return 1
    if [ "${FLAGSCALE_MUSA_BUILD_NO_DEVICE:-false}" = true ]; then
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "import torch" || return 1
        log_success "torch_musa package is installed; device validation deferred to runtime"
        return 0
    fi
    python -c "import torch, torch_musa; assert hasattr(torch, 'musa')" || return 1
    log_success "torch_musa runtime is importable"
}

main() {
    install_pip || die "MUSA train pip failed"
    install_src
    validate_transformer_engine || die "Native MUSA TransformerEngine validation failed"
    verify_musa_runtime || die "MUSA runtime validation failed"
}

main

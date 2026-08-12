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

# Inference task (MUSA): requirements/musa/inference.txt. The MUSA-compatible
# vLLM distribution is supplied by the pinned vendor base image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
REQ_FILE="$PROJECT_ROOT/requirements/musa/inference.txt"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_error "inference.txt not found"; return 1; }
        set_step "Installing MUSA inference requirements"
        retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "MUSA inference requirements installed"
    else
        local pkgs
        pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing MUSA inference pip packages (override)"
        run_cmd -d "$DEBUG" "$(get_pip_cmd)" install --root-user-action=ignore $pkgs || return 1
    fi
}

verify_runtime() {
    set_step "Validating MUSA inference runtime packages"
    "$(get_pip_cmd)" show torch-musa >/dev/null 2>&1 || return 1
    "$(get_pip_cmd)" show vllm >/dev/null 2>&1 || return 1
    if [ "${FLAGSCALE_MUSA_BUILD_NO_DEVICE:-false}" = true ]; then
        log_success "MUSA inference packages installed; device validation deferred to runtime"
        return 0
    fi
    FS_PLATFORM=musa python -c 'import vllm; import torch, torch_musa; assert torch.musa.is_available()' || return 1
    log_success "MUSA inference runtime is importable"
}

install_pip || die "MUSA inference pip failed"
verify_runtime || die "MUSA inference runtime validation failed"

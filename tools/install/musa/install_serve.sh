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

# Serve task (MUSA): requirements/musa/serve.txt. vLLM is supplied by the
# pinned vendor base image and is checked here rather than rebuilt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
REQ_FILE="$PROJECT_ROOT/requirements/musa/serve.txt"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

if is_phase_enabled task; then
    [ -f "$REQ_FILE" ] || die "serve.txt not found"
    set_step "Installing MUSA serve requirements"
    retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || die "MUSA serve pip failed"
else
    pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
    if [ -n "$pkgs" ]; then
        run_cmd -d "$DEBUG" "$(get_pip_cmd)" install --root-user-action=ignore $pkgs || die "MUSA serve pip failed"
    fi
fi

"$(get_pip_cmd)" show vllm >/dev/null 2>&1 || die "MUSA vLLM runtime not found"
log_success "MUSA serve runtime ready"

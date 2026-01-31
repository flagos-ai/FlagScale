#!/bin/bash
# Source dependencies for serve task (CUDA platform)
#
# This script is called by install.sh after base and pip requirements.
# It only handles source dependencies (git repos, etc.)
#
# Currently a placeholder - add source dependencies here when needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

# Use inherited values or defaults for standalone execution
PROJECT_ROOT="${PROJECT_ROOT:-$(get_project_root)}"
RETRY_COUNT="${RETRY_COUNT:-3}"

install_vllm_lm() {
    local vllm_dir="$PROJECT_ROOT/vllm-FL"
    local vllm_url="https://github.com/flagos-ai/vllm-FL.git"

    log_info "Installing vllm-FL"

    # Clone repository
    retry_git_clone "$vllm_url" "$vllm_dir" "$RETRY_COUNT"

    # Install from source
    cd "$vllm_dir"
    retry "$RETRY_COUNT" "pip install . -vvv"
    cd "$PROJECT_ROOT"

    log_success "vllm-FL installed"
}

main() {
    log_step "Installing source dependencies for serve task"
    install_vllm_lm
}

main "$@"

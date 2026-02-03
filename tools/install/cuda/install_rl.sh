#!/bin/bash
# RL task (CUDA): requirements/cuda/rl.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
REQ_FILE="$PROJECT_ROOT/requirements/cuda/rl.txt"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "rl.txt not found"; return 0; }
        set_step "Installing rl requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "RL requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing rl pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "RL pip packages installed"
    fi
}

main() {
    install_pip || die "RL pip failed"
    # No source deps for rl task yet
    # To add: SRC_DEPS_LIST="dep1 dep2" and install_src function
}

main

#!/bin/bash
# Inference task (CUDA): requirements/cuda/inference.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
REQ_FILE="$PROJECT_ROOT/requirements/cuda/inference.txt"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "inference.txt not found"; return 0; }
        set_step "Installing inference requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Inference requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing inference pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "Inference pip packages installed"
    fi
}

main() {
    install_pip || die "Inference pip failed"
    # No source deps for inference task yet
    # To add: SRC_DEPS_LIST="dep1 dep2" and install_src function
}

main

#!/bin/bash
# Base phase (CUDA): apt packages + requirements/cuda/base.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
REQ_FILE="$PROJECT_ROOT/requirements/cuda/base.txt"

APT_PACKAGES="libcudnn9-dev-cuda-12"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

install_apt() {
    is_phase_enabled base || return 0
    set_step "Installing CUDA apt packages"
    run_cmd -d $DEBUG apt-get install -y --no-install-recommends $APT_PACKAGES || return 1
    log_success "CUDA apt packages installed"
}

install_pip() {
    if is_phase_enabled base; then
        # Phase enabled: install full requirements
        [ ! -f "$REQ_FILE" ] && { log_info "base.txt not found"; return 0; }
        set_step "Installing base requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Base requirements installed"
    else
        # Phase disabled: install only matching pip-deps
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing base pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "Base pip packages installed"
    fi
}

main() {
    install_apt || die "CUDA apt packages failed"
    install_pip || die "Base pip failed"
}

main

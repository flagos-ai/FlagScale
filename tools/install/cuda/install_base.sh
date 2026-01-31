#!/bin/bash
# Base dependencies for CUDA platform
#
# This script installs:
#   1. CUDA-specific apt packages (cudnn-dev, etc.)
#   2. Source dependencies (git repos, etc.)
#
# NOTE: Pip requirements (common.txt, base.txt) are handled by install.sh
# via install_base_requirements().

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

# Use inherited values or defaults for standalone execution
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
RETRY_COUNT="${RETRY_COUNT:-3}"
DEBUG=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug) DEBUG=true; shift ;;
        *) shift ;;
    esac
done

# CUDA-specific apt packages
CUDA_PACKAGES="
    libcudnn9-dev-cuda-12
"

install_cuda_packages() {
    set_step "Installing CUDA-specific packages"

    run_cmd -d $DEBUG -m "Installing CUDA packages..." \
        apt-get install -y --no-install-recommends $CUDA_PACKAGES || return 1
    log_success "CUDA packages installed"
}

main() {
    install_cuda_packages || die "CUDA packages installation failed"
    log_info "CUDA base dependencies complete"
}

main

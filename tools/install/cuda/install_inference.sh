#!/bin/bash
# Source dependencies for inference task (CUDA platform)
# Available deps: (none yet)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"

FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
RETRY_COUNT="${RETRY_COUNT:-3}"
DEBUG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug) DEBUG=true; shift ;;
        *) shift ;;
    esac
done

# To add a dependency:
# install_example() {
#     set_step "Installing Example"
#     if ! should_build_package "example"; then return 0; fi
#     # ... clone and build
# }

main() {
    log_info "No source dependencies for inference task"
    # should_install_dep "example" && { install_example || die "Example failed"; }
}

main "$@"

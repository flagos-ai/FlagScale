#!/bin/bash
# Source dependencies for serve task (CUDA platform)
# Available deps: vllm

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

install_vllm() {
    set_step "Installing vLLM-FL"

    if ! should_build_package "vllm"; then
        return 0
    fi

    local vllm_dir="$FLAGSCALE_DEPS/vllm-FL"
    local vllm_url="https://github.com/flagos-ai/vllm-FL.git"

    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG "$vllm_url" "$vllm_dir" "$RETRY_COUNT" || return 1

    log_info "Building vLLM-FL"
    run_cmd -d $DEBUG -m "Installing from source..." \
        bash -c "cd '$vllm_dir' && pip install --root-user-action=ignore --no-build-isolation . -vvv" || return 1
    log_success "vLLM-FL ready"
}

main() {
    should_install_dep "vllm" && { install_vllm || die "vLLM installation failed"; }
}

main "$@"

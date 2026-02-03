#!/bin/bash
# Serve task (CUDA): requirements/cuda/serve.txt + source deps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/cuda/serve.txt"

# Source deps available for this task
SRC_DEPS_LIST="vllm"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

# =============================================================================
# Pip Installation
# =============================================================================
install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "serve.txt not found"; return 0; }
        set_step "Installing serve requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Serve requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing serve pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "Serve pip packages installed"
    fi
}

# =============================================================================
# Source Dependencies
# =============================================================================
install_vllm() {
    should_build_package "vllm" || return 0
    set_step "Installing vLLM-FL"
    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG "https://github.com/flagos-ai/vllm-FL.git" "$FLAGSCALE_DEPS/vllm-FL" "$RETRY_COUNT" || return 1
    local pip_cmd=$(get_pip_cmd)
    run_cmd -d $DEBUG bash -c "cd '$FLAGSCALE_DEPS/vllm-FL' && \
        $pip_cmd install --root-user-action=ignore --no-build-isolation . -vvv" || return 1
    log_success "vLLM-FL ready"
}

install_src() {
    # Skip if phase disabled and no matching src-deps
    is_phase_enabled task || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src task "vllm" && { install_vllm || die "vLLM failed"; }
}

main() {
    install_pip || die "Serve pip failed"
    install_src
}

main

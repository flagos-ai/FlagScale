#!/bin/bash
# Serve task (Ascend): requirements/ascend/serve.txt + source deps (FlagGems, vllm-plugin-FL)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/ascend/serve.txt"

# Source deps available for this task
SRC_DEPS_LIST="flaggems vllm-plugin"

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
install_flaggems() {
    should_build_package "flag_gems" || return 0
    set_step "Installing FlagGems"
    mkdir -p "$FLAGSCALE_DEPS"
    local pip_cmd=$(get_pip_cmd)
    # Install build dependencies for FlagGems
    run_cmd -d $DEBUG $pip_cmd install --root-user-action=ignore -U \
        scikit-build-core==0.11 pybind11 ninja cmake || return 1
    retry_git_clone -d $DEBUG "https://github.com/flagos-ai/FlagGems.git" "$FLAGSCALE_DEPS/FlagGems" "$RETRY_COUNT" || return 1
    run_cmd -d $DEBUG bash -c "cd '$FLAGSCALE_DEPS/FlagGems' && \
        $pip_cmd install --root-user-action=ignore --no-build-isolation -e ." || return 1
    log_success "FlagGems ready"
}

install_vllm_plugin() {
    should_build_package "vllm_plugin_fl" || return 0
    set_step "Installing vllm-plugin-FL"
    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG "https://github.com/flagos-ai/vllm-plugin-FL.git" "$FLAGSCALE_DEPS/vllm-plugin-FL" "$RETRY_COUNT" || return 1
    local pip_cmd=$(get_pip_cmd)
    # Install requirements.txt first, then install the plugin
    local req="$FLAGSCALE_DEPS/vllm-plugin-FL/requirements.txt"
    if [ -f "$req" ] || [ "$DEBUG" = true ]; then
        run_cmd -d $DEBUG $pip_cmd install --root-user-action=ignore -r "$req" || return 1
    fi
    run_cmd -d $DEBUG bash -c "cd '$FLAGSCALE_DEPS/vllm-plugin-FL' && \
        $pip_cmd install --root-user-action=ignore --no-build-isolation -e ." || return 1
    log_success "vllm-plugin-FL ready"
}

install_src() {
    # Skip in only-pip mode unless we have matching src-deps overrides
    if is_only_pip && ! has_src_deps_for_phase $SRC_DEPS_LIST; then
        log_info "Skipping source deps (only-pip mode)"
        return 0
    fi
    # Skip if phase disabled and no matching src-deps
    is_phase_enabled task || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src task "flaggems" && { install_flaggems || die "FlagGems failed"; }
    should_install_src task "vllm-plugin" && { install_vllm_plugin || die "vllm-plugin-FL failed"; }
}

main() {
    install_pip || die "Serve pip failed"
    install_src
}

main

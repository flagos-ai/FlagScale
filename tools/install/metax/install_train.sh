#!/bin/bash
# Train task (MetaX): requirements/metax/train.txt + source deps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/metax/train.txt"

SRC_DEPS_LIST="transformer-engine"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "train.txt not found"; return 0; }
        set_step "Installing train requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Train requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing train pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "Train pip packages installed"
    fi
}

install_transformer_engine() {
    should_build_package "transformer-engine" || return 0
    set_step "Installing TransformerEngine-FL"
    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG --depth 1 \
        "https://github.com/flagos-ai/TransformerEngine-FL.git" "$FLAGSCALE_DEPS/TransformerEngine-FL" "$RETRY_COUNT" || return 1
    local pip_cmd=$(get_pip_cmd)
    run_cmd -d $DEBUG bash -c "cd '$FLAGSCALE_DEPS/TransformerEngine-FL' && \
        TE_FL_SKIP_CUDA=1 $pip_cmd install --root-user-action=ignore --no-build-isolation ." || return 1
    log_success "TransformerEngine-FL ready"
}

install_src() {
    if is_only_pip && ! has_src_deps_for_phase $SRC_DEPS_LIST; then
        log_info "Skipping source deps (only-pip mode)"
        return 0
    fi
    is_phase_enabled task || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src task "transformer-engine" && { install_transformer_engine || die "TransformerEngine-FL failed"; }
}

main() {
    install_pip || die "Train pip failed"
    install_src
}

main

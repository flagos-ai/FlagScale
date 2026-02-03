#!/bin/bash
# Train task (CUDA): requirements/cuda/train.txt + source deps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/cuda/train.txt"

# Source deps available for this task
SRC_DEPS_LIST="apex flash-attn transformer-engine megatron-lm"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

# =============================================================================
# Pip Installation
# =============================================================================
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

# =============================================================================
# Source Dependencies
# =============================================================================
install_apex() {
    should_build_package "apex" || return 0
    set_step "Installing NVIDIA Apex"
    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG "https://github.com/NVIDIA/apex.git" "$FLAGSCALE_DEPS/apex" "$RETRY_COUNT" || return 1
    local pip_cmd=$(get_pip_cmd)
    run_cmd -d $DEBUG bash -c "cd '$FLAGSCALE_DEPS/apex' && \
        NVCC_APPEND_FLAGS='--threads 4' APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
        $pip_cmd install --root-user-action=ignore --no-build-isolation . -v" || return 1
    log_success "NVIDIA Apex ready"
}

install_flash_attn() {
    should_build_package "flash_attn" || return 0
    set_step "Installing Flash-Attention 2"
    local version="${FLASH_ATTN_VERSION:-2.8.1}"
    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG --branch "v${version}" --depth 1 \
        "https://github.com/Dao-AILab/flash-attention.git" "$FLAGSCALE_DEPS/flash-attention" "$RETRY_COUNT" || return 1
    local pip_cmd=$(get_pip_cmd)
    run_cmd -d $DEBUG bash -c "cd '$FLAGSCALE_DEPS/flash-attention' && \
        FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
        $pip_cmd install --root-user-action=ignore --no-build-isolation . -vvv" || return 1
    log_success "Flash-Attention 2 ready"
}

install_transformer_engine() {
    should_build_package "transformer_engine" || return 0
    set_step "Installing TransformerEngine"
    local pip_cmd=$(get_pip_cmd)
    run_cmd -d $DEBUG $pip_cmd install --root-user-action=ignore nvidia-mathdx --extra-index-url https://pypi.nvidia.com || return 1
    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG --recursive \
        "https://github.com/NVIDIA/TransformerEngine.git" "$FLAGSCALE_DEPS/TransformerEngine" "$RETRY_COUNT" || return 1
    run_cmd -d $DEBUG bash -c "cd '$FLAGSCALE_DEPS/TransformerEngine' && \
        NVTE_FRAMEWORK=pytorch $pip_cmd install --root-user-action=ignore --no-build-isolation . -vvv" || return 1
    log_success "TransformerEngine ready"
}

install_megatron_lm() {
    should_build_package "megatron-core" || return 0
    set_step "Installing Megatron-LM-FL"
    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG "https://github.com/flagos-ai/Megatron-LM-FL.git" "$FLAGSCALE_DEPS/Megatron-LM-FL" "$RETRY_COUNT" || return 1
    local pip_cmd=$(get_pip_cmd)
    run_cmd -d $DEBUG bash -c "cd '$FLAGSCALE_DEPS/Megatron-LM-FL' && \
        $pip_cmd install --root-user-action=ignore --no-build-isolation . -vvv" || return 1
    log_success "Megatron-LM-FL ready"
}

install_src() {
    # Skip if phase disabled and no matching src-deps
    is_phase_enabled task || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src task "apex" && { install_apex || die "Apex failed"; }
    should_install_src task "flash-attn" && { install_flash_attn || die "Flash-Attention failed"; }
    should_install_src task "transformer-engine" && { install_transformer_engine || die "TransformerEngine failed"; }
    should_install_src task "megatron-lm" && { install_megatron_lm || die "Megatron-LM failed"; }
}

main() {
    install_pip || die "Train pip failed"
    install_src
}

main

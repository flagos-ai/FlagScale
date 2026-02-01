#!/bin/bash
# Source dependencies for train task (CUDA platform)
# Installs: NVIDIA Apex, Flash-Attention 2, TransformerEngine, Megatron-LM-FL
#
# This script is called by install.sh after base and pip requirements.
# It only handles source dependencies (git repos, etc.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/versions.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"

# Use inherited values or defaults for standalone execution
# FLAGSCALE_HOME is the root, FLAGSCALE_DEPS is where source deps go
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

install_apex() {
    set_step "Installing NVIDIA Apex"

    # Skip if already installed (unless --force-build)
    if ! should_build_package "apex"; then
        return 0
    fi

    local apex_dir="$FLAGSCALE_DEPS/apex"
    local apex_url="https://github.com/NVIDIA/apex.git"

    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG "$apex_url" "$apex_dir" "$RETRY_COUNT" || return 1

    log_info "Building NVIDIA Apex"
    # Build with cpp and cuda extensions, parallel compilation
    run_cmd -d $DEBUG -m "Installing from source (may take a while)..." \
        bash -c "cd '$apex_dir' && NVCC_APPEND_FLAGS='--threads 4' APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install --root-user-action=ignore --no-build-isolation . -v" || return 1
    log_success "NVIDIA Apex ready"
}

# Install Flash-Attention
install_flash_attn() {
    set_step "Installing Flash-Attention"

    # Check if flash_attn_2 needs to be installed
    local need_flash_attn_2=$(should_build_package "flash_attn" && echo true || echo false)

    if [ "$need_flash_attn_2" = false ]; then
        log_info "Flash-Attention 2 already installed"
        return 0
    fi

    local flash_dir="$FLAGSCALE_DEPS/flash-attention"
    local flash_url="https://github.com/Dao-AILab/flash-attention.git"
    local flash_version=$(get_task "cuda" "train" "flash-attn")
    local flash_branch="v${flash_version:-2.8.1}"

    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG --branch "$flash_branch" --depth 1 "$flash_url" "$flash_dir" "$RETRY_COUNT" || return 1

    # Install flash-attn 2
    if [ "$need_flash_attn_2" = true ]; then
        # FLASH_ATTENTION_FORCE_BUILD=TRUE forces building from source (avoids cross-device link errors)
        log_info "Building Flash-Attention 2"
        run_cmd -d $DEBUG -m "Installing flash-attn 2 from source..." \
            bash -c "cd '$flash_dir' && FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 pip install --root-user-action=ignore --no-build-isolation . -vvv" || return 1
        log_success "Flash-Attention 2 ready"
    fi
}

install_transformer_engine() {
    set_step "Installing TransformerEngine-FL"

    # Skip if already installed (unless --force-build)
    if ! should_build_package "transformer_engine"; then
        return 0
    fi

    local te_dir="$FLAGSCALE_DEPS/TransformerEngine-FL"
    # TODO: switch back to TransformerEngine-FL fork when fixed
    # local te_url="https://github.com/flagos-ai/TransformerEngine-FL.git"
    local te_url="https://github.com/NVIDIA/TransformerEngine.git"

    # Install nvidia-mathdx dependency (required for building TransformerEngine)
    log_info "Installing nvidia-mathdx dependency"
    run_cmd -d $DEBUG -m "Installing nvidia-mathdx from NVIDIA index..." \
        pip install --root-user-action=ignore nvidia-mathdx --extra-index-url https://pypi.nvidia.com || return 1

    # Clone with submodules (--recursive)
    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG --recursive "$te_url" "$te_dir" "$RETRY_COUNT" || return 1

    log_info "Building TransformerEngine-FL"
    # Export NVTE_FRAMEWORK=pytorch to enable PyTorch bindings (te.pytorch module)
    run_cmd -d $DEBUG -m "Installing from source (may take a while)..." \
        bash -c "cd '$te_dir' && NVTE_FRAMEWORK=pytorch pip install --root-user-action=ignore --no-build-isolation . -vvv" || return 1
    log_success "TransformerEngine-FL ready"
}

install_megatron_lm() {
    set_step "Installing Megatron-LM-FL"

    # Skip if already installed (unless --force-build)
    if ! should_build_package "megatron-core"; then
        return 0
    fi

    local megatron_dir="$FLAGSCALE_DEPS/Megatron-LM-FL"
    local megatron_url="https://github.com/flagos-ai/Megatron-LM-FL.git"

    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_clone -d $DEBUG "$megatron_url" "$megatron_dir" "$RETRY_COUNT" || return 1

    log_info "Building Megatron-LM-FL"
    run_cmd -d $DEBUG -m "Installing from source (may take a while)..." \
        bash -c "cd '$megatron_dir' && pip install --root-user-action=ignore --no-build-isolation . -vvv" || return 1
    log_success "Megatron-LM-FL ready"
}

main() {
    install_apex || die "Apex installation failed"
    install_flash_attn || die "Flash-Attention installation failed"
    install_transformer_engine || die "TransformerEngine installation failed"
    install_megatron_lm || die "Megatron-LM installation failed"
}

main

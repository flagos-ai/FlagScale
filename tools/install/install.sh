#!/bin/bash
# =============================================================================
# FlagScale Dependency Installation
# =============================================================================
#
# Installs dependencies for FlagScale tasks. By default, all phases run.
# Use --no-* flags to skip phases, or --pip-deps/--src-deps to install specific deps.
#
# Usage:
#   ./install.sh --platform PLATFORM --task TASK [OPTIONS]
#
# Examples:
#   ./install.sh --platform cuda --task train                    # Full installation
#   ./install.sh --platform cuda --task train --src-deps megatron-lm  # Source dep only
#   ./install.sh --platform cuda --task train --pip-deps torch,numpy  # Pip packages only
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/utils.sh"
source "$SCRIPT_DIR/utils/versions.sh"
source "$SCRIPT_DIR/utils/pkg_utils.sh"
source "$SCRIPT_DIR/utils/retry_utils.sh"
source "$SCRIPT_DIR/utils/pyenv_utils.sh"

PROJECT_ROOT=$(get_project_root)

# =============================================================================
# Configuration
# =============================================================================
TASK=""
PLATFORM=""
RETRY_COUNT=3
PKG_MGR="uv"
ENV_NAME=""
DEBUG=false
PYTHON_VERSION="$(get_common "python")"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"

# Installation phases (default: install all)
INSTALL_SYSTEM=true
INSTALL_DEV=true
INSTALL_BASE=true
INSTALL_TASK=true

# Selective installation (empty = use phase defaults)
SRC_DEPS=""           # Comma-separated source deps (e.g., "megatron-lm")
PIP_DEPS=""           # Comma-separated pip packages (e.g., "torch,numpy")
FORCE_BUILD=false

# PyPI index URLs
INDEX_URL="${PIP_INDEX_URL:-}"
EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"

# =============================================================================
# Helper Functions
# =============================================================================
get_valid_tasks() {
    local tasks=()
    if [ -d "$SCRIPT_DIR/$PLATFORM" ]; then
        for script in "$SCRIPT_DIR/$PLATFORM"/install_*.sh; do
            [ -f "$script" ] || continue
            local task=$(basename "$script" | sed 's/^install_//' | sed 's/\.sh$//')
            [ "$task" != "base" ] && tasks+=("$task")
        done
    fi
    tasks+=("all")
    echo "${tasks[@]}"
}

# =============================================================================
# Installation Functions
# =============================================================================
install_system_deps() {
    set_step "Installing system dependencies"

    local system_script="$SCRIPT_DIR/install_system.sh"
    [ ! -f "$system_script" ] && die "install_system.sh not found"

    local args=""
    [ "$INSTALL_DEV" = false ] && args="--no-dev"
    [ -n "$PLATFORM" ] && args="$args --platform $PLATFORM"
    [ -n "$PKG_MGR" ] && args="$args --pkg-mgr $PKG_MGR"
    [ "$DEBUG" = true ] && args="$args --debug"

    FLAGSCALE_HOME="$FLAGSCALE_HOME" "$system_script" $args || die "System dependencies installation failed"
}

install_dev_deps() {
    set_step "Installing dev dependencies"

    local dev_req="$PROJECT_ROOT/requirements/dev.txt"
    [ ! -f "$dev_req" ] && [ "$DEBUG" != true ] && { log_warn "dev.txt not found, skipping"; return 0; }

    log_info "Dev requirements: dev.txt"
    retry_pip_install -d $DEBUG "$dev_req" "$RETRY_COUNT" || die "Dev requirements installation failed"
    log_success "Dev dependencies installed"
}

install_base_requirements() {
    set_step "Installing base requirements"

    local base_req="$PROJECT_ROOT/requirements/$PLATFORM/base.txt"
    [ ! -f "$base_req" ] && [ "$DEBUG" != true ] && return 0

    log_info "Base requirements: base.txt"
    retry_pip_install -d $DEBUG "$base_req" "$RETRY_COUNT" || die "Base requirements installation failed"
}

install_base_source() {
    set_step "Installing base source dependencies"

    local base_script="$SCRIPT_DIR/$PLATFORM/install_base.sh"
    [ ! -f "$base_script" ] && return 0

    local args=""
    [ "$DEBUG" = true ] && args="--debug"
    "$base_script" $args || die "Base source installation failed"
}

install_base_deps() {
    print_header "Base: $PLATFORM"
    install_base_requirements
    install_base_source
}

install_task_requirements() {
    local task=$1
    set_step "Installing $task requirements"

    local req_file="$PROJECT_ROOT/requirements/$PLATFORM/${task}.txt"
    [ ! -f "$req_file" ] && [ "$DEBUG" != true ] && return 0

    log_info "Requirements: $(basename "$req_file")"
    retry_pip_install -d $DEBUG "$req_file" "$RETRY_COUNT" || die "Task requirements installation failed: $task"
}

install_task_source() {
    local task=$1
    set_step "Installing $task source dependencies"

    local source_script="$SCRIPT_DIR/$PLATFORM/install_${task}.sh"
    [ ! -f "$source_script" ] && return 0

    local args=""
    [ "$DEBUG" = true ] && args="--debug"
    "$source_script" $args || die "Task source installation failed: $task"
}

install_task_deps() {
    local task=$1
    print_header "Task: $task ($PLATFORM)"
    install_task_requirements "$task"
    install_task_source "$task"
}

install_task() {
    local task=$1

    [ "$INSTALL_BASE" = true ] && install_base_deps
    [ "$INSTALL_TASK" = true ] && install_task_deps "$task"

    log_success "Task complete: $task"
}

# Install specific pip packages (when --pip-deps is used)
install_pip_deps() {
    [ -z "$PIP_DEPS" ] && return 0

    set_step "Installing pip packages"
    log_info "Pip packages: $PIP_DEPS"

    # Convert comma-separated to space-separated
    local packages=$(echo "$PIP_DEPS" | tr ',' ' ')
    run_cmd -d $DEBUG pip install --root-user-action=ignore $packages || die "Pip packages installation failed"
    log_success "Pip packages installed"
}

# Install specific source deps (when --src-deps is used)
install_src_deps() {
    [ -z "$SRC_DEPS" ] && return 0

    set_step "Installing source dependencies"
    log_info "Source deps: $SRC_DEPS"

    local source_script="$SCRIPT_DIR/$PLATFORM/install_${TASK}.sh"
    [ ! -f "$source_script" ] && { log_warn "No source script for task: $TASK"; return 0; }

    local args=""
    [ "$DEBUG" = true ] && args="--debug"
    "$source_script" $args || die "Source dependencies installation failed"
}

# =============================================================================
# Main
# =============================================================================
usage() {
    local tasks=($(get_valid_tasks))
    cat << EOF
Usage: $0 --platform PLATFORM --task TASK [OPTIONS]

OPTIONS:
    --platform NAME        Platform (required, e.g., cuda)
    --task TASK            Task: ${tasks[*]} (required)

  Phase Control (default: install all):
    --no-system            Skip system packages (apt, python, openmpi)
    --no-dev               Skip dev dependencies
    --no-base              Skip base dependencies
    --no-task              Skip task dependencies

  Selective Installation:
    --pip-deps PKGS        Install specific pip packages (comma-separated)
    --src-deps DEPS        Install specific source deps (comma-separated)
                           Available for train: apex,flash-attn,transformer-engine,megatron-lm
                           Available for serve: vllm

  Environment:
    --pkg-mgr MGR          Package manager: pip, uv, conda (default: uv)
    --env-name NAME        Conda environment name
    --install-dir DIR      Root installation directory (default: /opt/flagscale)
    --index-url URL        PyPI index URL
    --extra-index-url URL  Extra PyPI index URL

  Other:
    --retry-count N        Retry attempts (default: 3)
    --force-build          Force rebuild source deps
    --debug                Dry-run mode
    --help                 Show this help

EXAMPLES:
    $0 --platform cuda --task train                                # Full installation
    $0 --platform cuda --task train --no-system                    # Skip system deps
    $0 --platform cuda --task train --src-deps megatron-lm         # Only Megatron-LM source
    $0 --platform cuda --task train --pip-deps torch,numpy         # Only pip packages
    $0 --platform cuda --task train --pkg-mgr conda --env-name train
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --task)            TASK="$2"; shift 2 ;;
            --platform)        PLATFORM="$2"; shift 2 ;;
            --no-system)       INSTALL_SYSTEM=false; shift ;;
            --no-dev)          INSTALL_DEV=false; shift ;;
            --no-base)         INSTALL_BASE=false; shift ;;
            --no-task)         INSTALL_TASK=false; shift ;;
            --pkg-mgr)         PKG_MGR="$2"; shift 2 ;;
            --env-name)        ENV_NAME="$2"; shift 2 ;;
            --install-dir)     FLAGSCALE_HOME="$2"; shift 2 ;;
            --index-url)       INDEX_URL="$2"; shift 2 ;;
            --extra-index-url) EXTRA_INDEX_URL="$2"; shift 2 ;;
            --retry-count)     RETRY_COUNT="$2"; shift 2 ;;
            --force-build)     FORCE_BUILD=true; shift ;;
            --src-deps)        SRC_DEPS="$2"; shift 2 ;;
            --pip-deps)        PIP_DEPS="$2"; shift 2 ;;
            --debug)           DEBUG=true; shift ;;
            --help|-h)         usage; exit 0 ;;
            *)                 log_error "Unknown option: $1"; usage; exit 1 ;;
        esac
    done
}

validate_inputs() {
    if [ -z "$PLATFORM" ]; then
        log_error "Platform required (use --platform)"
        usage
        exit 1
    fi

    if [ ! -d "$SCRIPT_DIR/$PLATFORM" ]; then
        log_error "Invalid platform: $PLATFORM"
        exit 1
    fi

    if [ -z "$TASK" ]; then
        log_error "Task required (use --task)"
        usage
        exit 1
    fi

    local valid_tasks=($(get_valid_tasks))
    local valid=false
    for t in "${valid_tasks[@]}"; do
        [ "$TASK" = "$t" ] && valid=true && break
    done
    [ "$valid" = false ] && { log_error "Invalid task: $TASK. Valid: ${valid_tasks[*]}"; exit 1; }
}

setup_package_manager() {
    case "$PKG_MGR" in
        pip|uv|conda) set_pkg_manager "$PKG_MGR" ;;
        *) log_error "Invalid package manager: $PKG_MGR"; exit 1 ;;
    esac
}

setup_environment() {
    local manager=$(get_pkg_manager)

    case "$manager" in
        conda)
            if [ ! -f "$FLAGSCALE_CONDA/bin/conda" ]; then
                log_error "Conda not found at $FLAGSCALE_CONDA"
                return 1
            fi
            [ -n "$ENV_NAME" ] && { activate_conda -d $DEBUG "$ENV_NAME" "$FLAGSCALE_CONDA" "$PYTHON_VERSION" || return 1; }
            ;;
        uv)
            if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
                log_error "UV venv not found at $UV_PROJECT_ENVIRONMENT"
                return 1
            fi
            activate_uv_env -d $DEBUG "$UV_PROJECT_ENVIRONMENT"
            ;;
        pip)
            has_pip || { log_warn "pip not found"; return 1; }
            ;;
    esac
    return 0
}

setup_exports() {
    export FLAGSCALE_HOME
    export FLAGSCALE_CONDA="$FLAGSCALE_HOME/miniconda3"
    export FLAGSCALE_DEPS="$FLAGSCALE_HOME/deps"
    export FLAGSCALE_DOWNLOADS="$FLAGSCALE_HOME/downloads"
    export FLAGSCALE_FORCE_BUILD="$FORCE_BUILD"
    export FLAGSCALE_SRC_DEPS="$SRC_DEPS"
    export UV_PROJECT_ENVIRONMENT="$FLAGSCALE_HOME/venv"

    [ -n "$INDEX_URL" ] && { export PIP_INDEX_URL="$INDEX_URL" UV_INDEX_URL="$INDEX_URL"; }
    [ -n "$EXTRA_INDEX_URL" ] && { export PIP_EXTRA_INDEX_URL="$EXTRA_INDEX_URL" UV_EXTRA_INDEX_URL="$EXTRA_INDEX_URL"; }
}

main() {
    parse_args "$@"

    [ "$DEBUG" = true ] && log_info "Dry-run mode"

    validate_inputs

    print_header "FlagScale Installation"
    log_info "Task: $TASK | Platform: $PLATFORM | Pkg: $PKG_MGR"
    [ -n "$SRC_DEPS" ] && log_info "Source deps: $SRC_DEPS"
    [ -n "$PIP_DEPS" ] && log_info "Pip deps: $PIP_DEPS"
    log_info "Install dir: $FLAGSCALE_HOME"

    setup_exports

    # Phase 1: System dependencies
    [ "$INSTALL_SYSTEM" = true ] && install_system_deps

    # Check if we need environment setup
    local needs_env=false
    [ "$INSTALL_DEV" = true ] || [ "$INSTALL_BASE" = true ] || [ "$INSTALL_TASK" = true ] && needs_env=true
    [ -n "$PIP_DEPS" ] || [ -n "$SRC_DEPS" ] && needs_env=true

    if [ "$needs_env" = false ]; then
        log_success "Installation complete"
        print_header "Installation Complete"
        return 0
    fi

    # Phase 2: Environment setup
    setup_package_manager
    setup_environment || die "Environment setup failed"
    log_info "Env: $(get_current_env) | $(python --version 2>/dev/null || echo 'Python: N/A')"

    # Phase 3: Dev dependencies
    [ "$INSTALL_DEV" = true ] && { print_header "Dev Dependencies"; install_dev_deps; }

    # Phase 4: Task dependencies (or selective installation)
    if [ -n "$PIP_DEPS" ] || [ -n "$SRC_DEPS" ]; then
        # Selective installation mode
        print_header "Selective Installation"
        install_pip_deps
        install_src_deps
    else
        # Normal phase-based installation
        if [ "$TASK" = "all" ]; then
            for task in $(get_valid_tasks); do
                [ "$task" = "all" ] && continue
                install_task "$task"
            done
        else
            install_task "$TASK"
        fi
    fi

    print_header "Installation Complete"
}

main "$@"

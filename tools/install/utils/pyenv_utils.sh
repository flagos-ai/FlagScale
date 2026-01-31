#!/bin/bash
# =============================================================================
# Python Environment Utilities (conda, uv, pip)
# =============================================================================
#
# Provides Python environment management for conda, uv, and pip.
# Supports conda for CI/CD compatibility and uv for modern workflows.
#
# Usage:
#   source pyenv_utils.sh
#   activate_conda -d $DEBUG "env_name" "conda_path"  # For conda
#   activate_uv_env -d $DEBUG [venv_path]             # For uv
# =============================================================================

_PYENV_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_PYENV_UTILS_DIR/utils.sh"

# =============================================================================
# Environment Detection
# =============================================================================

# Check if running in uv environment
is_uv_env() {
    [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] || \
    ([ -n "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_DEFAULT_ENV:-}" ])
}

# Check if running in conda environment
is_conda_active() {
    [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]
}

# Note: has_uv, has_pip, has_conda are defined in utils.sh

# Get current environment name
get_current_env() {
    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        echo "$CONDA_DEFAULT_ENV"
    elif [ -n "${VIRTUAL_ENV:-}" ]; then
        basename "$VIRTUAL_ENV"
    else
        echo "base"
    fi
}

# =============================================================================
# UV Environment Activation
# =============================================================================

# Activate uv virtual environment
# Usage: activate_uv_env -d <debug> [venv_path]
activate_uv_env() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local venv_path=${1:-${UV_PROJECT_ENVIRONMENT:-"/opt/venv"}}

    if [ "$debug" = true ]; then
        log_info "Activating UV env: $venv_path"
        echo "    [dry-run] source $venv_path/bin/activate" >&2
        return 0
    fi

    [ ! -d "$venv_path" ] && { has_uv && uv venv "$venv_path" || { log_error "Cannot create venv"; return 1; }; }
    [ -f "$venv_path/bin/activate" ] || { log_error "Invalid venv: $venv_path"; return 1; }
    run_source -d $debug -m "Activating UV env: $venv_path" "$venv_path/bin/activate"
    export UV_PROJECT_ENVIRONMENT="$venv_path"
    return 0
}

# =============================================================================
# Conda Activation
# =============================================================================

# Check if conda environment exists
# Usage: conda_env_exists <env_name> <conda_path>
conda_env_exists() {
    local env_name=$1
    local conda_path=$2
    CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "$conda_path/bin/conda" env list 2>/dev/null | grep -q "^${env_name} " || \
    [ -d "$conda_path/envs/$env_name" ]
}

# Configure conda for non-interactive use (solver)
# Usage: configure_conda_silent <conda_path>
configure_conda_silent() {
    local conda_path=$1
    # Set solver to classic (libmamba may not be available)
    CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "$conda_path/bin/conda" config --set solver classic >/dev/null 2>&1 || true
}

# Accept conda Terms of Service and configure solver non-interactively (legacy alias)
# Usage: accept_conda_tos <conda_path>
accept_conda_tos() {
    configure_conda_silent "$1"
}

# Create conda environment if it doesn't exist
# Usage: create_conda_env -d <debug> <env_name> <conda_path> [python_version]
create_conda_env() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local env_name=$1
    local conda_path=$2
    local python_version=${3:-"3.12"}

    if [ "$debug" = true ]; then
        log_info "Creating conda env: $env_name (python=$python_version)"
        echo "    [dry-run] $conda_path/bin/conda create -y -n $env_name python=$python_version" >&2
        return 0
    fi

    if conda_env_exists "$env_name" "$conda_path"; then
        log_info "Conda env '$env_name' already exists"
        return 0
    fi

    # Configure conda for non-interactive use
    accept_conda_tos "$conda_path"

    log_step "Creating conda env: $env_name (python=$python_version)"
    CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "$conda_path/bin/conda" create -y -n "$env_name" "python=$python_version" || {
        log_error "Failed to create conda env: $env_name"
        return 1
    }
    log_success "Conda env '$env_name' created"
    return 0
}

# Activate conda environment (creates if doesn't exist)
# Usage: activate_conda -d <debug> <env_name> <conda_path> [python_version]
activate_conda() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local env_name=$1
    local conda_path=${2:-""}
    local python_version=${3:-"3.12"}

    [ -z "$conda_path" ] && { log_error "conda_path required"; return 1; }

    if [ "$debug" = true ]; then
        log_info "Activating conda env: $env_name (python=$python_version)"
        echo "    [dry-run] source $conda_path/etc/profile.d/conda.sh" >&2
        echo "    [dry-run] conda create -y -n $env_name python=$python_version (if not exists)" >&2
        echo "    [dry-run] conda activate $env_name" >&2
        return 0
    fi

    [ ! -f "$conda_path/etc/profile.d/conda.sh" ] && { log_error "Invalid conda: $conda_path"; return 1; }
    run_source -d $debug -m "Loading conda from $conda_path" "$conda_path/etc/profile.d/conda.sh"

    # Create env if it doesn't exist
    create_conda_env -d $debug "$env_name" "$conda_path" "$python_version" || return 1

    log_info "Activating conda env: $env_name"
    conda activate "$env_name" || { log_error "Failed: conda activate $env_name"; return 1; }
    return 0
}

# =============================================================================
# Legacy Functions (for backwards compatibility)
# =============================================================================

# Get current conda environment name
get_conda_env() {
    get_current_env
}

# Display environment info
display_env_info() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Environment Information"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if command -v python &> /dev/null; then
        echo "Python: $(which python)"
        echo "Version: $(python --version 2>&1)"
    else
        echo "Python: not found"
    fi

    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        echo "Conda env: $CONDA_DEFAULT_ENV"
    fi

    if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ]; then
        echo "UV venv: $UV_PROJECT_ENVIRONMENT"
    elif [ -n "${VIRTUAL_ENV:-}" ]; then
        echo "Virtual env: $VIRTUAL_ENV"
    fi

    if [ -z "${CONDA_DEFAULT_ENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
        echo "Environment: system"
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

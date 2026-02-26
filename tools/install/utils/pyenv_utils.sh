#!/bin/bash
# =============================================================================
# Python Environment Utilities
# =============================================================================
#
# Environment activation for conda and uv.
# Used by GitHub workflows to activate pre-installed environments.
#
# Usage:
#   source pyenv_utils.sh
#   activate_conda "env_name" "/path/to/conda"
#   activate_uv_env "/path/to/venv"
#
# With debug mode (optional):
#   activate_conda -d true "env_name" "/path/to/conda"
#   activate_uv_env -d true "/path/to/venv"
# =============================================================================

_PYENV_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_PYENV_UTILS_DIR/utils.sh"

# =============================================================================
# UV Environment
# =============================================================================

# Activate uv virtual environment
# Usage: activate_uv_env [-d debug] [venv_path]
activate_uv_env() {
    local debug=false
    [[ "$1" == "-d" ]] && { debug="$2"; shift 2; }

    local venv_path=${1:-${UV_PROJECT_ENVIRONMENT:-"/opt/venv"}}

    if [ "$debug" = true ]; then
        log_info "[dry-run] Activate UV env: $venv_path"
        return 0
    fi

    [ ! -d "$venv_path" ] && { log_error "Venv not found: $venv_path"; return 1; }
    [ ! -f "$venv_path/bin/activate" ] && { log_error "Invalid venv: $venv_path"; return 1; }

    source "$venv_path/bin/activate"
    export UV_PROJECT_ENVIRONMENT="$venv_path"
    log_info "Activated UV env: $venv_path"
    return 0
}

# =============================================================================
# Conda Environment
# =============================================================================

# Check if conda environment exists
conda_env_exists() {
    local env_name=$1
    local conda_path=$2
    CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "$conda_path/bin/conda" env list 2>/dev/null | grep -q "^${env_name} " || \
    [ -d "$conda_path/envs/$env_name" ]
}

# Create conda environment if it doesn't exist
# Usage: create_conda_env [-d debug] <env_name> <conda_path> [python_version]
create_conda_env() {
    local debug=false
    [[ "$1" == "-d" ]] && { debug="$2"; shift 2; }

    local env_name=$1
    local conda_path=$2
    local python_version=${3:-"3.12"}

    if [ "$debug" = true ]; then
        log_info "[dry-run] Create conda env: $env_name (python=$python_version)"
        return 0
    fi

    if conda_env_exists "$env_name" "$conda_path"; then
        log_info "Conda env '$env_name' already exists"
        return 0
    fi

    # Configure solver for non-interactive use
    CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "$conda_path/bin/conda" config --set solver classic >/dev/null 2>&1 || true

    log_info "Creating conda env: $env_name (python=$python_version)"
    CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "$conda_path/bin/conda" create -y -n "$env_name" "python=$python_version" || {
        log_error "Failed to create conda env: $env_name"
        return 1
    }
    log_success "Conda env '$env_name' created"
    return 0
}

# Activate conda environment (creates if doesn't exist)
# Usage: activate_conda [-d debug] <env_name> <conda_path> [python_version]
activate_conda() {
    local debug=false
    [[ "$1" == "-d" ]] && { debug="$2"; shift 2; }

    local env_name=$1
    local conda_path=${2:-""}
    local python_version=${3:-"3.12"}

    [ -z "$conda_path" ] && { log_error "conda_path required"; return 1; }

    if [ "$debug" = true ]; then
        log_info "[dry-run] Activate conda env: $env_name at $conda_path"
        return 0
    fi

    [ ! -f "$conda_path/etc/profile.d/conda.sh" ] && { log_error "Invalid conda: $conda_path"; return 1; }

    source "$conda_path/etc/profile.d/conda.sh"
    create_conda_env "$env_name" "$conda_path" "$python_version" || return 1

    log_info "Activating conda env: $env_name"
    conda activate "$env_name" || { log_error "Failed: conda activate $env_name"; return 1; }
    return 0
}

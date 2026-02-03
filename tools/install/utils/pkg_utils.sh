#!/bin/bash
# =============================================================================
# Package Manager Utilities
# =============================================================================
#
# Unified interface for pip/uv/conda package installation.
#
# Environment:
#   FLAGSCALE_PKG_MGR - "uv", "pip", or "conda" (default: uv)
#   FLAGSCALE_CONDA - path to conda installation
#   FLAGSCALE_ENV_NAME - conda environment name (optional)
# =============================================================================

_PKG_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_PKG_UTILS_DIR/utils.sh"

# =============================================================================
# Package Manager
# =============================================================================

get_pkg_manager() {
    echo "${FLAGSCALE_PKG_MGR:-uv}"
}

# Get the pip command for the current package manager
# Returns the full path to pip for conda environments
get_pip_cmd() {
    local manager=$(get_pkg_manager)
    case "$manager" in
        conda)
            local conda_path="${FLAGSCALE_CONDA:-/opt/flagscale/miniconda3}"
            local env_name="${FLAGSCALE_ENV_NAME:-}"
            if [ -n "$env_name" ]; then
                echo "$conda_path/envs/$env_name/bin/pip"
            else
                echo "$conda_path/bin/pip"
            fi
            ;;
        *)
            echo "pip"
            ;;
    esac
}

# =============================================================================
# Package Checks
# =============================================================================

is_package_installed() {
    local package=$1
    local normalized=$(echo "$package" | tr '-' '_')
    local pip_cmd=$(get_pip_cmd)
    $pip_cmd show "$normalized" &>/dev/null || $pip_cmd show "$package" &>/dev/null
}

get_package_version() {
    local package=$1
    local normalized=$(echo "$package" | tr '-' '_')
    local pip_cmd=$(get_pip_cmd)
    $pip_cmd show "$normalized" 2>/dev/null | grep -i "^Version:" | awk '{print $2}' || \
    $pip_cmd show "$package" 2>/dev/null | grep -i "^Version:" | awk '{print $2}'
}

# Check if should build from source (not installed or FLAGSCALE_FORCE_BUILD=true)
should_build_package() {
    local package=$1

    if [ "${FLAGSCALE_FORCE_BUILD:-false}" = true ]; then
        log_info "Force build enabled, will build $package"
        return 0
    fi

    if is_package_installed "$package"; then
        local version=$(get_package_version "$package")
        log_info "$package already installed (version: ${version:-unknown}), skipping"
        return 1
    fi
    return 0
}

# =============================================================================
# Phase Control
# =============================================================================
# Environment variables (from install.sh):
#   FLAGSCALE_INSTALL_SYSTEM/DEV/BASE/TASK - true/false
#   FLAGSCALE_PIP_DEPS - comma-separated pip packages
#   FLAGSCALE_SRC_DEPS - comma-separated source deps

is_phase_enabled() {
    local phase="$1"
    case "$phase" in
        system) [ "${FLAGSCALE_INSTALL_SYSTEM:-true}" = true ] ;;
        dev)    [ "${FLAGSCALE_INSTALL_DEV:-true}" = true ] ;;
        base)   [ "${FLAGSCALE_INSTALL_BASE:-true}" = true ] ;;
        task)   [ "${FLAGSCALE_INSTALL_TASK:-true}" = true ] ;;
        *)      return 1 ;;
    esac
}

is_in_override() {
    local type="$1" item="$2" list=""
    case "$type" in
        pip) list="${FLAGSCALE_PIP_DEPS:-}" ;;
        src) list="${FLAGSCALE_SRC_DEPS:-}" ;;
        *)   return 1 ;;
    esac
    [ -n "$list" ] && echo ",$list," | grep -q ",$item,"
}

# Should install source dep?
# Usage: should_install_src <phase> <dep_name>
should_install_src() {
    local phase="$1" item="$2"
    is_phase_enabled "$phase" && return 0
    is_in_override src "$item" && return 0
    return 1
}

# =============================================================================
# Phase-Scoped Filtering
# =============================================================================

# Get pip-deps that match a requirements file
get_pip_deps_for_requirements() {
    local req_file="$1"
    local pip_deps="${FLAGSCALE_PIP_DEPS:-}"
    local matched=""

    [ -z "$pip_deps" ] || [ ! -f "$req_file" ] && return 0

    for pkg in $(echo "$pip_deps" | tr ',' ' '); do
        grep -qiE "^${pkg}([=<>!~\[]|$)" "$req_file" 2>/dev/null && matched="$matched $pkg"
    done
    echo "$matched" | xargs
}

# Check if any src-deps match the valid list
has_src_deps_for_phase() {
    local valid_deps="$*"
    local src_deps="${FLAGSCALE_SRC_DEPS:-}"
    [ -z "$src_deps" ] && return 1

    for dep in $(echo "$src_deps" | tr ',' ' '); do
        for valid in $valid_deps; do
            [ "$dep" = "$valid" ] && return 0
        done
    done
    return 1
}

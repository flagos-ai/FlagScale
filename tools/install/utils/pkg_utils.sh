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
#   FLAGSCALE_ONLY_PIP - true/false (skip apt and source builds)

# Check if only pip mode is enabled (skip apt and source builds)
is_only_pip() {
    [ "${FLAGSCALE_ONLY_PIP:-false}" = true ]
}

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
# Priority: --src-deps override > --only-pip > phase enabled
should_install_src() {
    local phase="$1" item="$2"
    # Override flags have highest priority
    is_in_override src "$item" && return 0
    # Skip source builds in only-pip mode (unless overridden above)
    is_only_pip && return 1
    is_phase_enabled "$phase" && return 0
    return 1
}

# =============================================================================
# Phase-Scoped Filtering
# =============================================================================

# Expand requirements file content, resolving -r includes recursively
# Usage: expand_requirements_file <req_file>
expand_requirements_file() {
    local req_file="$1"
    local base_dir
    base_dir="$(dirname "$req_file")"

    [ ! -f "$req_file" ] && return 0

    while IFS= read -r line || [ -n "$line" ]; do
        # Handle -r includes (e.g., "-r ../common.txt" or "-r common.txt")
        if echo "$line" | grep -qE '^-r[[:space:]]+'; then
            local included_file
            included_file="$(echo "$line" | sed 's/^-r[[:space:]]*//')"
            # Resolve relative path from the base directory
            if [ "${included_file#/}" = "$included_file" ]; then
                included_file="$base_dir/$included_file"
            fi
            # Recursively expand the included file
            expand_requirements_file "$included_file"
        else
            echo "$line"
        fi
    done < "$req_file"
}

# Get pip-deps that match a requirements file (resolves -r includes)
get_pip_deps_for_requirements() {
    local req_file="$1"
    local pip_deps="${FLAGSCALE_PIP_DEPS:-}"
    local matched=""

    [ -z "$pip_deps" ] || [ ! -f "$req_file" ] && return 0

    # Expand requirements file to include all -r references
    local expanded_content
    expanded_content="$(expand_requirements_file "$req_file")"

    for pkg in $(echo "$pip_deps" | tr ',' ' '); do
        echo "$expanded_content" | grep -qiE "^${pkg}([=<>!~\[]|$)" 2>/dev/null && matched="$matched $pkg"
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

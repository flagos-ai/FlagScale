#!/bin/bash
# =============================================================================
# Package Manager Utilities
# =============================================================================
#
# Provides a unified interface for package installation supporting:
#   - pip: Standard Python package installer
#   - uv: Fast, modern Python package installer (uses uv pip)
#   - conda: Conda package manager (uses conda install for conda packages,
#            pip for PyPI packages within conda environments)
#
# Usage:
#   source pkg_utils.sh
#   set_pkg_manager "uv"  # or "pip" or "conda"
#   pkg_install -d $DEBUG -r requirements.txt
#   pkg_install -d $DEBUG package1 package2
#
# Environment:
#   FLAGSCALE_PKG_MANAGER - Set to "uv", "pip", or "conda" (default: uv)
#
# =============================================================================

_PKG_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_PKG_UTILS_DIR/utils.sh"

# =============================================================================
# Package Manager Functions
# =============================================================================
# Note: has_uv, has_pip, has_conda are defined in utils.sh

# Get current package manager
# Returns: "uv", "pip", or "conda"
get_pkg_manager() {
    echo "${FLAGSCALE_PKG_MANAGER:-uv}"
}

# Set package manager
set_pkg_manager() {
    local manager=$1
    case "$manager" in
        uv|pip|conda)
            export FLAGSCALE_PKG_MANAGER="$manager"
            ;;
        *)
            log_error "Unknown package manager: $manager"
            return 1
            ;;
    esac
}

# =============================================================================
# Installation Functions
# =============================================================================

# Install packages using the configured package manager
# Usage: pkg_install -d <debug> [-r requirements.txt] [package1 package2 ...]
#
# For conda: Uses pip within the conda environment for requirements files
#            (conda install is used for conda-specific packages via pkg_conda_install)
pkg_install() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local manager=$(get_pkg_manager)
    local args=("$@")

    case "$manager" in
        uv)
            _uv_install "$debug" "${args[@]}"
            ;;
        pip)
            _pip_install "$debug" "${args[@]}"
            ;;
        conda)
            # For requirements files, use pip within conda environment
            # This is the standard approach for PyPI packages in conda
            _conda_pip_install "$debug" "${args[@]}"
            ;;
        *)
            log_error "Unknown package manager: $manager"
            return 1
            ;;
    esac
}

# Install from requirements file
# Usage: pkg_install_requirements -d <debug> <requirements_file>
pkg_install_requirements() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local req_file=$1

    if [ ! -f "$req_file" ] && [ "$debug" != true ]; then
        log_error "Requirements file not found: $req_file"
        return 1
    fi

    pkg_install -d $debug -r "$req_file"
}

# Install conda packages directly (only for conda manager)
# Usage: pkg_conda_install -d <debug> package1 package2 ...
pkg_conda_install() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local manager=$(get_pkg_manager)

    if [ "$manager" != "conda" ]; then
        log_warn "pkg_conda_install called but manager is $manager, using pip instead"
        pkg_install -d $debug "$@"
        return
    fi

    if ! has_conda && [ "$debug" != true ]; then
        log_error "Conda not available"
        return 1
    fi

    run_cmd -d $debug conda install -y "$@"
}

# Install package from source (editable mode)
# Usage: pkg_install_editable -d <debug> <path> [extra_args...]
pkg_install_editable() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local path=$1
    shift
    local extra_args=("$@")
    local manager=$(get_pkg_manager)

    case "$manager" in
        uv)
            run_cmd -d $debug uv pip install -e "$path" "${extra_args[@]}"
            ;;
        pip|conda)
            run_cmd -d $debug pip install --root-user-action=ignore -e "$path" "${extra_args[@]}"
            ;;
    esac
}

# Install package without build isolation (for packages with complex builds)
# Usage: pkg_install_no_isolation -d <debug> <path>
pkg_install_no_isolation() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local path=$1
    local manager=$(get_pkg_manager)

    case "$manager" in
        uv)
            run_cmd -d $debug uv pip install --no-build-isolation "$path" -v
            ;;
        pip|conda)
            run_cmd -d $debug pip install --root-user-action=ignore --no-build-isolation "$path" -vvv
            ;;
    esac
}

# =============================================================================
# Internal Functions
# =============================================================================

# pip install wrapper
_pip_install() {
    local debug=$1; shift
    run_cmd -d $debug pip install --root-user-action=ignore "$@"
}

# uv pip install wrapper
_uv_install() {
    local debug=$1; shift
    run_cmd -d $debug uv pip install "$@"
}

# conda pip install wrapper (uses pip within conda environment)
_conda_pip_install() {
    local debug=$1; shift
    # In conda environments, pip is the standard way to install PyPI packages
    run_cmd -d $debug pip install --root-user-action=ignore "$@"
}

# =============================================================================
# Display Functions
# =============================================================================

# Display package manager info
display_pkg_info() {
    local manager=$(get_pkg_manager)

    echo "Package Manager: $manager"

    case "$manager" in
        uv)
            if has_uv; then
                echo "UV Version: $(uv --version 2>/dev/null || echo 'unknown')"
                echo "UV Environment: ${UV_PROJECT_ENVIRONMENT:-${VIRTUAL_ENV:-not set}}"
            fi
            ;;
        pip)
            if has_pip; then
                echo "Pip Version: $(pip --version 2>/dev/null | awk '{print $2}' || echo 'unknown')"
            fi
            ;;
        conda)
            if has_conda; then
                echo "Conda Version: $(conda --version 2>/dev/null | awk '{print $2}' || echo 'unknown')"
                echo "Conda Environment: ${CONDA_DEFAULT_ENV:-base}"
            fi
            if has_pip; then
                echo "Pip Version: $(pip --version 2>/dev/null | awk '{print $2}' || echo 'unknown')"
            fi
            ;;
    esac
}

# Get install command for display purposes
get_install_cmd() {
    local manager=$(get_pkg_manager)
    case "$manager" in
        uv)    echo "uv pip install" ;;
        pip)   echo "pip install" ;;
        conda) echo "pip install (in conda)" ;;
    esac
}

# =============================================================================
# Package Check Functions
# =============================================================================

# Check if a Python package is installed
# Usage: is_package_installed <package_name>
# Returns: 0 if installed, 1 if not
is_package_installed() {
    local package=$1
    # Normalize package name (replace - with _ for pip show compatibility)
    local normalized=$(echo "$package" | tr '-' '_')
    pip show "$normalized" &>/dev/null || pip show "$package" &>/dev/null
}

# Get installed version of a package
# Usage: get_package_version <package_name>
# Returns: version string or empty if not installed
get_package_version() {
    local package=$1
    local normalized=$(echo "$package" | tr '-' '_')
    pip show "$normalized" 2>/dev/null | grep -i "^Version:" | awk '{print $2}' || \
    pip show "$package" 2>/dev/null | grep -i "^Version:" | awk '{print $2}'
}

# Check if we should build a package from source
# Usage: should_build_package <package_name>
# Returns: 0 if should build (not installed or FORCE_BUILD=true), 1 if skip
# Environment: FLAGSCALE_FORCE_BUILD - Set to "true" to force rebuild
should_build_package() {
    local package=$1

    # Always build if force-build is set
    if [ "${FLAGSCALE_FORCE_BUILD:-false}" = true ]; then
        log_info "Force build enabled, will build $package"
        return 0
    fi

    # Check if already installed
    if is_package_installed "$package"; then
        local version=$(get_package_version "$package")
        log_info "$package already installed (version: ${version:-unknown}), skipping build"
        return 1
    fi

    # Not installed, should build
    return 0
}

# Check if a specific source dependency should be installed
# Usage: should_install_dep <dep_name>
# Returns: 0 if should install, 1 if skip
# When FLAGSCALE_SRC_DEPS is empty, all deps are installed (default behavior)
# When set, only specified deps are installed (e.g., "megatron-lm" or "apex,megatron-lm")
# Environment: FLAGSCALE_SRC_DEPS - Comma-separated list of deps to install
should_install_dep() {
    local dep_name="$1"
    local src_deps="${FLAGSCALE_SRC_DEPS:-}"

    # Empty means install all
    if [ -z "$src_deps" ]; then
        return 0
    fi

    # Check if dep_name is in the comma-separated list
    if echo ",$src_deps," | grep -q ",$dep_name,"; then
        return 0
    fi

    log_info "Skipping $dep_name (not in --src-deps list)"
    return 1
}

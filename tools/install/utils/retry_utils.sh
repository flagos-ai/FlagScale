#!/bin/bash
# =============================================================================
# Retry Utilities
# =============================================================================
#
# Retry wrappers for network-dependent operations (pip install, git clone).
# =============================================================================

# Source utils for logging functions and package manager
_RETRY_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_RETRY_UTILS_DIR/utils.sh"
source "$_RETRY_UTILS_DIR/pkg_utils.sh"

# Retry command with specified attempts
# Usage: retry -d <debug> <retries> <cmd>
retry() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local retries=$1
    shift
    local cmd="$*"
    local count=0

    if [ "$debug" = true ]; then
        echo "    [dry-run] $cmd" >&2
        return 0
    fi

    until eval "$cmd"; do
        count=$((count + 1))
        [ $count -ge $retries ] && { log_error "Failed after $retries attempts"; return 1; }
        log_warn "Retry $count/$retries in 5s..."
        sleep 5
    done
    return 0
}

# Retry pip/uv install from requirements file
# Handles per-package annotations: if the file contains # [--option] comments,
# annotated packages are installed separately with their required flags.
# Usage: retry_pip_install -d <debug> <requirements_file> [retries]
retry_pip_install() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local requirements_file=$1
    local retries=${2:-3}
    local manager=$(get_pkg_manager)
    local pip_cmd=$(get_pip_cmd)

    [ ! -f "$requirements_file" ] && [ "$debug" != true ] && { log_error "Not found: $requirements_file"; return 1; }

    # Check for per-package annotations
    local annotations=""
    [ -f "$requirements_file" ] && annotations="$(parse_pkg_annotations "$requirements_file")"

    if [ -z "$annotations" ]; then
        # No annotations — install normally (fast path)
        log_info "Installing $(basename "$requirements_file")..."
        case "$manager" in
            uv)    retry -d $debug $retries "uv pip install -r '$requirements_file'" ;;
            *)     retry -d $debug $retries "$pip_cmd install --root-user-action=ignore -r '$requirements_file'" ;;
        esac
    else
        # Has annotations — filter and install separately
        local filtered
        filtered="$(mktemp)"
        create_filtered_requirements "$requirements_file" "$filtered"

        # Install normal packages from filtered file
        log_info "Installing $(basename "$requirements_file") (normal packages)..."
        case "$manager" in
            uv)    retry -d $debug $retries "uv pip install -r '$filtered'" ;;
            *)     retry -d $debug $retries "$pip_cmd install --root-user-action=ignore -r '$filtered'" ;;
        esac
        local rc=$?
        rm -f "$filtered"
        [ $rc -ne 0 ] && return $rc

        # Install annotated packages with their per-package options
        while IFS=$'\t' read -r pkg opts; do
            local pkg_name
            pkg_name="$(echo "$pkg" | cut -d@ -f1 | xargs)"
            log_info "Installing $pkg_name with $opts..."
            case "$manager" in
                uv)    retry -d $debug $retries "uv pip install $opts '$pkg'" ;;
                *)     retry -d $debug $retries "$pip_cmd install --root-user-action=ignore $opts '$pkg'" ;;
            esac || return 1
        done <<< "$annotations"
    fi
}

# Retry git clone with options
# Usage: retry_git_clone -d <debug> [--branch BRANCH] [--depth N] [--recursive] <repo_url> <target_dir> [retries]
retry_git_clone() {
    local debug=false branch="" depth="" recursive=""

    while [[ "$1" == -* ]]; do
        case "$1" in
            -d) debug="$2"; shift 2 ;;
            --branch) branch="$2"; shift 2 ;;
            --depth) depth="$2"; shift 2 ;;
            --recursive) recursive="--recursive"; shift ;;
            *) break ;;
        esac
    done

    local repo_url=$1
    local target_dir=$2
    local retries=${3:-3}

    # Build clone options
    local opts=""
    [ -n "$branch" ] && opts="$opts --branch $branch"
    [ -n "$depth" ] && opts="$opts --depth $depth"
    [ -n "$recursive" ] && opts="$opts $recursive"

    log_info "Cloning $(basename "$repo_url" .git)"
    retry -d $debug $retries "rm -rf '$target_dir' && git clone$opts '$repo_url' '$target_dir'"
}

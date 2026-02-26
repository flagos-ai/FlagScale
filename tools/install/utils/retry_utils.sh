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
# Usage: retry_pip_install -d <debug> <requirements_file> [retries]
retry_pip_install() {
    local debug=false
    if [[ "$1" == "-d" ]]; then
        debug="$2"; shift 2
    fi

    local requirements_file=$1
    local retries=${2:-3}
    local manager=$(get_pkg_manager)

    [ ! -f "$requirements_file" ] && [ "$debug" != true ] && { log_error "Not found: $requirements_file"; return 1; }

    log_info "Installing $(basename "$requirements_file")..."
    local pip_cmd=$(get_pip_cmd)
    case "$manager" in
        uv)    retry -d $debug $retries "uv pip install -r '$requirements_file'" ;;
        *)     retry -d $debug $retries "$pip_cmd install --root-user-action=ignore -r '$requirements_file'" ;;
    esac
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

#!/bin/bash
# Common utility functions for install scripts

# =============================================================================
# Error Handling (explicit, no traps)
# =============================================================================

# Global variable to track the current step for error messages
CURRENT_STEP=""

# Print error message and exit
# Usage: die "Error message" [exit_code]
die() {
    local msg="$1"
    local code="${2:-1}"

    echo "" >&2
    echo "══════════════════════════════════════════════════════════════════" >&2
    echo "  ✗ INSTALLATION FAILED" >&2
    echo "══════════════════════════════════════════════════════════════════" >&2
    [ -n "$CURRENT_STEP" ] && echo "  Step: $CURRENT_STEP" >&2
    echo "  Error: $msg" >&2
    echo "  Exit code: $code" >&2
    echo "" >&2
    echo "  Please check the output above for details." >&2
    echo "══════════════════════════════════════════════════════════════════" >&2
    exit "$code"
}

# Check last command and die if failed
# Usage: command || check_error "Failed to do X"
check_error() {
    local code=$?
    [ $code -ne 0 ] && die "$1" "$code"
    return 0
}

# Setup error handling for a script (no-op for compatibility, kept for transition)
# Scripts should use explicit error checking: cmd || die "message"
setup_error_handling() {
    # No automatic error handling - use explicit checks instead
    # Example: some_command || die "some_command failed"
    :
}

# Set the current step name for error reporting
# Usage: set_step "Installing dependencies"
set_step() {
    CURRENT_STEP="$1"
    log_step "$1"
}

# =============================================================================
# Command Execution with Debug Support
# =============================================================================

# Run command or print in debug mode
# Usage: run_cmd -d <true|false> [-m "message"] command args...
#   -d flag     Debug flag (required): true=dry-run, false=execute
#   -m "msg"    Optional message to log before running
run_cmd() {
    local msg="" debug="false"
    while [[ "$1" == -* ]]; do
        case "$1" in
            -m) msg="$2"; shift 2 ;;
            -d) debug="$2"; shift 2 ;;
            *) break ;;
        esac
    done

    [ -n "$msg" ] && log_info "$msg"

    if [ "$debug" = true ]; then
        echo "    [dry-run] $*" >&2
        return 0
    fi
    "$@"
}

# Run source command (for shell builtins that affect current shell)
# Usage: run_source -d <true|false> [-m "message"] file
#   -d flag     Debug flag (required): true=dry-run, false=execute
#   -m "msg"    Optional message to log before running
run_source() {
    local msg="" debug="false"
    while [[ "$1" == -* ]]; do
        case "$1" in
            -m) msg="$2"; shift 2 ;;
            -d) debug="$2"; shift 2 ;;
            *) break ;;
        esac
    done

    [ -n "$msg" ] && log_info "$msg"

    if [ "$debug" = true ]; then
        echo "    [dry-run] source $1" >&2
        return 0
    fi
    source "$1"
}

# =============================================================================
# Logging Functions - Clean, minimal output
# All output goes to stderr to avoid buffering issues
# =============================================================================
log_info() {
    echo "  · $*" >&2
}

log_warn() {
    echo "  ! $*" >&2
}

log_error() {
    echo "  ✗ $*" >&2
}

log_success() {
    echo "  ✓ $*" >&2
}

log_step() {
    echo "→ $*" >&2
}

# Get the project root directory
get_project_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir/../../.."
    pwd
}

# Check Python version meets minimum requirement
check_python_version() {
    local min_version=${1:-"3.10"}

    if ! command -v python &> /dev/null; then
        log_error "Python not found"
        return 1
    fi

    local python_version
    python_version=$(python --version 2>&1 | awk '{print $2}')

    local py_major py_minor min_major min_minor
    py_major=$(echo "$python_version" | cut -d. -f1)
    py_minor=$(echo "$python_version" | cut -d. -f2)
    min_major=$(echo "$min_version" | cut -d. -f1)
    min_minor=$(echo "$min_version" | cut -d. -f2)

    if [ "$py_major" -lt "$min_major" ] || \
       ([ "$py_major" -eq "$min_major" ] && [ "$py_minor" -lt "$min_minor" ]); then
        log_error "Python $min_version+ required (found $python_version)"
        return 1
    fi

    log_info "Python $python_version"
    return 0
}

# Check if we're in a conda environment
is_conda_env() {
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        return 0
    else
        return 1
    fi
}

# Get current conda environment name
get_conda_env() {
    if is_conda_env; then
        echo "$CONDA_DEFAULT_ENV"
    else
        echo "none"
    fi
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Common command checks (used by pkg_utils.sh and pyenv_utils.sh)
has_uv() { command -v uv &>/dev/null; }
has_pip() { command -v pip &>/dev/null; }
has_conda() { command -v conda &>/dev/null; }

# Print section header
print_header() {
    echo "" >&2
    echo "══════════════════════════════════════════════════════════════════" >&2
    echo "  $*" >&2
    echo "══════════════════════════════════════════════════════════════════" >&2
}

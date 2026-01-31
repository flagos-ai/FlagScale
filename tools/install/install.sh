#!/bin/bash
# =============================================================================
# FlagScale Dependency Installation
# =============================================================================
#
# Installs all dependencies for FlagScale tasks in phases:
#   Phase 1: System deps - apt packages, python, openmpi
#   Phase 2: Environment setup - activate conda/uv env
#   Phase 3: Dev deps - build, lint, test tools (use --no-dev to skip)
#   Phase 4: Task deps (per task):
#       - base: Base requirements (base.txt) + install_base.sh
#       - task: Task requirements (<task>.txt) + install_<task>.sh (source repos)
#
# By default, all phases are installed. Use --no-dev to skip dev deps.
#
# FLAGSCALE_HOME is the root directory for all installations:
#   - $FLAGSCALE_HOME/miniconda3  - Conda installation
#   - $FLAGSCALE_HOME/venv        - UV virtual environment
#   - $FLAGSCALE_HOME/deps        - Source dependencies (Megatron, etc.)
#   - $FLAGSCALE_HOME/downloads   - Cached downloads (miniconda, etc.)
#
# Usage:
#   ./install.sh --platform PLATFORM --task TASK [OPTIONS]
#
# Examples:
#   ./install.sh --platform cuda --task train                             # Full installation
#   ./install.sh --platform cuda --task train --no-system                 # Skip system packages
#   ./install.sh --platform cuda --task train --install-dir /home/user/opt
#   ./install.sh --platform cuda --task train --pkg-mgr conda --env-name flagscale-train
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
PLATFORM=""  # Required: use --platform to specify
RETRY_COUNT=3
PKG_MGR="uv"        # pip, uv, conda (default: uv)
ENV_NAME=""         # Environment name: conda env name (for conda only)
DEBUG=false         # Debug mode: print commands without executing
PYTHON_VERSION="$(get_common "python")"  # Python version from versions.json

# Root installation directory (single source of truth for all paths)
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"

# Installation phases (default: install all)
INSTALL_SYSTEM=true       # Install system dependencies (apt, python, openmpi)
INSTALL_DEV=true          # Install dev dependencies (build, lint, test tools)
INSTALL_BASE=true         # Install base dependencies (common packages)
INSTALL_TASK=true         # Install task dependencies (pip requirements + source repos)

# Build options
FORCE_BUILD=false         # Force rebuild source deps even if already installed

# PyPI index URLs (optional, for custom mirrors)
# These are exported as env vars for pip/uv to pick up automatically
INDEX_URL="${PIP_INDEX_URL:-}"
EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"

# Get valid tasks from install scripts
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

# Install dev dependencies (build, lint, test tools)
# Called when --dev flag is set, after environment setup
install_dev_deps() {
    set_step "Installing dev dependencies"

    local dev_req="$PROJECT_ROOT/requirements/dev.txt"
    [ ! -f "$dev_req" ] && [ "$DEBUG" != true ] && { log_warn "dev.txt not found, skipping"; return 0; }

    log_info "Dev requirements: dev.txt"
    retry_pip_install -d $DEBUG "$dev_req" "$RETRY_COUNT" || die "Dev requirements installation failed"
    log_success "Dev dependencies installed"
}

# Install system dependencies (apt packages, python, openmpi)
# This is a ONE-TIME setup shared by ALL tasks - runs once before any task
install_system_deps() {
    set_step "Installing system dependencies"

    local system_script="$SCRIPT_DIR/install_system.sh"
    [ ! -f "$system_script" ] && die "install_system.sh not found"

    local args=""
    [ "$INSTALL_DEV" = false ] && args="--no-dev"
    [ -n "$PLATFORM" ] && args="$args --platform $PLATFORM"
    [ -n "$PKG_MGR" ] && args="$args --pkg-mgr $PKG_MGR"
    [ "$DEBUG" = true ] && args="$args --debug"

    # Pass FLAGSCALE_HOME to install_system.sh via environment
    FLAGSCALE_HOME="$FLAGSCALE_HOME" "$system_script" $args || die "System dependencies installation failed"
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

  Phase Control (default: install all phases):
    --no-system            Skip system packages (apt, python, openmpi)
    --no-dev               Skip dev dependencies (build, lint, test)
    --no-base              Skip base dependencies
    --no-task              Skip task dependencies

  Environment:
    --pkg-mgr MGR          Package manager: pip, uv, conda (default: uv)
    --env-name NAME        Environment name: conda env name (for conda only)
    --install-dir DIR      Root installation directory (default: /opt/flagscale)
                           All paths derived from this:
                             - \$FLAGSCALE_HOME/miniconda3  (conda)
                             - \$FLAGSCALE_HOME/venv        (uv venv)
                             - \$FLAGSCALE_HOME/deps        (source dependencies)
                             - \$FLAGSCALE_HOME/downloads   (cached downloads)
    --index-url URL        PyPI index URL (for custom mirrors)
    --extra-index-url URL  Extra PyPI index URL

  Other:
    --retry-count N        Retry attempts (default: 3)
    --force-build          Force rebuild source deps even if already installed
    --debug                Debug mode: print commands without executing (dry-run)
    --help                 Show this help

PACKAGE MANAGERS:
    pip    - Use pip directly (standard Python)
    uv     - Use uv pip (fast, modern) [default]
    conda  - Use conda environment with pip for PyPI packages

INSTALLATION PHASES:
    system       - System packages (apt, python, openmpi)
    base         - Base dependencies (torch, cuda libs) from install_base.sh
    task         - Task dependencies (pip requirements + source repos like Megatron, etc.)

EXAMPLES:
    $0 --platform cuda --task train                                    # Full installation
    $0 --platform cuda --task train --no-system                        # Skip system packages
    $0 --platform cuda --task train --no-system --no-base              # Task deps only
    $0 --platform cuda --task train --pkg-mgr conda --env-name train   # Use conda
    $0 --platform cuda --task train --install-dir /home/user/opt       # Custom root dir
    $0 --platform cuda --task train --debug                            # Preview (dry-run)
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --task)            TASK="$2"; shift 2 ;;
            --platform)        PLATFORM="$2"; shift 2 ;;
            # Phase toggles: --no-<phase> to skip
            --no-system)       INSTALL_SYSTEM=false; shift ;;
            --no-dev)          INSTALL_DEV=false; shift ;;
            --no-base)         INSTALL_BASE=false; shift ;;
            --no-task)         INSTALL_TASK=false; shift ;;
            # Environment options
            --pkg-mgr)         PKG_MGR="$2"; shift 2 ;;
            --env-name)        ENV_NAME="$2"; shift 2 ;;
            --install-dir)     FLAGSCALE_HOME="$2"; shift 2 ;;
            --index-url)       INDEX_URL="$2"; shift 2 ;;
            --extra-index-url) EXTRA_INDEX_URL="$2"; shift 2 ;;
            --retry-count)     RETRY_COUNT="$2"; shift 2 ;;
            --force-build)     FORCE_BUILD=true; shift ;;
            --debug)           DEBUG=true; shift ;;
            --help|-h)         usage; exit 0 ;;
            *)                 log_error "Unknown option: $1"; usage; exit 1 ;;
        esac
    done
}

validate_inputs() {
    # Platform is required
    if [ -z "$PLATFORM" ]; then
        log_error "Platform required (use --platform, e.g., --platform cuda)"
        usage
        exit 1
    fi

    # Validate platform directory exists
    if [ ! -d "$SCRIPT_DIR/$PLATFORM" ]; then
        log_error "Invalid platform: $PLATFORM (directory not found: $SCRIPT_DIR/$PLATFORM)"
        exit 1
    fi

    # Task is required
    if [ -z "$TASK" ]; then
        log_error "Task required (use --task)"
        usage
        exit 1
    fi

    # Validate task
    local valid_tasks=($(get_valid_tasks))
    local valid=false
    for t in "${valid_tasks[@]}"; do
        [ "$TASK" = "$t" ] && valid=true && break
    done
    [ "$valid" = false ] && { log_error "Invalid task: $TASK. Valid: ${valid_tasks[*]}"; exit 1; }
}

setup_package_manager() {
    # Set package manager based on --package-manager option
    case "$PKG_MGR" in
        pip|uv|conda)
            set_pkg_manager "$PKG_MGR"
            ;;
        *)
            log_error "Invalid package manager: $PKG_MGR (use pip, uv, or conda)"
            exit 1
            ;;
    esac
}

setup_environment() {
    local manager=$(get_pkg_manager)

    # Use exported paths from setup_exports()
    case "$manager" in
        conda)
            # Check if conda exists at the expected path
            if [ ! -f "$FLAGSCALE_CONDA/bin/conda" ]; then
                log_error "Conda not found at $FLAGSCALE_CONDA"
                log_error "Run with system phase (remove --no-system) or install conda manually"
                return 1
            fi
            if [ -n "$ENV_NAME" ]; then
                activate_conda -d $DEBUG "$ENV_NAME" "$FLAGSCALE_CONDA" "$PYTHON_VERSION" || return 1
            fi
            ;;
        uv)
            # Check if venv exists
            if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
                log_error "UV venv not found at $UV_PROJECT_ENVIRONMENT"
                log_error "Run with system phase (remove --no-system) or create venv manually"
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
    # Export FLAGSCALE_* paths for child scripts (prefixed to avoid conflicts)
    export FLAGSCALE_HOME
    export FLAGSCALE_CONDA="$FLAGSCALE_HOME/miniconda3"
    export FLAGSCALE_DEPS="$FLAGSCALE_HOME/deps"
    export FLAGSCALE_DOWNLOADS="$FLAGSCALE_HOME/downloads"

    # Export build options
    export FLAGSCALE_FORCE_BUILD="$FORCE_BUILD"

    # Standard tool paths (not prefixed - expected by tools)
    export UV_PROJECT_ENVIRONMENT="$FLAGSCALE_HOME/venv"

    # Export PyPI index URLs if set
    [ -n "$INDEX_URL" ] && { export PIP_INDEX_URL="$INDEX_URL" UV_INDEX_URL="$INDEX_URL"; }
    [ -n "$EXTRA_INDEX_URL" ] && { export PIP_EXTRA_INDEX_URL="$EXTRA_INDEX_URL" UV_EXTRA_INDEX_URL="$EXTRA_INDEX_URL"; }
}

main() {
    parse_args "$@"

    [ "$DEBUG" = true ] && log_info "Dry-run mode: commands printed, not executed"

    validate_inputs

    print_header "FlagScale Installation"
    local phases=""
    [ "$INSTALL_SYSTEM" = true ] && phases="${phases}system,"
    [ "$INSTALL_DEV" = true ] && phases="${phases}dev,"
    [ "$INSTALL_BASE" = true ] && phases="${phases}base,"
    [ "$INSTALL_TASK" = true ] && phases="${phases}task,"
    phases="${phases%,}"  # Remove trailing comma
    [ -z "$phases" ] && phases="none"
    log_info "Task: ${TASK:-none} | Platform: $PLATFORM | Pkg: $PKG_MGR | Phases: $phases"
    log_info "Install dir: $FLAGSCALE_HOME"

    setup_exports

    # ==========================================================================
    # Phase 1: System dependencies (shared by ALL tasks)
    # Installs: apt packages, python (via conda/uv/pip), openmpi
    # ==========================================================================
    if [ "$INSTALL_SYSTEM" = true ]; then
        install_system_deps
    fi

    # Check if any pip phases are enabled (require environment setup)
    local has_pip_phases=$( [ "$INSTALL_DEV" = true ] || [ "$INSTALL_BASE" = true ] || [ "$INSTALL_TASK" = true ] && echo true || echo false )
    if [ "$has_pip_phases" = false ]; then
        log_success "Installation complete"
        print_header "Installation Complete"
        return 0
    fi

    # ==========================================================================
    # Phase 2: Environment setup (required for task phases)
    # Activates: conda env or uv venv
    # ==========================================================================
    setup_package_manager
    setup_environment || die "Environment setup failed"
    log_info "Env: $(get_current_env) | $(python --version 2>/dev/null || echo 'Python: N/A')"

    # ==========================================================================
    # Phase 3: Dev dependencies (default, use --no-dev to skip)
    # Installs: build, lint, test tools from requirements/dev.txt
    # ==========================================================================
    if [ "$INSTALL_DEV" = true ]; then
        print_header "Dev Dependencies"
        install_dev_deps
    fi

    # ==========================================================================
    # Phase 4: Task-specific dependencies (runs for each task)
    # Installs: base deps, pip requirements, source deps (git repos)
    # ==========================================================================
    if [ "$TASK" = "all" ]; then
        for task in $(get_valid_tasks); do
            [ "$task" = "all" ] && continue
            install_task "$task"
        done
    else
        install_task "$TASK"
    fi

    print_header "Installation Complete"
}

main "$@"

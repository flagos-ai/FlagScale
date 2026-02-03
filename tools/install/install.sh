#!/bin/bash
# =============================================================================
# FlagScale Dependency Installation
# =============================================================================
#
# Main entry point for installing FlagScale dependencies.
# Orchestrates four installation phases: system, dev, base, task.
#
# Usage:
#   ./install.sh --platform PLATFORM --task TASK [OPTIONS]
#
# Examples:
#   ./install.sh --platform cuda --task train                    # Full installation
#   ./install.sh --platform cuda --task train --no-system        # Skip system phase
#   ./install.sh --platform cuda --task train --src-deps megatron-lm  # Only megatron-lm
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/utils.sh"
source "$SCRIPT_DIR/utils/pkg_utils.sh"

PROJECT_ROOT=$(get_project_root)

# =============================================================================
# Configuration (defaults)
# =============================================================================
TASK=""
PLATFORM=""
PKG_MGR="uv"
ENV_NAME=""
DEBUG=false
RETRY_COUNT=3
FORCE_BUILD=false
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"

# Phase flags (default: install all)
INSTALL_SYSTEM=true
INSTALL_DEV=true
INSTALL_BASE=true
INSTALL_TASK=true

# Override flags (selective installation)
SRC_DEPS=""
PIP_DEPS=""

# PyPI index URLs
INDEX_URL="${PIP_INDEX_URL:-}"
EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"

# =============================================================================
# Helper Functions
# =============================================================================
get_valid_tasks() {
    local tasks=()
    if [ -n "$PLATFORM" ] && [ -d "$SCRIPT_DIR/$PLATFORM" ]; then
        for script in "$SCRIPT_DIR/$PLATFORM"/install_*.sh; do
            [ -f "$script" ] || continue
            local task=$(basename "$script" | sed 's/^install_//' | sed 's/\.sh$//')
            [ "$task" != "base" ] && tasks+=("$task")
        done
    fi
    tasks+=("all")
    echo "${tasks[@]}"
}

# Export all configuration as environment variables for phase scripts
export_config() {
    # Paths
    export FLAGSCALE_HOME
    export FLAGSCALE_CONDA="$FLAGSCALE_HOME/miniconda3"
    export FLAGSCALE_DEPS="$FLAGSCALE_HOME/deps"
    export FLAGSCALE_DOWNLOADS="$FLAGSCALE_HOME/downloads"
    export UV_PROJECT_ENVIRONMENT="$FLAGSCALE_HOME/venv"

    # Phase flags (for should_install functions)
    export FLAGSCALE_INSTALL_SYSTEM="$INSTALL_SYSTEM"
    export FLAGSCALE_INSTALL_DEV="$INSTALL_DEV"
    export FLAGSCALE_INSTALL_BASE="$INSTALL_BASE"
    export FLAGSCALE_INSTALL_TASK="$INSTALL_TASK"

    # Override flags
    export FLAGSCALE_SRC_DEPS="$SRC_DEPS"
    export FLAGSCALE_PIP_DEPS="$PIP_DEPS"
    export FLAGSCALE_FORCE_BUILD="$FORCE_BUILD"

    # Other config
    export FLAGSCALE_PLATFORM="$PLATFORM"
    export FLAGSCALE_TASK="$TASK"
    export FLAGSCALE_PKG_MGR="$PKG_MGR"
    export FLAGSCALE_ENV_NAME="$ENV_NAME"
    export FLAGSCALE_DEBUG="$DEBUG"
    export FLAGSCALE_RETRY_COUNT="$RETRY_COUNT"

    # PyPI index
    [ -n "$INDEX_URL" ] && { export PIP_INDEX_URL="$INDEX_URL" UV_INDEX_URL="$INDEX_URL"; }
    [ -n "$EXTRA_INDEX_URL" ] && { export PIP_EXTRA_INDEX_URL="$EXTRA_INDEX_URL" UV_EXTRA_INDEX_URL="$EXTRA_INDEX_URL"; }
}

# =============================================================================
# Phase Execution
# =============================================================================
run_phase() {
    local phase="$1"
    local script="$2"
    local args="${3:-}"

    # Check if phase should run (enabled OR has overrides)
    local phase_enabled
    case "$phase" in
        system) phase_enabled="$INSTALL_SYSTEM" ;;
        dev)    phase_enabled="$INSTALL_DEV" ;;
        base)   phase_enabled="$INSTALL_BASE" ;;
        task)   phase_enabled="$INSTALL_TASK" ;;
    esac

    # Skip if phase disabled and no relevant overrides
    if [ "$phase_enabled" = false ]; then
        case "$phase" in
            dev|base|task)
                # These phases can have pip/src overrides
                [ -z "$PIP_DEPS" ] && [ -z "$SRC_DEPS" ] && { log_info "Skipping $phase phase"; return 0; }
                ;;
            system)
                # System phase has no overrides currently
                log_info "Skipping $phase phase"
                return 0
                ;;
        esac
    fi

    # Run the phase script
    [ ! -f "$script" ] && { log_warn "Phase script not found: $script"; return 0; }

    print_header "${phase^} Phase"
    [ "$DEBUG" = true ] && args="$args --debug"
    "$script" $args || die "${phase^} phase failed"
}

# =============================================================================
# Main
# =============================================================================
usage() {
    cat << EOF
Usage: $0 --platform PLATFORM --task TASK [OPTIONS]

OPTIONS:
    --platform NAME        Platform (required, e.g., cuda)
    --task TASK            Task (required, e.g., train, serve, inference, rl, all)

  Phase Control (default: install all):
    --no-system            Skip system phase (apt, python, openmpi)
    --no-dev               Skip dev phase (dev requirements)
    --no-base              Skip base phase (base requirements + source)
    --no-task              Skip task phase (task requirements + source)

  Selective Installation (overrides --no-* for specific packages):
    --pip-deps PKGS        Install specific pip packages (comma-separated)
    --src-deps DEPS        Install specific source deps (comma-separated)
                           dev: sccache
                           train: apex,flash-attn,transformer-engine,megatron-lm
                           serve: vllm

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
    $0 --platform cuda --task train --no-system                    # Skip system phase
    $0 --platform cuda --task train --no-system --no-dev --no-base --no-task --src-deps megatron-lm
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
    [ -z "$PLATFORM" ] && { log_error "Platform required (use --platform)"; usage; exit 1; }
    [ ! -d "$SCRIPT_DIR/$PLATFORM" ] && { log_error "Invalid platform: $PLATFORM"; exit 1; }
    [ -z "$TASK" ] && { log_error "Task required (use --task)"; usage; exit 1; }

    local valid_tasks=($(get_valid_tasks))
    local valid=false
    for t in "${valid_tasks[@]}"; do
        [ "$TASK" = "$t" ] && valid=true && break
    done
    [ "$valid" = false ] && { log_error "Invalid task: $TASK. Valid: ${valid_tasks[*]}"; exit 1; }
}

main() {
    parse_args "$@"
    validate_inputs

    [ "$DEBUG" = true ] && log_info "Dry-run mode"

    print_header "FlagScale Installation"
    log_info "Platform: $PLATFORM | Task: $TASK | Pkg: $PKG_MGR"
    [ -n "$SRC_DEPS" ] && log_info "Source deps override: $SRC_DEPS"
    [ -n "$PIP_DEPS" ] && log_info "Pip deps override: $PIP_DEPS"
    log_info "Install dir: $FLAGSCALE_HOME"

    export_config

    # Phase 1: System (apt, python, openmpi)
    run_phase system "$SCRIPT_DIR/install_system.sh" "--platform $PLATFORM --pkg-mgr $PKG_MGR"

    # Phase 2: Dev (dev requirements)
    run_phase dev "$SCRIPT_DIR/install_dev.sh"

    # Phase 3: Base (base requirements + source for platform)
    run_phase base "$SCRIPT_DIR/$PLATFORM/install_base.sh"

    # Phase 4: Task (task requirements + source)
    if [ "$TASK" = "all" ]; then
        for task in $(get_valid_tasks); do
            [ "$task" = "all" ] && continue
            FLAGSCALE_TASK="$task"
            export FLAGSCALE_TASK
            run_phase task "$SCRIPT_DIR/$PLATFORM/install_${task}.sh"
        done
    else
        run_phase task "$SCRIPT_DIR/$PLATFORM/install_${TASK}.sh"
    fi

    print_header "Installation Complete"
}

main "$@"

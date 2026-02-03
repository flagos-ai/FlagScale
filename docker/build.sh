#!/bin/bash
# FlagScale Docker Build Script
#
# NOTE: This script is experimental and requires further testing.
#       Please report issues at https://github.com/FlagOpen/FlagScale/issues
#
# Usage: ./docker/build.sh [OPTIONS]
#
# Examples:
#   ./docker/build.sh --platform cuda
#   ./docker/build.sh --platform cuda --task train
#   ./docker/build.sh --platform cuda --task train --target dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# =============================================================================
# Logging functions
# =============================================================================
log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

# =============================================================================
# Default versions (same as tools/install, override via environment variables)
# =============================================================================
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_VERSION="${UV_VERSION:-0.7.2}"
OPENMPI_VERSION="${OPENMPI_VERSION:-4.1.6}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"

# =============================================================================
# Default values
# =============================================================================
PLATFORM="cuda"
TASK=""
TARGET="dev"
TAG_PREFIX="flagscale"
NO_CACHE=false

# PyPI index URLs (optional, for custom mirrors)
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"

# =============================================================================
# Platform and task discovery
# =============================================================================

# Get available tasks by scanning Dockerfile.* files
get_platform_tasks() {
    local platform=$1
    local platform_dir="$SCRIPT_DIR/$platform"
    if [ -d "$platform_dir" ]; then
        ls "$platform_dir"/Dockerfile.* 2>/dev/null | xargs -n1 basename | sed 's/Dockerfile\.//' || true
    fi
}

# Get first task as default
get_default_task() {
    local platform=$1
    get_platform_tasks "$platform" | head -1
}

# Validate platform exists
validate_platform() {
    local platform=$1
    if [ ! -d "$SCRIPT_DIR/$platform" ]; then
        log_error "Platform directory not found: $SCRIPT_DIR/$platform"
        exit 1
    fi
}

# Validate task exists for platform
validate_task() {
    local platform=$1
    local task=$2
    local dockerfile="$SCRIPT_DIR/$platform/Dockerfile.${task}"
    if [ ! -f "$dockerfile" ]; then
        log_error "Task '$task' not found for platform '$platform'"
        log_error "Available tasks: $(get_platform_tasks "$platform" | tr '\n' ' ')"
        exit 1
    fi
}

# =============================================================================
# Usage
# =============================================================================
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build FlagScale Docker images.

OPTIONS:
    --platform PLATFORM  Platform to build (default: cuda)
    --task TASK          Task to build (default: first task in platform)
    --target TARGET      Build target: dev, release (default: dev)
    --tag-prefix PREFIX  Image tag prefix (default: flagscale)
    --index-url URL      PyPI index URL (for custom mirrors)
    --extra-index-url URL  Extra PyPI index URL
    --no-cache           Build without cache
    --help               Show this help message

VERSIONS (override via environment variables):
    PYTHON_VERSION       Python version (default: ${PYTHON_VERSION})
    UV_VERSION           uv version (default: ${UV_VERSION})
    OPENMPI_VERSION      OpenMPI version (default: ${OPENMPI_VERSION})
    CUDA_VERSION         CUDA version (default: ${CUDA_VERSION})
    UBUNTU_VERSION       Ubuntu version (default: ${UBUNTU_VERSION})

EXAMPLES:
    $0 --platform cuda
    $0 --platform cuda --task train
    $0 --platform cuda --task train --target dev
    CUDA_VERSION=12.4.0 $0 --platform cuda --task train

EOF
}

# =============================================================================
# Argument parsing
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --platform)         PLATFORM="$2"; shift 2 ;;
            --task)             TASK="$2"; shift 2 ;;
            --target)           TARGET="$2"; shift 2 ;;
            --tag-prefix)       TAG_PREFIX="$2"; shift 2 ;;
            --index-url)        PIP_INDEX_URL="$2"; shift 2 ;;
            --extra-index-url)  PIP_EXTRA_INDEX_URL="$2"; shift 2 ;;
            --no-cache)         NO_CACHE=true; shift ;;
            --help|-h)          usage; exit 0 ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# Image tag generation
# =============================================================================
get_image_tag() {
    local platform=$1
    local task=$2
    local tag="${TAG_PREFIX}-${task}:${TARGET}"

    # Add CUDA version suffix for cuda platform
    if [ "$platform" = "cuda" ]; then
        local cuda_major=$(echo "$CUDA_VERSION" | cut -d. -f1)
        local cuda_minor=$(echo "$CUDA_VERSION" | cut -d. -f2)
        tag="${tag}-cu${cuda_major}${cuda_minor}"
    fi

    # Add python version
    tag="${tag}-py${PYTHON_VERSION}"

    echo "$tag"
}

# =============================================================================
# Build image
# =============================================================================
build_image() {
    local platform=$PLATFORM
    local task=$TASK
    local dockerfile="$SCRIPT_DIR/$platform/Dockerfile.${task}"

    local image_tag=$(get_image_tag "$platform" "$task")

    log_info "Building image: $image_tag"
    log_info "Dockerfile: $dockerfile"
    log_info "Platform: $platform"
    log_info "Task: $task"
    log_info "Target: $TARGET"

    # Build command
    local build_cmd="docker build -f $dockerfile --target $TARGET -t $image_tag"

    # Add version build args
    log_info "PYTHON_VERSION: $PYTHON_VERSION"
    log_info "UV_VERSION: $UV_VERSION"
    log_info "OPENMPI_VERSION: $OPENMPI_VERSION"
    build_cmd="$build_cmd --build-arg PYTHON_VERSION=$PYTHON_VERSION"
    build_cmd="$build_cmd --build-arg UV_VERSION=$UV_VERSION"
    build_cmd="$build_cmd --build-arg OPENMPI_VERSION=$OPENMPI_VERSION"

    # Compute and add derived values for CUDA platform
    if [ "$platform" = "cuda" ]; then
        local cuda_major=$(echo "$CUDA_VERSION" | cut -d. -f1)
        local cuda_minor=$(echo "$CUDA_VERSION" | cut -d. -f2)

        local base_image="nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}"
        local pytorch_index="https://download.pytorch.org/whl/cu${cuda_major}${cuda_minor}"

        log_info "CUDA_VERSION: $CUDA_VERSION"
        log_info "UBUNTU_VERSION: $UBUNTU_VERSION"
        log_info "BASE_IMAGE: $base_image"
        log_info "PYTORCH_INDEX: $pytorch_index"
        build_cmd="$build_cmd --build-arg BASE_IMAGE=$base_image"
        build_cmd="$build_cmd --build-arg PYTORCH_INDEX=$pytorch_index"
    fi

    # Add PyPI index URLs if specified
    if [ -n "$PIP_INDEX_URL" ]; then
        log_info "PIP_INDEX_URL: $PIP_INDEX_URL"
        build_cmd="$build_cmd --build-arg PIP_INDEX_URL=$PIP_INDEX_URL"
    fi
    if [ -n "$PIP_EXTRA_INDEX_URL" ]; then
        log_info "PIP_EXTRA_INDEX_URL: $PIP_EXTRA_INDEX_URL"
        build_cmd="$build_cmd --build-arg PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL"
    fi

    [ "$NO_CACHE" = true ] && build_cmd="$build_cmd --no-cache"
    build_cmd="$build_cmd $PROJECT_ROOT"

    log_info "Running: $build_cmd"
    eval "$build_cmd"

    log_info "Successfully built: $image_tag"
}

# =============================================================================
# Main
# =============================================================================
main() {
    parse_args "$@"

    # Validate platform
    validate_platform "$PLATFORM"

    # Set default task if not specified
    if [ -z "$TASK" ]; then
        TASK=$(get_default_task "$PLATFORM")
        log_info "No task specified, using default: $TASK"
    fi

    # Validate task
    validate_task "$PLATFORM" "$TASK"

    log_info "FlagScale Docker Build"
    log_info "======================"

    build_image
}

main "$@"

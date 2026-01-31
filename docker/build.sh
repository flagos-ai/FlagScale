#!/bin/bash
# FlagScale Docker Build Script
#
# Reads version configuration from versions.json. To change versions
# (CUDA, Python, Ubuntu, etc.), edit versions.json directly.
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
VERSIONS_FILE="$PROJECT_ROOT/versions.json"

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
# Prerequisites
# =============================================================================
check_jq() {
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed. Please install jq."
        exit 1
    fi
}

check_versions_file() {
    if [ ! -f "$VERSIONS_FILE" ]; then
        log_error "versions.json not found: $VERSIONS_FILE"
        exit 1
    fi
}

# =============================================================================
# Version getters (handles nested structure)
# =============================================================================

# Get version from common section
get_common() {
    local name=$1
    jq -r ".common.\"${name}\".version // empty" "$VERSIONS_FILE"
}

# Get version from platform section (handles nested base and platform-level props)
# Usage: get_platform "cuda" "torch"  -> looks in cuda.base.torch
#        get_platform "cuda" "cuda"   -> looks in cuda.cuda (platform-level)
get_platform() {
    local platform=$1
    local name=$2
    # First try platform.base.<name> (for packages like torch)
    local version=$(jq -r ".\"${platform}\".base.\"${name}\".version // empty" "$VERSIONS_FILE")
    # If not found, try platform.<name> (for platform-level props like cuda version)
    if [ -z "$version" ]; then
        version=$(jq -r ".\"${platform}\".\"${name}\".version // empty" "$VERSIONS_FILE")
    fi
    echo "$version"
}

# Check if entry is a pip package (pip: true means pip package, false means tool)
is_pip_package() {
    local section=$1
    local name=$2
    # First check common/dev sections
    local result=$(jq -r ".\"${section}\".\"${name}\".pip // empty" "$VERSIONS_FILE")
    if [ -z "$result" ]; then
        # For platform sections, check base subsection first, then platform level
        result=$(jq -r ".\"${section}\".base.\"${name}\".pip // empty" "$VERSIONS_FILE")
        if [ -z "$result" ]; then
            result=$(jq -r ".\"${section}\".\"${name}\".pip // empty" "$VERSIONS_FILE")
        fi
    fi
    [ "$result" = "true" ]
}

# =============================================================================
# Platform and task discovery
# =============================================================================

# Get available platforms from versions.json (excluding "common" and "dev")
get_platforms() {
    jq -r 'keys[] | select(. != "common" and . != "dev")' "$VERSIONS_FILE"
}

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
    if ! jq -e ".\"${platform}\"" "$VERSIONS_FILE" > /dev/null 2>&1; then
        log_error "Platform '$platform' not found in versions.json"
        log_error "Available platforms: $(get_platforms | tr '\n' ' ')"
        exit 1
    fi
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
    check_jq
    check_versions_file

    cat << EOF
Usage: $0 [OPTIONS]

Build FlagScale Docker images. Configuration is read from versions.json.
To change versions, edit versions.json directly.

OPTIONS:
    --platform PLATFORM  Platform to build (default: cuda)
    --task TASK          Task to build (default: first task in platform)
    --target TARGET      Build target: dev, release (default: dev)
    --tag-prefix PREFIX  Image tag prefix (default: flagscale)
    --index-url URL      PyPI index URL (for custom mirrors)
    --extra-index-url URL  Extra PyPI index URL
    --no-cache           Build without cache
    --help               Show this help message

VERSIONS (from versions.json):

  [common]
EOF
    for key in $(jq -r '.common | keys[]' "$VERSIONS_FILE"); do
        local value=$(get_common "$key")
        local is_pip=$(jq -r ".common.\"${key}\".pip" "$VERSIONS_FILE")
        local type_label="tool"
        [ "$is_pip" = "true" ] && type_label="pip"
        printf "    %-16s = %-10s (%s)\n" "$key" "$value" "$type_label"
    done

    for platform in $(get_platforms); do
        echo ""
        echo "  [$platform]"
        echo "    Tasks: $(get_platform_tasks "$platform" | tr '\n' ' ')"

        # Show platform-level properties (cuda, ubuntu)
        for key in $(jq -r ".\"${platform}\" | to_entries[] | select(.value | type == \"object\" and has(\"version\")) | .key" "$VERSIONS_FILE"); do
            local value=$(get_platform "$platform" "$key")
            local is_pip=$(jq -r ".\"${platform}\".\"${key}\".pip" "$VERSIONS_FILE")
            local type_label="tool"
            [ "$is_pip" = "true" ] && type_label="pip"
            printf "    %-16s = %-10s (%s)\n" "$key" "$value" "$type_label"
        done

        # Show base packages
        if jq -e ".\"${platform}\".base" "$VERSIONS_FILE" > /dev/null 2>&1; then
            echo "    [base]"
            for key in $(jq -r ".\"${platform}\".base | keys[]" "$VERSIONS_FILE"); do
                local value=$(jq -r ".\"${platform}\".base.\"${key}\".version // empty" "$VERSIONS_FILE")
                local is_pip=$(jq -r ".\"${platform}\".base.\"${key}\".pip" "$VERSIONS_FILE")
                local type_label="tool"
                [ "$is_pip" = "true" ] && type_label="pip"
                printf "      %-14s = %-10s (%s)\n" "$key" "$value" "$type_label"
            done
        fi
    done

    cat << EOF

EXAMPLES:
    $0 --platform cuda
    $0 --platform cuda --task train
    $0 --platform cuda --task train --target dev
    $0 --platform cuda --task all --target release

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
        local cuda_version=$(get_platform "$platform" "cuda")
        local cuda_major=$(echo "$cuda_version" | cut -d. -f1)
        local cuda_minor=$(echo "$cuda_version" | cut -d. -f2)
        tag="${tag}-cu${cuda_major}${cuda_minor}"
    fi

    # Add python version
    local python_version=$(get_common "python")
    tag="${tag}-py${python_version}"

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

    # Add common tool versions as build args (only non-Python packages)
    for key in $(jq -r '.common | keys[]' "$VERSIONS_FILE"); do
        if ! is_pip_package "common" "$key"; then
            local value=$(get_common "$key")
            local arg_name=$(echo "$key" | tr '[:lower:]' '[:upper:]' | tr '-' '_')_VERSION
            log_info "${arg_name}: $value"
            build_cmd="$build_cmd --build-arg ${arg_name}=$value"
        fi
    done

    # Add platform-specific tool versions as build args (platform-level properties only)
    for key in $(jq -r ".\"${platform}\" | to_entries[] | select(.value | type == \"object\" and has(\"version\")) | .key" "$VERSIONS_FILE"); do
        if ! is_pip_package "$platform" "$key"; then
            local value=$(get_platform "$platform" "$key")
            local arg_name=$(echo "$key" | tr '[:lower:]' '[:upper:]' | tr '-' '_')_VERSION
            log_info "${arg_name}: $value"
            build_cmd="$build_cmd --build-arg ${arg_name}=$value"
        fi
    done

    # Compute and add derived values for CUDA platform
    if [ "$platform" = "cuda" ]; then
        local cuda_version=$(get_platform "$platform" "cuda")
        local ubuntu_version=$(get_platform "$platform" "ubuntu")
        local cuda_major=$(echo "$cuda_version" | cut -d. -f1)
        local cuda_minor=$(echo "$cuda_version" | cut -d. -f2)

        local base_image="nvidia/cuda:${cuda_version}-devel-ubuntu${ubuntu_version}"
        local pytorch_index="https://download.pytorch.org/whl/cu${cuda_major}${cuda_minor}"

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
    check_jq
    check_versions_file

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

#!/bin/bash
# =============================================================================
# Version Loading Utility
# =============================================================================
#
# Loads version information from versions.json (single source of truth)
# located at the project root.
#
# Structure of versions.json:
#   common              - Versions shared by all platforms/tasks
#   dev                 - Development tool versions
#   <platform>          - Platform-specific versions (e.g., cuda)
#     base              - Base packages for the platform (torch, cuda, etc.)
#     <task>            - Task-specific packages (train, inference, rl, etc.)
#
# Each entry has: {"version": "x.y.z", "pip": true/false}
#   pip: true  - Python package (pip install)
#   pip: false - System tool (build from source)
#
# Usage:
#   source utils/versions.sh
#   python_ver=$(get_common "python")
#   torch_ver=$(get_platform "cuda" "torch")        # looks in cuda.base
#   flash_ver=$(get_task "cuda" "train" "flash-attn")  # looks in cuda.train
# =============================================================================

# Get the project root directory
_get_versions_project_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir/../../.."
    pwd
}

# Path to versions.json
VERSIONS_FILE="$(_get_versions_project_root)/versions.json"

# Check if jq is available (required for JSON parsing)
_check_jq() {
    if ! command -v jq &> /dev/null; then
        return 1
    fi
    return 0
}

# =============================================================================
# Version getters
# =============================================================================

# Get version from common section (with fallback defaults)
# Usage: python_ver=$(get_common "python")
get_common() {
    local name=$1
    local version=""

    if _check_jq && [ -f "$VERSIONS_FILE" ]; then
        version=$(jq -r ".common.\"${name}\".version // empty" "$VERSIONS_FILE")
    fi

    # Fallback defaults for critical tools when jq unavailable
    if [ -z "$version" ]; then
        case "$name" in
            python)   version="3.12" ;;
            uv)       version="0.7.2" ;;
            openmpi)  version="4.1.6" ;;
        esac
    fi

    echo "$version"
}

# Get version from platform section
# Usage: torch_ver=$(get_platform "cuda" "torch")
#        cuda_ver=$(get_platform "cuda" "cuda")
# Looks in <platform>.base.<name> first, then <platform>.<name> for platform-level props
get_platform() {
    local platform=$1
    local name=$2

    if _check_jq && [ -f "$VERSIONS_FILE" ]; then
        # First try platform.base.<name> (for packages like torch)
        local version=$(jq -r ".\"${platform}\".base.\"${name}\".version // empty" "$VERSIONS_FILE")
        # If not found, try platform.<name> (for platform-level props like cuda version)
        if [ -z "$version" ]; then
            version=$(jq -r ".\"${platform}\".\"${name}\".version // empty" "$VERSIONS_FILE")
        fi
        echo "$version"
    fi
}

# Get version from task section
# Usage: flash_ver=$(get_task "cuda" "train" "flash-attn")
# Looks in <platform>.<task>.<name>
get_task() {
    local platform=$1
    local task=$2
    local name=$3

    if _check_jq && [ -f "$VERSIONS_FILE" ]; then
        jq -r ".\"${platform}\".\"${task}\".\"${name}\".version // empty" "$VERSIONS_FILE"
    fi
}

# Check if entry is a pip package
# Usage: if is_pip "common" "hydra-core"; then ...
#        if is_pip "cuda" "base" "torch"; then ...
#        if is_pip "cuda" "train" "flash-attn"; then ...
is_pip() {
    local section=$1
    local subsection=$2
    local name=${3:-}

    if _check_jq && [ -f "$VERSIONS_FILE" ]; then
        local result
        if [ -z "$name" ]; then
            # Two-arg form: is_pip "common" "hydra-core"
            name="$subsection"
            result=$(jq -r ".\"${section}\".\"${name}\".pip // false" "$VERSIONS_FILE")
        else
            # Three-arg form: is_pip "cuda" "base" "torch"
            result=$(jq -r ".\"${section}\".\"${subsection}\".\"${name}\".pip // false" "$VERSIONS_FILE")
        fi
        [ "$result" = "true" ]
    else
        return 1
    fi
}

# =============================================================================
# Section helpers
# =============================================================================

# Get all keys from a section (flat sections like common, dev)
# Usage: for key in $(get_section_keys "common"); do ...
get_section_keys() {
    local section=$1

    if _check_jq && [ -f "$VERSIONS_FILE" ]; then
        jq -r ".\"${section}\" | keys[]" "$VERSIONS_FILE"
    fi
}

# Get all keys from a platform subsection (base or task)
# Usage: for key in $(get_platform_keys "cuda" "base"); do ...
#        for key in $(get_platform_keys "cuda" "train"); do ...
get_platform_keys() {
    local platform=$1
    local subsection=$2

    if _check_jq && [ -f "$VERSIONS_FILE" ]; then
        jq -r ".\"${platform}\".\"${subsection}\" | keys[] // empty" "$VERSIONS_FILE"
    fi
}

# Get all task names for a platform (only task subsections, not platform properties)
# Tasks like train/inference/rl have nested package objects: {pkg: {version, pip}}
# Platform properties like cuda/ubuntu have direct structure: {version, pip}
# Usage: for task in $(get_platform_tasks "cuda"); do ...
get_platform_tasks() {
    local platform=$1

    if _check_jq && [ -f "$VERSIONS_FILE" ]; then
        # Only return keys that are task objects (have nested packages, not direct version/pip)
        # A task object doesn't have a direct "version" key - platform properties do
        jq -r ".\"${platform}\" | to_entries[] | select(.key != \"base\" and (.value | type) == \"object\" and (.value | has(\"version\") | not)) | .key" "$VERSIONS_FILE"
    fi
}

# Get all platforms (excluding "common" and "dev")
# Usage: for platform in $(get_all_platforms); do ...
get_all_platforms() {
    if _check_jq && [ -f "$VERSIONS_FILE" ]; then
        jq -r 'keys[] | select(. != "common" and . != "dev")' "$VERSIONS_FILE"
    fi
}

# =============================================================================
# Display helpers
# =============================================================================

# Print all versions (for debugging)
print_versions() {
    if ! _check_jq || [ ! -f "$VERSIONS_FILE" ]; then
        echo "[WARN] Cannot read versions.json" >&2
        return 1
    fi

    echo "Versions from: $VERSIONS_FILE"
    echo ""
    echo "Common:"
    for key in $(get_section_keys "common"); do
        local version=$(get_common "$key")
        local pip_flag=""
        is_pip "common" "$key" && pip_flag=" (pip)"
        printf "  %-14s %s%s\n" "$key:" "$version" "$pip_flag"
    done

    echo ""
    echo "Dev:"
    for key in $(get_section_keys "dev"); do
        local version=$(jq -r ".dev.\"${key}\".version // empty" "$VERSIONS_FILE")
        local pip_flag=""
        is_pip "dev" "$key" && pip_flag=" (pip)"
        printf "  %-14s %s%s\n" "$key:" "$version" "$pip_flag"
    done

    for platform in $(get_all_platforms); do
        echo ""
        echo "${platform^} (base):"
        for key in $(get_platform_keys "$platform" "base"); do
            local version=$(get_platform "$platform" "$key")
            local pip_flag=""
            is_pip "$platform" "base" "$key" && pip_flag=" (pip)"
            printf "  %-14s %s%s\n" "$key:" "$version" "$pip_flag"
        done

        for task in $(get_platform_tasks "$platform"); do
            local task_keys=$(get_platform_keys "$platform" "$task")
            if [ -n "$task_keys" ]; then
                echo ""
                echo "${platform^} ($task):"
                for key in $task_keys; do
                    local version=$(get_task "$platform" "$task" "$key")
                    local pip_flag=""
                    is_pip "$platform" "$task" "$key" && pip_flag=" (pip)"
                    printf "  %-14s %s%s\n" "$key:" "$version" "$pip_flag"
                done
            fi
        done
    done
}

# Print pip package versions for requirements files
print_package_versions() {
    if ! _check_jq || [ ! -f "$VERSIONS_FILE" ]; then
        echo "[ERROR] Cannot read versions without jq and versions.json" >&2
        return 1
    fi

    echo "# Package versions from versions.json"
    echo ""
    echo "# Common packages (requirements/common.txt):"
    jq -r '.common | to_entries[] | select(.value.pip == true) | "\(.key)==\(.value.version)"' "$VERSIONS_FILE"
    echo ""
    echo "# CUDA base packages (requirements/cuda/base.txt):"
    jq -r '.cuda.base | to_entries[] | select(.value.pip == true) | "\(.key)==\(.value.version)"' "$VERSIONS_FILE"
    echo ""
    echo "# CUDA task packages:"
    for task in $(get_platform_tasks "cuda"); do
        local task_pkgs=$(jq -r ".cuda.\"${task}\" | to_entries[] | select(.value.pip == true) | \"\(.key)==\(.value.version)\"" "$VERSIONS_FILE")
        if [ -n "$task_pkgs" ]; then
            echo "# - $task (requirements/cuda/${task}.txt):"
            echo "$task_pkgs"
        fi
    done
}

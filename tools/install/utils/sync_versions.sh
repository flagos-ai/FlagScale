#!/bin/bash
# =============================================================================
# Sync Package Versions from versions.json to Requirements Files
# =============================================================================
#
# Updates requirements files to match versions defined in versions.json.
# Only updates packages marked with "pip": true.
#
# Usage:
#   ./sync_versions.sh [--check]
#
# Options:
#   --check    Check if files are in sync (exit 1 if not)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VERSIONS_FILE="$PROJECT_ROOT/versions.json"
REQUIREMENTS_DIR="$PROJECT_ROOT/requirements"

CHECK_MODE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# =============================================================================
# Prerequisites
# =============================================================================
check_prerequisites() {
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed"
        exit 1
    fi
    if [ ! -f "$VERSIONS_FILE" ]; then
        log_error "versions.json not found: $VERSIONS_FILE"
        exit 1
    fi
}

# =============================================================================
# Update Functions
# =============================================================================

# Update a single package version in a requirements file
# Returns 0 if updated, 1 if no change needed, 2 if package not found
update_package_version() {
    local file=$1
    local package=$2
    local version=$3

    if [ ! -f "$file" ]; then
        return 2
    fi

    # Escape special characters in package name for regex
    local escaped_package=$(echo "$package" | sed 's/[.[\*^$()+?{|]/\\&/g')

    # Check if package exists in file (with any version)
    if grep -qE "^${escaped_package}==" "$file"; then
        local current=$(grep -E "^${escaped_package}==" "$file" | head -1 | sed 's/.*==//')
        if [ "$current" = "$version" ]; then
            return 1  # No change needed
        fi
        # Update the version
        if [ "$CHECK_MODE" = false ]; then
            sed -i "s/^${escaped_package}==.*/${package}==${version}/" "$file"
        fi
        return 0  # Updated
    fi
    return 2  # Not found
}

# Sync versions for a flat section (common, dev)
sync_section() {
    local section=$1
    local target_file=$2
    local updates=0
    local mismatches=0

    log_info "Syncing $section packages to $target_file"

    # Get all pip packages from section
    local packages=$(jq -r ".\"${section}\" | to_entries[] | select(.value.pip == true) | .key" "$VERSIONS_FILE")

    for package in $packages; do
        local version=$(jq -r ".\"${section}\".\"${package}\".version" "$VERSIONS_FILE")

        update_package_version "$target_file" "$package" "$version"
        local result=$?

        case $result in
            0)
                if [ "$CHECK_MODE" = true ]; then
                    log_warn "MISMATCH: $package should be $version in $target_file"
                    ((mismatches++))
                else
                    log_info "Updated: $package==$version"
                    ((updates++))
                fi
                ;;
            1)
                # Already up to date
                ;;
            2)
                log_warn "Package $package not found in $target_file (skipped)"
                ;;
        esac
    done

    if [ "$CHECK_MODE" = true ]; then
        return $mismatches
    fi
    return 0
}

# Sync versions for a nested platform section (platform.base or platform.task)
# Usage: sync_platform_section "cuda" "base" "/path/to/base.txt"
#        sync_platform_section "cuda" "train" "/path/to/train.txt"
sync_platform_section() {
    local platform=$1
    local subsection=$2
    local target_file=$3
    local updates=0
    local mismatches=0

    log_info "Syncing $platform.$subsection packages to $target_file"

    # Get all pip packages from platform.subsection
    local packages=$(jq -r ".\"${platform}\".\"${subsection}\" | to_entries[] | select(.value.pip == true) | .key" "$VERSIONS_FILE" 2>/dev/null)

    [ -z "$packages" ] && return 0

    for package in $packages; do
        local version=$(jq -r ".\"${platform}\".\"${subsection}\".\"${package}\".version" "$VERSIONS_FILE")

        update_package_version "$target_file" "$package" "$version"
        local result=$?

        case $result in
            0)
                if [ "$CHECK_MODE" = true ]; then
                    log_warn "MISMATCH: $package should be $version in $target_file"
                    ((mismatches++))
                else
                    log_info "Updated: $package==$version"
                    ((updates++))
                fi
                ;;
            1)
                # Already up to date
                ;;
            2)
                log_warn "Package $package not found in $target_file (skipped)"
                ;;
        esac
    done

    if [ "$CHECK_MODE" = true ]; then
        return $mismatches
    fi
    return 0
}

# =============================================================================
# Main
# =============================================================================
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Sync package versions from versions.json to requirements files.

OPTIONS:
    --check    Check if files are in sync (exit 1 if not)
    --help     Show this help message

FILES:
    Source: $VERSIONS_FILE
    Target: $REQUIREMENTS_DIR/common.txt              (common section)
            $REQUIREMENTS_DIR/dev.txt                 (dev section)
            $REQUIREMENTS_DIR/<platform>/base.txt     (<platform>.base section)
            $REQUIREMENTS_DIR/<platform>/<task>.txt   (<platform>.<task> section)

STRUCTURE:
    versions.json supports nested platform sections:
      common              - Shared packages
      dev                 - Development tools
      <platform>          - Platform (e.g., cuda)
        base              - Base packages (torch, cuda version)
        <task>            - Task-specific packages (train, inference, rl)

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check)    CHECK_MODE=true; shift ;;
            --help|-h)  usage; exit 0 ;;
            *)          log_error "Unknown option: $1"; exit 1 ;;
        esac
    done
}

main() {
    parse_args "$@"
    check_prerequisites

    local total_mismatches=0

    if [ "$CHECK_MODE" = true ]; then
        log_info "Checking version sync status..."
    else
        log_info "Syncing versions from $VERSIONS_FILE"
    fi

    # Sync common packages
    if [ -f "$REQUIREMENTS_DIR/common.txt" ]; then
        sync_section "common" "$REQUIREMENTS_DIR/common.txt" || ((total_mismatches+=$?))
    fi

    # Sync dev packages to dev.txt
    if jq -e '.dev' "$VERSIONS_FILE" > /dev/null 2>&1; then
        if [ -f "$REQUIREMENTS_DIR/dev.txt" ]; then
            sync_section "dev" "$REQUIREMENTS_DIR/dev.txt" || ((total_mismatches+=$?))
        fi
    fi

    # Sync platform packages (skip common and dev)
    for platform in $(jq -r 'keys[] | select(. != "common" and . != "dev")' "$VERSIONS_FILE"); do
        # Sync platform.base to <platform>/base.txt
        local base_file="$REQUIREMENTS_DIR/$platform/base.txt"
        if [ -f "$base_file" ]; then
            sync_platform_section "$platform" "base" "$base_file" || ((total_mismatches+=$?))
        else
            log_warn "Platform base requirements not found: $base_file"
        fi

        # Sync platform.<task> to <platform>/<task>.txt for each task
        for task in $(jq -r ".\"${platform}\" | keys[] | select(. != \"base\" and . != \"ubuntu\")" "$VERSIONS_FILE" 2>/dev/null); do
            local task_file="$REQUIREMENTS_DIR/$platform/${task}.txt"
            if [ -f "$task_file" ]; then
                sync_platform_section "$platform" "$task" "$task_file" || ((total_mismatches+=$?))
            fi
        done
    done

    # Summary
    echo ""
    if [ "$CHECK_MODE" = true ]; then
        if [ $total_mismatches -gt 0 ]; then
            log_error "Found $total_mismatches version mismatches"
            log_info "Run '$0' without --check to fix"
            exit 1
        else
            log_info "All versions are in sync"
        fi
    else
        log_info "Version sync complete"
    fi
}

main "$@"

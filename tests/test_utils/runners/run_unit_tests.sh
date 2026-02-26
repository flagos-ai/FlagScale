#!/bin/bash
# Unit Test Runner
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/utils.sh"

# Defaults
PLATFORM="default"
DEVICE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform) PLATFORM="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        -h|--help) cat <<EOF && exit 0
Usage: $(basename "$0") [OPTIONS]

Run unit tests with platform-specific configurations.

OPTIONS:
    --platform PLATFORM  Platform type (default: default)
    --device DEVICE      Device type (e.g., a100, a800, h100, generic)
                         If not specified, runs tests for all devices in the platform
    -h, --help           Show this help message

EXAMPLES:
    # Run tests for all devices in the platform
    $(basename "$0") --platform cuda

    # Run tests for specific device
    $(basename "$0") --platform cuda --device a100
    $(basename "$0") --platform default
EOF
        ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

# Validate platform
validate_platform "$PLATFORM" "$SCRIPT_DIR" || exit 1

# Function to run unit tests for a specific device
run_unit_tests_for_device() {
    local device="$1"

    log_info "Running unit tests for device: $device"

    # Set up PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/flagscale/train:${PYTHONPATH:-}"

    # Print configuration
    echo "=========================================="
    echo "Running Unit Tests"
    echo "=========================================="
    echo "Platform:    $PLATFORM"
    echo "Device:      $device"
    echo "PYTHONPATH:  $PYTHONPATH"
    echo "=========================================="

    # Get unit test patterns from platform configuration
    PARSE_CMD="python \"$SCRIPT_DIR/parse_config.py\" --platform \"$PLATFORM\" --device \"$device\" --type unit"

    PATTERNS=$(eval "$PARSE_CMD" 2>/dev/null) || {
        log_error "Failed to parse test configuration for device: $device"
        return 1
    }

    # Extract include and exclude patterns using helper
    PATTERN_OUTPUT=$(echo "$PATTERNS" | python "$SCRIPT_DIR/helpers.py" extract-patterns)
    INCLUDE=$(echo "$PATTERN_OUTPUT" | grep "^INCLUDE=" | cut -d= -f2-)
    EXCLUDE=$(echo "$PATTERN_OUTPUT" | grep "^EXCLUDE=" | cut -d= -f2-)

    # Build pytest command with torchrun for distributed test support
    PYTEST_CMD="torchrun --nproc_per_node=8 -m pytest tests/unit_tests/ -v --tb=short"

    # Apply exclude patterns if any
    if [ -n "$EXCLUDE" ]; then
        PYTEST_CMD="torchrun --nproc_per_node=8 -m pytest $EXCLUDE tests/unit_tests/ -v --tb=short"
    fi

    log_info "Command: $PYTEST_CMD"

    # Run unit tests
    eval "$PYTEST_CMD"
    return $?
}

# If device is specified, run for that device only
if [ -n "$DEVICE" ]; then
    validate_device "$PLATFORM" "$DEVICE" "$SCRIPT_DIR" || exit 1
    run_unit_tests_for_device "$DEVICE"
    EXIT_CODE=$?
else
    # No device specified, run for all devices in the platform
    DEVICE_TYPES=$(get_device_types "$PLATFORM" "$SCRIPT_DIR")

    if [ -z "$DEVICE_TYPES" ] || [ "$DEVICE_TYPES" = "[]" ]; then
        log_error "No device types found for platform: $PLATFORM"
        exit 1
    fi

    log_info "Running tests for all devices: $DEVICE_TYPES"

    # Parse device types using helper
    DEVICES=$(echo "$DEVICE_TYPES" | python "$SCRIPT_DIR/helpers.py" parse-devices)

    OVERALL_EXIT_CODE=0
    for device in $DEVICES; do
        if ! run_unit_tests_for_device "$device"; then
            log_error "Unit tests failed for device: $device"
            OVERALL_EXIT_CODE=1
        fi
        echo ""
    done

    EXIT_CODE=$OVERALL_EXIT_CODE
fi

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    log_success "Unit tests passed"
else
    log_error "Unit tests failed (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE

#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Unit Test Runner
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/utils.sh"

# Defaults
PLATFORM="default"
DEVICE=""
COVERAGE_DIR_INPUT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform) PLATFORM="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --coverage-dir) COVERAGE_DIR_INPUT="$2"; shift 2 ;;
        -h|--help) cat <<EOF && exit 0
Usage: $(basename "$0") [OPTIONS]

Run unit tests with platform-specific configurations.

OPTIONS:
    --platform PLATFORM  Platform type (default: default)
    --device DEVICE      Device type (e.g., a100, a800, h100, generic)
                         If not specified, runs tests for all devices in the platform
    --coverage-dir DIR   Optional coverage output directory
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
[ -n "$COVERAGE_DIR_INPUT" ] && export COVERAGE_DIR="$COVERAGE_DIR_INPUT"

# Validate platform
validate_platform "$PLATFORM" "$SCRIPT_DIR" || exit 1

# Function to run unit tests for a specific device
run_unit_tests_for_device() {
    local device="$1"

    log_info "Running unit tests for device: $device"

    # Set up PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/flagscale/train:${PYTHONPATH:-}"
    export FLAGSCALE_TEST_PLATFORM="$PLATFORM"
    export FLAGSCALE_TEST_DEVICE_TYPE="$device"
    export FLAGSCALE_TEST_DIST_BACKEND="${FLAGSCALE_TEST_DIST_BACKEND:-$(default_dist_backend "$PLATFORM")}"
    export FLAGSCALE_TEST_TORCH_DEVICE_TYPE="${FLAGSCALE_TEST_TORCH_DEVICE_TYPE:-$(default_torch_device_type "$PLATFORM")}"

    # Print configuration
    echo "=========================================="
    echo "Running Unit Tests"
    echo "=========================================="
    echo "Platform:    $PLATFORM"
    echo "Device:      $device"
    echo "Backend:     $FLAGSCALE_TEST_DIST_BACKEND"
    echo "Torch device:$FLAGSCALE_TEST_TORCH_DEVICE_TYPE"
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

    # Build coverage config if COVERAGE_DIR is set
    USE_COVERAGE=false
    if [ -n "${COVERAGE_DIR:-}" ]; then
        USE_COVERAGE=true
        mkdir -p "$COVERAGE_DIR"
        COVERAGERC="$COVERAGE_DIR/.coveragerc"
        cat > "$COVERAGERC" <<EOF
[run]
parallel = true
source = $PROJECT_ROOT
data_file = $COVERAGE_DIR/.coverage
EOF
    fi

    # Auto-detect accelerator count for the selected platform.
    NPROC=$(detect_accelerator_count "$PLATFORM")
    if ! [[ "$NPROC" =~ ^[0-9]+$ ]] || [ "$NPROC" -le 0 ]; then
        NPROC=1
    fi
    log_info "Detected $NPROC accelerator(s)"

    # Use 'coverage run' instead of pytest-cov to avoid SQLite concurrent write conflicts:
    # each torchrun rank writes its own .coverage.<host>.<pid>.<random> fragment independently.
    if [ "$USE_COVERAGE" = true ]; then
        RUNNER_CMD="-m coverage run --rcfile=$COVERAGERC -m pytest"
    else
        RUNNER_CMD="-m pytest"
    fi

    TEST_TARGETS="tests/unit_tests/"
    if [ -n "$INCLUDE" ] && [ "$INCLUDE" != "*" ]; then
        TEST_TARGETS="$INCLUDE"
    fi

    PYTEST_CMD="torchrun --nproc_per_node=$NPROC $RUNNER_CMD $TEST_TARGETS -v --tb=short"
    wait_for_gpu
    # Apply exclude patterns if any
    if [ -n "$EXCLUDE" ]; then
        PYTEST_CMD="torchrun --nproc_per_node=$NPROC $RUNNER_CMD $EXCLUDE $TEST_TARGETS -v --tb=short"
    fi

    log_info "Command: $PYTEST_CMD"

    # Run unit tests
    set +e
    eval "$PYTEST_CMD"
    local test_exit=$?
    set -e

    # All ranks have exited — safe to combine fragment files and generate report
    if [ "$USE_COVERAGE" = true ]; then
        log_info "Combining distributed coverage data..."
        python -m coverage combine --rcfile="$COVERAGERC" "$COVERAGE_DIR"
        python -m coverage json --rcfile="$COVERAGERC" -o "$COVERAGE_DIR/coverage.json"
    fi

    return $test_exit
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

#!/bin/bash
# FlagScale Test Runner - Unified entry point for all tests
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

[ -f "$SCRIPT_DIR/utils.sh" ] || { echo "Error: utils.sh not found"; exit 1; }
source "$SCRIPT_DIR/utils.sh"

# Defaults
PLATFORM=""
DEVICE=""
TEST_TYPE=""
TASK=""
MODEL=""
TEST_LIST=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run FlagScale tests with platform-specific configurations.

OPTIONS:
    --platform PLATFORM    Platform: cuda (REQUIRED)
                           See tests/test_utils/config/platforms/ for available platforms
                           Use template.yaml to create new platform configurations
    --device DEVICE        Device type: a100, a800 (optional)
                           If not specified, runs tests for all devices in the platform
    --type TYPE            Test type: unit or functional (optional, runs all if not specified)
    --task TASK            Task name for functional tests: train, hetero_train (optional)
    --model MODEL          Model name: aquila, mixtral, deepseek (optional)
    --list TESTS           Comma-separated test list (optional)
    -h, --help             Show this help message

EXAMPLES:
    # Run all tests for all devices in the platform
    $(basename "$0") --platform cuda

    # Run all tests for specific device
    $(basename "$0") --platform cuda --device a100

    # Run specific test type for all devices
    $(basename "$0") --platform cuda --type unit

    # Run specific test type for specific device
    $(basename "$0") --platform cuda --device a100 --type unit
    $(basename "$0") --platform cuda --device a800 --type functional

    # Run specific functional task
    $(basename "$0") --platform cuda --device h100 --type functional --task train

    # Run specific model in a task
    $(basename "$0") --platform cuda --device a100 --type functional --task train --model aquila

    # Run specific test cases
    $(basename "$0") --platform cuda --device a100 --type functional --task train --model aquila --list tp2_pp2,tp4_pp2
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform) PLATFORM="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --type) TEST_TYPE="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --list) TEST_LIST="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

# Validate test type if provided
if [ -n "$TEST_TYPE" ] && [ "$TEST_TYPE" != "unit" ] && [ "$TEST_TYPE" != "functional" ]; then
    log_error "Invalid test type '$TEST_TYPE'. Must be 'unit' or 'functional'"
    exit 1
fi

# Require platform to be specified
if [ -z "$PLATFORM" ]; then
    log_error "Platform must be specified with --platform. Available platforms: cuda"
    log_error "See tests/test_utils/config/platforms/template.yaml to create new platforms"
    exit 1
fi

# Validate platform
validate_platform "$PLATFORM" "$SCRIPT_DIR" || exit 1

# Validate device if specified
if [ -n "$DEVICE" ]; then
    validate_device "$PLATFORM" "$DEVICE" "$SCRIPT_DIR" || exit 1
fi

# Display configuration
echo "=========================================="
echo "FlagScale Test Runner"
echo "=========================================="
echo "Platform:   $PLATFORM"
echo "Device:     ${DEVICE:-all}"
echo "Test Type:  ${TEST_TYPE:-all}"
echo "Task:       ${TASK:-all}"
echo "Model:      ${MODEL:-all}"
echo "Tests:      ${TEST_LIST:-all}"
echo "=========================================="

cd "$PROJECT_ROOT"

# Build arguments for test scripts
SCRIPT_ARGS="--platform $PLATFORM"
[ -n "$DEVICE" ] && SCRIPT_ARGS="$SCRIPT_ARGS --device $DEVICE"

# Determine which tests to run
run_unit=false
run_functional=false

if [ -z "$TEST_TYPE" ] && [ -z "$TASK" ]; then
    # No type specified - run all tests
    run_unit=true
    run_functional=true
elif [ "$TEST_TYPE" = "unit" ]; then
    run_unit=true
elif [ "$TEST_TYPE" = "functional" ] || [ -n "$TASK" ]; then
    run_functional=true
fi

# Run tests
EXIT_CODE=0

if [ "$run_unit" = "true" ]; then
    log_info "Running unit tests"
    if ! "$SCRIPT_DIR/run_unit_tests.sh" $SCRIPT_ARGS; then
        log_error "Unit tests failed"
        EXIT_CODE=1
    fi
fi

if [ "$run_functional" = "true" ]; then
    log_info "Running functional tests"
    func_args="$SCRIPT_ARGS"
    [ -n "$TASK" ] && func_args="$func_args --task $TASK"
    [ -n "$MODEL" ] && func_args="$func_args --model $MODEL"
    [ -n "$TEST_LIST" ] && func_args="$func_args --list $TEST_LIST"

    if ! "$SCRIPT_DIR/run_functional_tests.sh" $func_args; then
        log_error "Functional tests failed"
        EXIT_CODE=1
    fi
fi

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    log_success "All tests completed successfully!"
else
    log_error "Some tests failed"
fi
echo "=========================================="

exit $EXIT_CODE

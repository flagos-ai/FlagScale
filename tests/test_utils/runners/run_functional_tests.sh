#!/bin/bash
# Functional Test Runner
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/utils.sh"

# Defaults
PLATFORM="default"
DEVICE=""
TASK=""
MODEL=""
TEST_LIST=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform) PLATFORM="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --list) TEST_LIST="$2"; shift 2 ;;
        -h|--help) cat <<EOF && exit 0
Usage: $(basename "$0") [OPTIONS]

Run functional tests with platform-specific configurations.

OPTIONS:
    --platform PLATFORM  Platform type (default: default)
    --device DEVICE      Device type (e.g., a100, a800, h100, generic)
                         If not specified, runs tests for all devices in the platform
    --task TASK          Task name (e.g., train, hetero_train)
    --model MODEL        Model name (e.g., aquila, mixtral, deepseek)
    --list TESTS         Comma-separated test list
    -h, --help           Show this help message

EXAMPLES:
    # Run all tests for all devices in the platform
    $(basename "$0") --platform cuda

    # Run tests for specific device
    $(basename "$0") --platform cuda --device a100

    # Run specific task
    $(basename "$0") --platform cuda --device a800 --task train

    # Run specific model in a task
    $(basename "$0") --task train --model aquila --platform cuda --device h100

    # Run specific test cases
    $(basename "$0") --task train --model aquila --list tp2_pp2,tp4_pp2
EOF
        ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Function to run a single test
run_test() {
    local task="$1" model="$2" config="$3"
    local test_dir="tests/functional_tests/$task/$model"
    local conf_dir="$test_dir/conf"

    [ -d "$conf_dir" ] || { log_error "Config dir not found: $conf_dir"; return 1; }

    # Check config file exists
    local config_file=""
    if [ -f "$conf_dir/$config.yaml" ]; then
        config_file="$conf_dir/$config.yaml"
    elif [ -f "$conf_dir/$config.yml" ]; then
        config_file="$conf_dir/$config.yml"
    else
        log_error "Config not found: $conf_dir/$config.{yaml,yml}"
        return 1
    fi

    log_info "Running: $task/$model/$config"
    wait_for_gpu

    # Clean old results
    # Extract exp_dir from config file and clean it
    local exp_dir=$(grep -E '^\s*exp_dir:' "$config_file" | head -1 | sed 's/.*exp_dir:\s*//' | tr -d '"' | tr -d "'")
    if [ -n "$exp_dir" ]; then
        log_info "Cleaning old results in: $exp_dir"
        rm -rf "$exp_dir"/* 2>/dev/null || true
    fi

    # Run test
    log_info "Start operation for executing the task: "
    log_info "  python run.py --config-path ${conf_dir} --config-name ${config} action=test"
    python run.py --config-path ${conf_dir} --config-name ${config} action=test || return 1

    # Match the corresponding comparison function according to task type
    # Matching rules:
    #   - *train*: Tasks containing "train" (e.g., train, hetero_train), use training result comparison function
    #   - inference: Exact match for inference tasks, use inference result comparison function
    #   - *: Unsupported task type, throw error and exit execution
    local compare_function
    case "$task" in
        *train*)
            compare_function="test_train_equal"
            ;;
        inference)
            compare_function="test_inference_equal"
            ;;
        serve)
            compare_function="test_serve_equal"
            ;;
        rl)
            compare_function="test_rl_equal"
            ;;
        *)
            log_error "Unsupported task type: $task, no corresponding comparison function for standard vs test values"
            return 1
            ;;
    esac

    # Validate results if validator exists
    if [ -f "$PROJECT_ROOT/tests/test_utils/runners/check_results.py" ]; then
        local validator_cmd="python -m pytest \"$PROJECT_ROOT/tests/test_utils/runners/check_results.py::$compare_function\" \
            --path=tests/functional_tests --task=\"$task\" --model=\"$model\" \
            --case=\"$config\" --platform=\"$PLATFORM\""
        [ -n "$CURRENT_DEVICE" ] && validator_cmd="$validator_cmd --device=\"$CURRENT_DEVICE\""

        if [ "$task" = "serve" ]; then
            log_info "Waiting 1 minute for service to be ready..."
            sleep 1m
        fi

        if ! eval "$validator_cmd"; then
            log_error "Validation failed for $task/$model/$config"
            return 1
        fi

        if [ "$task" = "serve" ]; then
            log_info "Stop operation for executing the serve task: "
            log_info "  python run.py --config-path ${conf_dir} --config-name ${config} action=stop"
            python run.py --config-path ${conf_dir} --config-name ${config} action=stop
        fi
    fi

    log_success "Test completed: $task/$model/$config"
}

# Get tests from platform configuration
get_test_configs() {
    local device="$1"
    local task="$2"
    local model="$3"
    local list="$4"

    local cmd="python \"$SCRIPT_DIR/parse_config.py\" --platform \"$PLATFORM\" --device \"$device\" --type functional --task \"$task\""
    [ -n "$model" ] && cmd="$cmd --model \"$model\""
    [ -n "$list" ] && cmd="$cmd --list \"$list\""
    eval "$cmd" 2>/dev/null || echo ""
}

# Parse and run tests using helper module
run_tests_from_json() {
    local tests_json="$1"
    local failed=0

    # Use helper module to parse test cases
    # Use process substitution to avoid subshell and allow failure tracking
    while IFS=' ' read -r task model config; do
        [ -z "$task" ] && continue
        if ! run_test "$task" "$model" "$config"; then
            log_error "FAIL: $task/$model/$config"
            failed=1
        fi
    done < <(echo "$tests_json" | python "$SCRIPT_DIR/helpers.py" parse-test-cases)

    return $failed
}

# Function to run functional tests for a specific device
run_functional_tests_for_device() {
    local device="$1"
    export CURRENT_DEVICE="$device"
    local failed=0

    log_info "Running functional tests for device: $device"

    # Print configuration
    echo "=========================================="
    echo "Running Functional Tests"
    echo "=========================================="
    echo "Platform:   $PLATFORM"
    echo "Device:     $device"
    echo "Task:       ${TASK:-all}"
    echo "Model:      ${MODEL:-all}"
    echo "Tests:      ${TEST_LIST:-all}"
    echo "=========================================="

    cd "$PROJECT_ROOT"

    # If no task specified, run all tasks
    if [ -z "$TASK" ]; then
        for task_dir in tests/functional_tests/*/; do
            task_name=$(basename "$task_dir")
            [ -d "$task_dir" ] || continue

            log_info "Processing task: $task_name"

            tests_json=$(get_test_configs "$device" "$task_name" "$MODEL" "$TEST_LIST")
            [ -z "$tests_json" ] && { log_info "No tests found for task=$task_name"; continue; }

            if ! run_tests_from_json "$tests_json"; then
                log_error "Some tests failed for task: $task_name"
                failed=1
            fi
        done
    else
        # Task specified, run only that task
        [ -d "tests/functional_tests/$TASK" ] || { log_error "Task directory not found: tests/functional_tests/$TASK"; return 1; }

        tests_json=$(get_test_configs "$device" "$TASK" "$MODEL" "$TEST_LIST")
        [ -z "$tests_json" ] && { log_error "No tests found for task=$TASK"; return 1; }

        if ! run_tests_from_json "$tests_json"; then
            log_error "Some tests failed for task: $TASK"
            failed=1
        fi
    fi

    return $failed
}

# Validate platform
validate_platform "$PLATFORM" "$SCRIPT_DIR" || exit 1

# If device is specified, run for that device only
if [ -n "$DEVICE" ]; then
    validate_device "$PLATFORM" "$DEVICE" "$SCRIPT_DIR" || exit 1
    run_functional_tests_for_device "$DEVICE"
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
        if ! run_functional_tests_for_device "$device"; then
            log_error "Functional tests failed for device: $device"
            OVERALL_EXIT_CODE=1
        fi
        echo ""
    done

    EXIT_CODE=$OVERALL_EXIT_CODE
fi

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    log_success "All tests completed"
else
    log_error "Some tests failed"
fi
echo "=========================================="

exit $EXIT_CODE

#!/bin/bash
# FlagScale Test Runner - Unified entry point for all tests
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

[ -f "$SCRIPT_DIR/utils.sh" ] || { echo "Error: utils.sh not found"; exit 1; }
source "$SCRIPT_DIR/utils.sh"

# Defaults
PLATFORM="default" TEST_TYPE="" TASK="" MODEL="" TEST_LIST=""

usage() { cat <<EOF
Usage: $(basename "$0") [OPTIONS]
Run FlagScale tests with platform-specific configurations.

USAGE SCENARIOS:
  1. Run all tests for a platform:
     $(basename "$0") --platform default

  2. Run specific test type for a platform:
     $(basename "$0") --platform default --type unit
     $(basename "$0") --platform default --type functional

  3. Run specific task within functional tests:
     $(basename "$0") --platform default --type functional --task train

  4. Run specific model within a task:
     $(basename "$0") --platform default --type functional --task train --model aquila

  5. Run specific test cases from a model:
     $(basename "$0") --platform default --type functional --task train --model aquila --list tp2_pp2,tp4_pp2

OPTIONS:
    --platform PLATFORM    Platform: default (cpu) or a100 (nvidia) (default: default)
    --type TYPE           Test type: unit or functional (optional)
    --task TASK           Task name for functional tests: train, hetero_train (optional)
    --model MODEL         Model name: aquila, mixtral, deepseek, etc (optional)
    --list TESTS          Comma-separated test list (optional)
    -h, --help            Show this help message
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform) PLATFORM="$2"; shift 2 ;;
        --type) TEST_TYPE="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --list) TEST_LIST="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Error: Unknown option '$1'" >&2; usage ;;
    esac
done

# Validate test type if provided
if [ -n "$TEST_TYPE" ] && [ "$TEST_TYPE" != "unit" ] && [ "$TEST_TYPE" != "functional" ]; then
    echo "Error: Invalid test type '$TEST_TYPE'. Must be 'unit' or 'functional'" >&2
    usage
fi

# Display info
echo "=========================================="
echo "FlagScale Test Runner"
echo "=========================================="
echo "Platform:   $PLATFORM"
echo "Test Type:  ${TEST_TYPE:-all}"
echo "Task:       ${TASK:-all}"
echo "Model:      ${MODEL:-all}"
echo "Tests:      ${TEST_LIST:-all}"
echo "=========================================="

cd "$PROJECT_ROOT"

# Scenario 1: Only platform specified - run all tests
if [ -z "$TEST_TYPE" ] && [ -z "$TASK" ]; then
    echo "[INFO] Running all tests for platform: $PLATFORM"
    "$SCRIPT_DIR/run_unit_tests.sh" --platform "$PLATFORM"
    "$SCRIPT_DIR/run_functional_tests.sh" --platform "$PLATFORM"
    exit 0
fi

# Scenario 2: Platform + Type specified
if [ "$TEST_TYPE" = "unit" ]; then
    echo "[INFO] Running unit tests for platform: $PLATFORM"
    "$SCRIPT_DIR/run_unit_tests.sh" --platform "$PLATFORM"
    exit 0
fi

# Scenarios 3-5: Platform + functional + optional task/model/list
if [ "$TEST_TYPE" = "functional" ] || [ -n "$TASK" ]; then
    args="--platform $PLATFORM"
    [ -n "$TASK" ] && args="$args --task $TASK"
    [ -n "$MODEL" ] && args="$args --model $MODEL"
    [ -n "$TEST_LIST" ] && args="$args --list $TEST_LIST"

    echo "[INFO] Running functional tests"
    "$SCRIPT_DIR/run_functional_tests.sh" $args
    exit 0
fi

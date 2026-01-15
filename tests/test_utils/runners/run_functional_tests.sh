#!/bin/bash
# Functional Test Runner with platform-aware filtering
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/utils.sh"

TASK="" MODEL="" TEST_LIST="" PLATFORM="default"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --list) TEST_LIST="$2"; shift 2 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        -h|--help) cat <<EOF && exit 0
Usage: $(basename "$0") [--task TASK] [--model MODEL] [--list TESTS] [--platform PLATFORM]
Run functional test cases with platform filtering.

SCENARIOS:
  1. Run all tasks with all models/configs:
     $(basename "$0") --platform default
     
  2. Run all models/configs in a task:
     $(basename "$0") --task train --platform default
     
  3. Run specific model in a task:
     $(basename "$0") --task train --model aquila --platform default
     
  4. Run specific test cases from a model:
     $(basename "$0") --task train --model aquila --list tp2_pp2,tp4_pp2 --platform default

OPTIONS:
    --task TASK          Task name (optional): train, hetero_train (default: all tasks)
    --model MODEL        Model name (optional): aquila, mixtral, deepseek (default: all models)
    --list TESTS         Comma-separated test list (optional) (default: all tests)
    --platform PLATFORM  Platform type (default: default)
EOF
        ;;
        *) echo "Error: Unknown option: $1" >&2; exit 1 ;;
    esac
done

run_test() {
    local task="$1" model="$2" config="$3"
    local test_dir="tests/functional_tests/$task/$model"
    local conf_dir="$test_dir/conf"
    
    [ -d "$conf_dir" ] || { echo "Error: Config dir not found: $conf_dir" >&2; return 1; }
    
    # Check config file exists
    local config_file=""
    if [ -f "$conf_dir/$config.yaml" ]; then
        config_file="$conf_dir/$config.yaml"
    elif [ -f "$conf_dir/$config.yml" ]; then
        config_file="$conf_dir/$config.yml"
    else
        echo "Error: Config not found: $conf_dir/$config.{yaml,yml}" >&2
        return 1
    fi
    
    echo "[INFO] Running: $task/$model/$config"
    wait_for_gpu
    
    # Clean old results from exp_dir defined in the yaml config
    # Extract exp_dir from yaml (handles format like "exp_dir: path/to/dir")
    local exp_dir=$(grep -E '^\s*exp_dir:' "$config_file" | head -1 | sed 's/.*exp_dir:\s*//' | tr -d '"' | tr -d "'")
    if [ -n "$exp_dir" ]; then
        echo "[INFO] Cleaning previous results in: $exp_dir"
        rm -rf "$exp_dir"/* 2>/dev/null || true
    fi
    
    # Clean old results (legacy path)
    rm -rf "$test_dir/results_test/$config" 2>/dev/null || true
    
    # Run test
    python run.py --config-path "$conf_dir" --config-name "$config" action=test || return 1
    
    # Validate results (if validator exists)
    if [ -f "$PROJECT_ROOT/tests/test_utils/runners/check_results.py" ]; then
        python -m pytest "$PROJECT_ROOT/tests/test_utils/runners/check_results.py::test_train_equal" \
            --test_path=tests/functional_tests --test_type="$task" --test_task="$model" \
            --test_case="$config" --platform="$PLATFORM" 2>/dev/null || true
    fi
    
    echo "[OK] Test completed: $task/$model/$config"
}

cd "$PROJECT_ROOT"
echo "[INFO] =========================================="
echo "[INFO] Platform: $PLATFORM"
echo "[INFO] Task: ${TASK:-all}, Model: ${MODEL:-all}, Tests: ${TEST_LIST:-all}"
echo "[INFO] =========================================="

# Get tests from platform configuration using parse_config.py
get_test_configs() {
    python "$SCRIPT_DIR/parse_config.py" --platform "$PLATFORM" --type functional --task "$1" ${2:+--model "$2"} ${3:+--list "$3"} 2>/dev/null || echo ""
}

cd "$PROJECT_ROOT"

# If no task specified, run all tasks
if [ -z "$TASK" ]; then
    # Discover all tasks from functional_tests directory
    for task_dir in tests/functional_tests/*/; do
        task_name=$(basename "$task_dir")
        [ -d "$task_dir" ] || continue
        
        echo "[INFO] Processing task: $task_name"
        
        # Get test configuration from platform YAML for this task
        tests_json=$(get_test_configs "$task_name" "$MODEL" "$TEST_LIST") || {
            echo "Warning: Failed to get test configuration for task=$task_name" >&2
            continue
        }
        
        if [ -z "$tests_json" ]; then
            echo "Warning: No tests found for task=$task_name, model=$MODEL, list=$TEST_LIST in platform=$PLATFORM" >&2
            continue
        fi
        
        # Parse JSON and run tests
        cat > /tmp/parse_json_tests.py << 'EOF'
import json, sys
tests_json_str = sys.argv[1]
tests_config = json.loads(tests_json_str)
for task, models_data in tests_config.items():
    for model, test_configs in models_data.items():
        if isinstance(test_configs, list):
            for config in test_configs:
                print(f"{task} {model} {config}")
EOF

        python /tmp/parse_json_tests.py "$tests_json" | while read task model config; do
            [ -n "$task" ] && [ -n "$model" ] && [ -n "$config" ] && run_test "$task" "$model" "$config"
        done
    done
    
    rm -f /tmp/parse_json_tests.py
else
    # Task specified, run only that task
    # Validate task directory exists
    task_dir="tests/functional_tests/$TASK"
    [ -d "$task_dir" ] || { echo "Error: Task directory not found: $task_dir" >&2; exit 1; }

    # Get test configuration from platform YAML
    tests_json=$(get_test_configs "$TASK" "$MODEL" "$TEST_LIST") || {
        echo "Error: Failed to get test configuration from platform YAML" >&2
        exit 1
    }

    if [ -z "$tests_json" ]; then
        echo "Error: No tests found for task=$TASK, model=$MODEL, list=$TEST_LIST in platform=$PLATFORM" >&2
        exit 1
    fi

    # Parse JSON and run tests
    cat > /tmp/parse_json_tests.py << 'EOF'
import json, sys
tests_json_str = sys.argv[1]
tests_config = json.loads(tests_json_str)
for task, models_data in tests_config.items():
    for model, test_configs in models_data.items():
        if isinstance(test_configs, list):
            for config in test_configs:
                print(f"{task} {model} {config}")
EOF

    python /tmp/parse_json_tests.py "$tests_json" | while read task model config; do
        [ -n "$task" ] && [ -n "$model" ] && [ -n "$config" ] && run_test "$task" "$model" "$config"
    done

    rm -f /tmp/parse_json_tests.py
fi

echo "[OK] =========================================="
echo "[OK] All tests completed successfully"
echo "[OK] =========================================="

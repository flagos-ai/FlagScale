#!/bin/bash
# Platform configuration loading script
# Extracts platform configuration and groups tests by task
# Usage: source load_platform_config.sh && load_platform_config <platform_name>

set -euo pipefail

load_platform_config() {
    local PLATFORM=$1
    local CONFIG_FILE=".github/configs/${PLATFORM}.yml"

    echo "Loading platform configuration for: $PLATFORM"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "❌ Error: Platform configuration file not found: $CONFIG_FILE"
        return 1
    fi

    # Extract CI/CD configuration from .github/configs using yq
    echo "Extracting configuration from $CONFIG_FILE"
    CI_IMAGE=$(/usr/local/bin/yq -r '.ci_image' "$CONFIG_FILE")
    RUNNER_LABELS=$(/usr/local/bin/yq -o=json -I=0 '.runner_labels' "$CONFIG_FILE")
    VOLUMES=$(/usr/local/bin/yq -o=json -I=0 '.container_volumes' "$CONFIG_FILE")
    CONTAINER_OPTIONS=$(/usr/local/bin/yq -r '.container_options' "$CONFIG_FILE")

    # Extract package manager configuration
    PKG_MGR=$(/usr/local/bin/yq -r '.pkg_mgr // "uv"' "$CONFIG_FILE")
    ENV_PATH=$(/usr/local/bin/yq -r '.env_path // "/opt/venv"' "$CONFIG_FILE")
    ENV_NAME_TRAIN=$(/usr/local/bin/yq -r '.env_names.train // "flagscale-train"' "$CONFIG_FILE")
    ENV_NAME_INFERENCE=$(/usr/local/bin/yq -r '.env_names.inference // "flagscale-inference"' "$CONFIG_FILE")
    ENV_NAME_SERVE=$(/usr/local/bin/yq -r '.env_names.serve // "flagscale-serve"' "$CONFIG_FILE")
    ENV_NAME_RL=$(/usr/local/bin/yq -r '.env_names.rl // "flagscale-rl"' "$CONFIG_FILE")

    echo "Package manager: $PKG_MGR"
    echo "Environment path: $ENV_PATH"
    echo "Environment names: train=$ENV_NAME_TRAIN, inference=$ENV_NAME_INFERENCE, serve=$ENV_NAME_SERVE, rl=$ENV_NAME_RL"

    # Validate required fields
    if [ -z "$CI_IMAGE" ] || [ "$CI_IMAGE" = "null" ]; then
        echo "❌ Error: ci_image not found in $CONFIG_FILE"
        return 1
    fi
    if [ -z "$RUNNER_LABELS" ] || [ "$RUNNER_LABELS" = "null" ]; then
        echo "❌ Error: runner_labels not found in $CONFIG_FILE"
        return 1
    fi

    # Query device types from test platform configuration
    echo "Querying device types for platform: $PLATFORM"
    DEVICE_TYPES=$(python tests/test_utils/runners/parse_config.py --platform "$PLATFORM" --type device_types 2>/dev/null || echo '[]')
    echo "Device types: $DEVICE_TYPES"

    # Discover functional test matrix (device/task/model/case combinations)
    echo "Discovering functional test matrix..."
    FUNCTIONAL_TEST_MATRIX="[]"

    # Parse functional tests from platform configuration for each device type
    if [ -f "tests/test_utils/runners/parse_config.py" ]; then
        FUNCTIONAL_TEST_MATRIX=$(echo "$DEVICE_TYPES" | jq -r '.[]' | while read device; do
            # Get functional tests for this device from platform config
            functional_json=$(python tests/test_utils/runners/parse_config.py \
                --platform "$PLATFORM" \
                --device "$device" \
                --type functional 2>/dev/null || echo '{}')

            # Convert the functional tests JSON to device/task/model/case matrix entries
            echo "$functional_json" | jq -r 'to_entries[] | .key as $task | .value | to_entries[] | .key as $model | .value[] | {device: "'"$device"'", task: $task, model: $model, case: .}'
        done | jq -s -c '.')
    fi
    echo "Functional test matrix: $FUNCTIONAL_TEST_MATRIX"

    # Group tests by task type for separate workflows
    echo "Grouping tests by task type..."
    TRAIN_TESTS=$(echo "$FUNCTIONAL_TEST_MATRIX" | jq -c '[.[] | select(.task == "train")]')
    HETERO_TRAIN_TESTS=$(echo "$FUNCTIONAL_TEST_MATRIX" | jq -c '[.[] | select(.task == "hetero_train")]')
    INFERENCE_TESTS=$(echo "$FUNCTIONAL_TEST_MATRIX" | jq -c '[.[] | select(.task == "inference")]')
    SERVE_TESTS=$(echo "$FUNCTIONAL_TEST_MATRIX" | jq -c '[.[] | select(.task == "serve")]')
    RL_TESTS=$(echo "$FUNCTIONAL_TEST_MATRIX" | jq -c '[.[] | select(.task == "rl")]')

    echo "Train tests: $(echo "$TRAIN_TESTS" | jq 'length') test(s)"
    echo "Hetero train tests: $(echo "$HETERO_TRAIN_TESTS" | jq 'length') test(s)"
    echo "Inference tests: $(echo "$INFERENCE_TESTS" | jq 'length') test(s)"
    echo "Serve tests: $(echo "$SERVE_TESTS" | jq 'length') test(s)"
    echo "RL tests: $(echo "$RL_TESTS" | jq 'length') test(s)"

    echo "✅ Loaded config for platform: $PLATFORM"

    # Output values to $GITHUB_OUTPUT
    echo "ci_image=$CI_IMAGE" >> $GITHUB_OUTPUT
    echo "container_options=$CONTAINER_OPTIONS" >> $GITHUB_OUTPUT

    { echo 'runs_on<<EOFRUNSON'; echo "$RUNNER_LABELS"; echo 'EOFRUNSON'; } >> $GITHUB_OUTPUT
    { echo 'container_volumes<<EOFVOLUMES'; echo "$VOLUMES"; echo 'EOFVOLUMES'; } >> $GITHUB_OUTPUT
    { echo 'device_types<<EOFDEVICETYPES'; echo "$DEVICE_TYPES"; echo 'EOFDEVICETYPES'; } >> $GITHUB_OUTPUT
    { echo 'train_test_matrix<<EOFTRAIN'; echo "$TRAIN_TESTS"; echo 'EOFTRAIN'; } >> $GITHUB_OUTPUT
    { echo 'hetero_train_test_matrix<<EOFHETEROTRAIN'; echo "$HETERO_TRAIN_TESTS"; echo 'EOFHETEROTRAIN'; } >> $GITHUB_OUTPUT
    { echo 'inference_test_matrix<<EOFINFERENCE'; echo "$INFERENCE_TESTS"; echo 'EOFINFERENCE'; } >> $GITHUB_OUTPUT
    { echo 'serve_test_matrix<<EOFINFERENCE'; echo "$SERVE_TESTS"; echo 'EOFINFERENCE'; } >> $GITHUB_OUTPUT
    { echo 'rl_test_matrix<<EOFRL'; echo "$RL_TESTS"; echo 'EOFRL'; } >> $GITHUB_OUTPUT

    # Output package manager configuration
    echo "pkg_mgr=$PKG_MGR" >> $GITHUB_OUTPUT
    echo "env_path=$ENV_PATH" >> $GITHUB_OUTPUT
    echo "env_name_train=$ENV_NAME_TRAIN" >> $GITHUB_OUTPUT
    echo "env_name_inference=$ENV_NAME_INFERENCE" >> $GITHUB_OUTPUT
    echo "env_name_serve=$ENV_NAME_SERVE" >> $GITHUB_OUTPUT
    echo "env_name_rl=$ENV_NAME_RL" >> $GITHUB_OUTPUT
}

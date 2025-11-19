#!/bin/bash

# Run each command and capture its return value
commands=(
    # For inference
    "tests/scripts/functional_tests/test_task.sh --type inference --task deepseek_r1_distill_qwen"
    "tests/scripts/functional_tests/test_task.sh --type inference --task deepseek_r1_distill_qwen-flaggems"
    "tests/scripts/functional_tests/test_task.sh --type inference --task opi_llama3_1_instruct"
    "tests/scripts/functional_tests/test_task.sh --type inference --task opi_llama3_1_instruct-flaggems"
    "tests/scripts/functional_tests/test_task.sh --type inference --task qwen3"
    "tests/scripts/functional_tests/test_task.sh --type inference --task qwen3-flaggems"
    "tests/scripts/functional_tests/test_task.sh --type inference --task robobrain2"
    "tests/scripts/functional_tests/test_task.sh --type inference --task robobrain2-flaggems"
)

for cmd in "${commands[@]}"; do
    # Execute the command
    $cmd
    # Capture the return value
    return_value=$?
    # Check if the return value is non-zero (an error occurred)
    if [ $return_value -ne 0 ]; then
        # Output the error message and the failing command
        echo "Error: Command '$cmd' failed"
        # Throw an exception by exiting the script with a non-zero status
        exit 1
    fi
done

#!/bin/bash

# Run each command and capture its return value
commands=(
    # For train
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type train --task aquila"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type train --task deepseek"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type train --task mixtral"
    #"tests/scripts/benchmark_tests/test_task.sh --type train --task llava_onevision"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type hetero_train --task aquila"
    
    # For inference
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task deepseek_r1_distill_qwen"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task deepseek_r1_distill_qwen-flaggems"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task deepseek_r1_distill_qwen-metax"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task deepseek_r1_distill_qwen-flaggems-metax"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task opi_llama3_1_instruct-metax"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task opi_llama3_1_instruct-flaggems-metax"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task qwen3"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task qwen3-flaggems"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task qwen3-metax"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task qwen3-flaggems-metax"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task robobrain2-metax"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task robobrain2-flaggems-metax"
    "tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task robobrain2"
    #"tests/benchmarks/scripts/benchmark_tests/test_task.sh --type inference --task robobrain2-flaggems"
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

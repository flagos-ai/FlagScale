#!/bin/bash
echo "Set fd limit to 65535"
ulimit -n 65535

if [ -d "/workspace/report/0/cov-report-megatron" ]; then rm -r /workspace/report/0/cov-report-megatron; fi
if [ -d "/workspace/report/0/cov-temp-megatron" ]; then rm -r /workspace/report/0/cov-temp-megatron; fi
if [ -d "/workspace/report/0/cov-report-flagscale" ]; then rm -r /workspace/report/0/cov-report-flagscale; fi
if [ -d "/workspace/report/0/cov-temp-flagscale" ]; then rm -r /workspace/report/0/cov-temp-flagscale; fi

# Run each command and capture its return value
commands=(
    # unit tests -> train_flagscale
    "tests/scripts/unit_tests/test_subset.sh --backend train_flagscale --subset runner"
    "tests/scripts/unit_tests/test_subset.sh --backend train_flagscale --subset ./"
    # # coverage test -> train_flagscale
    # "tests/scripts/unit_tests/test_coverage.sh --backend train_flagscale --status offline"

    # unit tests -> inference_flagscale
    "tests/scripts/unit_tests/test_subset.sh --backend inference_flagscale --subset inference"
    "tests/scripts/unit_tests/test_subset.sh --backend inference_flagscale --subset transformations"

    # You can add your own test subset here
    # "tests/scripts/unit_tests/test_subset.sh --backend flagscale --subset your_subset"
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
    echo "Success: Command '$cmd' succeeded"
done

#!/bin/bash

# Run each command and capture its return value
commands=(
    # For train
    "tests/scripts/functional_tests/test_task.sh --type train --task aquila"
    "tests/scripts/functional_tests/test_task.sh --type train --task deepseek"
    "tests/scripts/functional_tests/test_task.sh --type train --task mixtral"
    "tests/scripts/functional_tests/test_task.sh --type hetero_train --task aquila"
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

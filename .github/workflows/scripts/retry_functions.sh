#!/bin/bash

# A retry strategy is triggered automatically upon failure in installing dependencies,
# especially under network instability or interruptions.

retry() {
    local retries=$1
    shift
    local cmd="$*"
    local count=0
    until eval "$cmd"; do
        count=$((count + 1))
        if [ $count -ge $retries ]; then
            echo "❌ Command failed after $retries retries: $*"
            return 1
        fi
        echo "⚠️ Command failed (attempt $count/$retries), retrying in 5 seconds..."
        sleep 5
    done
    echo "✅ Command succeeded after $count retries: $*"
    return 0
}

# Retry command array: Parameter 1 specifies the retry attempts,
# Parameter 2 provides the command array's name as a string.
retry_commands() {
    local retries=$1
    shift
    # How to convert a string to an array in Bash to handle inter-function array passing.
    local -a cmds=("$@")

    echo "🔍 Retry config: max retries = $retries"
    echo "📝 Total commands to execute: ${#cmds[@]}"

    # For each command in the array, run it and retry upon failure.
    # Proceed to return only when all commands have executed successfully.
    for cmd in "${cmds[@]}"; do
        echo "🔧 Executing command: $cmd"
        # Run commands with special characters or variables via eval,
        # and trigger the retry function as needed.
        retry $retries "$cmd"
        local cmd_exit_code=$?
        if [ $cmd_exit_code -ne 0 ]; then
            echo "❌ Batch commands failed at: $cmd"
            return $cmd_exit_code
        fi
    done
    echo "✅ All batch commands executed successfully!"
    return 0
}

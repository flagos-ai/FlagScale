#!/bin/bash

set -eo pipefail

echo "The current directory is: $(pwd)"

# Function to execute a command and handle failures
run_command() {
  echo "$1"
  eval "$1"
  local command_status=$?  # Capture the command's exit status
  local attempt_i=$2       # Capture the attempt number from the second argument
  local type_name=$3       # Capture the test type name from the third argument
  local task_name=$4       # Capture the test task name from the fourth argument
  local case_name=$5       # Capture the test case name from the fifth argument

  # If the command status is not equal to 0, it indicates failure
  if [ $command_status -ne 0 ]; then
      echo "Test failed on attempt $attempt_i for $task_name $type_name $case_name."  # Print failure message
      exit 1  # Exit the script
  fi
}

source tests/scripts/_gpu_check.sh

# Path to the YAML configuration file
CONFIG_FILE="tests/scripts/functional_tests/config.yml"

# Function to parse the configuration file and run tests
test_task() {
  local _type=$1
  local _task=$2
  # Use parse_config.py to parse the YAML file with test type and test task
  local _cases=$(python tests/scripts/functional_tests/parse_config.py --config $CONFIG_FILE --type $_type --task $_task)

  # Convert the parsed test cases to an array
  IFS=' ' read -r -a _cases <<< "$_cases"

  # Check if _cases is not an empty list
  if [ ${#_cases[@]} -eq 0 ]; then
    echo "No test cases found for task '$_task' with test type '$_type'. Exiting."
    exit 1
  fi

  # Loop through each test case, remove leading '-', and run the test
  for _case in "${_cases[@]}"; do
    # Remove leading '-'
    _case=${_case#-}

    test_times=1
    # Attempt to run the test specified number of times
    for attempt_i in $(seq 1 $test_times); do
      echo "---------"
      echo "Attempt $attempt_i for task ${_task} with type ${_type} and case: ${_case}"
      echo "---------"

      wait_for_gpu

      # Remove previous results if they exist
      if [ "${_type}" = "train" ] || [ "${_type}" = "hetero_train" ] || [ "${_type}" = "inference" ] || [ "${_type}" = "rl" ] || [ "${_type}" = "serve" ]; then
        result_path="tests/functional_tests/test_cases/${_type}/${_task}/results_test/${_case}"
        if [ -d "$result_path" ]; then
          rm -r "$result_path"
        fi
      fi

      if [ "${_type}" = "train" ] || [ "${_type}" = "hetero_train" ]; then
        run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_task}/conf --config-name ${_case} action=test" $attempt_i $_task $_type $_case
        run_command "pytest tests/functional_tests/test_utils/test_result.py::test_train_equal --test_path=tests/functional_tests/test_cases --test_type=${_type} --test_task=${_task} --test_case=${_case}" $attempt_i $_task $_type $_case
      fi

      if [ "${_type}" = "inference" ]; then
        run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_task}/conf --config-name ${_case} action=test" $attempt_i $_task $_type $_case
        run_command "pytest -s tests/functional_tests/test_utils/test_result.py::test_inference_equal --test_path=tests/functional_tests/test_cases --test_type=${_type} --test_task=${_task} --test_case=${_case}" $attempt_i $_task $_type $_case
      fi

      if [ "${_type}" = "serve" ]; then
         run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_task}/conf --config-name ${_case} action=test; sleep 1m" $attempt_i $_task $_type $_case
         run_command "pytest -s tests/functional_tests/test_utils/test_result.py::test_serve_equal --test_path=tests/functional_tests/test_cases --test_type=${_type} --test_task=${_task} --test_case=${_case}" $attempt_i $_task $_type $_case
         run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_task}/conf --config-name ${_case} action=stop" $attempt_i $_task $_type $_case
      fi

      if [ "${_type}" = "rl" ]; then
        run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_task}/conf --config-name ${_case} action=test" $attempt_i $_task $_type $_case
        run_command "pytest -s tests/functional_tests/test_utils/test_result.py::test_rl_equal --test_path=tests/functional_tests/test_cases --test_type=${_type} --test_task=${_task} --test_case=${_case}" $attempt_i $_task $_type $_case
      fi

      # Ensure that pytest check is completed before deleting the folder
      sleep 10s
    done
    echo "All $test_times attempts successful for case $_case for task ${_task}."
  done
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type) type="$2"; shift ;;
        --task) task="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate that the required parameters --type and --task are provided
if [ -z "$type" ]; then
  echo "Error: --type is required"
  exit 1
fi

if [ -z "$task" ]; then
  echo "Error: --task is required"
  exit 1
fi

# Print final parameter values for confirmation
echo "Type: $type"
echo "Task: $task"

# Run the tests based on the provided test type and test task
test_task "$type" "$task"

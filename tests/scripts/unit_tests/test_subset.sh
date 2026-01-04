#!/bin/bash

set -eo pipefail

source tests/scripts/_gpu_check.sh

echo "The current directory is: $(pwd)"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backend) backend="$2"; shift ;;
        --subset) subset="$2"; shift ;;
        --coverage) coverage="$2"; shift ;;
        --id) id="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure necessary arguments are provided
if [ -z "$backend" ] || [ -z "$subset" ]; then
    echo "Usage: $0 --backend <backend> --subset <subset> [--id <id>]"
    exit 1
fi

# Set default values for optional parameters
coverage=${coverage:-"False"}

# Configuration file path
config_file="tests/scripts/unit_tests/config.yml"
if [ ! -f "$config_file" ]; then
    echo "Configuration file $config_file not found!"
    exit 1
fi

# Call the Python script to extract the relevant configuration
config=$(python3 tests/scripts/unit_tests/parse_config.py "$config_file" "$backend" "$subset")
echo "config:" $config

# Split the Python output and set default values
set_environment=$(echo $config | cut -d '|' -f 1)
root=$(echo $config | cut -d '|' -f 2);root=${root:-"tests/unit_tests"}
coverage_fold=$(echo $config | cut -d '|' -f 3)
type=$(echo $config | cut -d '|' -f 4);type=${type:-"batch"}
depth=$(echo $config | cut -d '|' -f 5);depth=${depth:-"all"}
ignore=$(echo $config | cut -d '|' -f 6)
deselect=$(echo $config | cut -d '|' -f 7)
id=${id:-0}

# Set the test path based on root and subset
if [[ "$subset" =~ [./]$ ]]; then
    subset="${subset##*[./]}"
fi
path="${root}/${subset}"
path="${path%/}"

# Output the test name
echo "Running test: $backend -> ${subset:-./}"

# Execute the set_environment commands
eval "$set_environment"

# Function to check if both reports are complete
check_reports_complete() {
    local xml_report="$1"
    local html_report="$2"

    local xml_previous_size=0
    local html_previous_size=0
    local xml_current_size
    local html_current_size

    while true; do
        xml_current_size=$(stat --format=%s "$xml_report" 2>/dev/null || echo 0)
        html_current_size=$(stat --format=%s "$html_report" 2>/dev/null || echo 0)

        # Check if sizes are stable
        if [ "$xml_previous_size" -eq "$xml_current_size" ] && [ "$html_previous_size" -eq "$html_current_size" ]; then
            echo "Reports are complete: $xml_report $html_report"
            return  # Exit the function normally if reports are complete
        fi

        xml_previous_size=$xml_current_size
        html_previous_size=$html_current_size
        sleep 5s
    done

    # If the loop exits without returning, it means reports are not stable
    echo "Check reports failed: $xml_report $html_report"
    exit 1  # Exit with error code if checks fail
}

# Function to execute a command and handle failures
run_command() {
    echo $(which conda) $(which python)
    wait_for_gpu
    echo "$1"
    eval "$1"
    # Check the exit status of pytest
    if [ $? -ne 0 ]; then
        echo "Failed: $1"
        exit 1
    fi
}

# Function to run tests at a specific depth
run_tests() {
    local _type="$1"
    local _path="$2"
    local _depth="$3"
    local _ignore="$4"
    local _deselect="$5"

    local _test_files="$_path"
    if ! ([ "$_depth" = "all" ] && [ "$_type" = "batch" ]); then
        if [ "$_depth" = "all" ]; then
            _depth=$(find "$_path" -type d | awk -F/ '{print NF-1}' | sort -nr | head -n 1)
        fi
        _test_files=$(find "$_path" -mindepth 1 -maxdepth $_depth -type f -name "test_*.py" | sort)
    fi

    # Process the raw ignore into bash-friendly --ignore parameters
    ignore_cmd=""
    if [ -n "$_ignore" ]; then
        # Handle the entire string as a list of files
        for item in $_ignore; do
            # Remove the leading '-' from each item if it exists
            _clean_item=${item#-}
            _clean_item=$(echo "$_clean_item" | tr -d "',[]")
            ignore_cmd+="--ignore=${path}/${_clean_item} "
            _test_files=$(echo "$_test_files" | grep -v "${_path}/${_clean_item}")
        done
    fi

    _test_files=$(echo "$_test_files" | tr '\n' ' ')

    # Process the raw deselect into bash-friendly --deselect parameters
    deselect_cmd=""
    if [ -n "$_deselect" ]; then
        # Handle the entire string as a list of files
        for item in $_deselect; do
            # Remove the leading '-' from each item if it exists
            _clean_item=${item#-}
            _clean_item=$(echo "$_clean_item" | tr -d "',[]")
            deselect_cmd+="--deselect=${path}/${_clean_item} "
        done
    fi

    local html_report="/workspace/report/$id/cov-report-${backend}"
    local xml_report="$html_report/coverage.xml"
    export COMMIT_ID=$id

    # Set coverage_parameters based on the value of coverage
    if [ "$coverage" == "False" ]; then
        coverage_parameters=""  # If coverage is False, set coverage_parameters to an empty string
    else
        # If coverage is not False, set coverage_parameters to include coverage-related flags
        coverage_parameters="--cov=${backend}/${coverage_fold} --cov-append --cov-report=xml:$xml_report --cov-report=html:$html_report"
        check_reports="check_reports_complete $xml_report $html_report"
    fi

    if [ "$_type" == "batch" ]; then
        run_command "torchrun --nproc_per_node=8 -m pytest $coverage_parameters -q -x -p no:warnings $ignore_cmd $deselect_cmd $_test_files"
        eval $check_reports
    elif [ "$_type" == "single" ]; then
        for _test_file in $_test_files; do
            run_command "torchrun --nproc_per_node=8 -m pytest $coverage_parameters -q -x -p no:warnings $ignore_cmd $deselect_cmd $_test_file"
            eval $check_reports
        done
    fi
}

# Run tests based on type, path, and depth
run_tests "$type" "$path" "$depth" "$ignore" "$deselect"

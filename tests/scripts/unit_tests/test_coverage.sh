#!/bin/bash

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backend) backend="$2"; shift ;;
        --id) id="$2"; shift ;;
        --status) status="$2"; shift ;;  # Accept --status from command line
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure necessary arguments are provided
if [ -z "$backend" ] || [ -z "$status" ]; then
    echo "Usage: $0 --backend <backend> --status <status> [--id <id>]"
    exit 1
fi

# Set the default value of id to 0
if [ -z "$id" ]; then
    id=0
fi

# Validate status input
if [ "$status" != "online" ] && [ "$status" != "offline" ]; then
    echo "Invalid status: $status. Must be 'online' or 'offline'."
    exit 1
fi

# Define the coverage report directory and the coverage XML report file
report_dir="/workspace/report/$id/cov-report-${backend}"
coverage_file="coverage.xml"

# Check if the coverage XML file exists
if [ ! -f "$report_dir/$coverage_file" ]; then
    echo "Error: Coverage file $report_dir/$coverage_file not found!"
    exit 1
fi

# Add the current working directory to the list of safe directories in Git
git config --global --add safe.directory /__w/FlagScale/FlagScale

# Check if the upstream remote already exists or add it
if git remote get-url upstream > /dev/null 2>&1; then
    echo "Upstream remote already exists."
else
    git remote add upstream https://github.com/FlagOpen/FlagScale.git
fi

# Fetch behavior based on status
if [ "$status" == "online" ]; then
    git fetch --unshallow upstream main
elif [ "$status" == "offline" ]; then
    git fetch upstream main
else
    echo "Unknown status: $status"
    exit 1
fi

# Get the latest common ancestor between the current branch and upstream main
common_ancestor=$(git merge-base HEAD upstream/main)

# Check if a common ancestor was found
if [ -z "$common_ancestor" ]; then
    echo "Error: No common ancestor found between the current branch and upstream main."
    echo "Ensure that your branch has a common history with upstream/main."
    exit 1
fi

# Get the changed files between the current branch and the common ancestor
git diff --name-only $common_ancestor HEAD > changed_files.txt

# If no changes detected
if [ ! -s changed_files.txt ]; then
    echo "No changes detected between $common_ancestor and HEAD."
    exit 1
fi

# Check the coverage for the new code changes
echo "Checking coverage for the new code changes..."
echo "diff-cover "$report_dir/$coverage_file" --compare-branch=$common_ancestor --html-report "$report_dir/diff-cover-report-${backend}.html" --fail-under=70"
diff-cover "$report_dir/$coverage_file" --compare-branch=$common_ancestor --html-report "$report_dir/diff-cover-report-${backend}.html" --fail-under=70

# If diff-cover exits with a non-zero status, it means the coverage is below 70%
if [ $? -ne 0 ]; then
    echo "Test coverage for new code is below 70% in the $backend. Please add more tests."
    exit 1
else
    echo "Test coverage for new code meets the 70% requirement in the $backend."
fi

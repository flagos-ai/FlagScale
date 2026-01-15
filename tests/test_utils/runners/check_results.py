import json
import os
import re

import numpy as np
import pytest


def find_directory(start_path, target_dir_name):
    """Recursively find directory by name."""
    for root, dirs, _ in os.walk(start_path):
        if target_dir_name in dirs:
            return os.path.join(root, target_dir_name)
    return None


def load_log_file(log_path):
    """Load log file with existence check."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    with open(log_path, "r") as f:
        return f.readlines()


def load_gold_file(gold_path):
    """Load gold result file."""
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    with open(gold_path, "r") as f:
        return json.load(f) if gold_path.endswith(".json") else f.readlines()


def extract_metrics_from_log(lines, metric_keys=None):
    """
    Extract metrics from training log lines.

    Log format (pipe-separated):
        " [2026-01-15 09:13:30] iteration 4/10 | ... | lm loss: 1.161108E+01 | ... |"

    Args:
        lines: List of log lines
        metric_keys: List of metric keys to extract (e.g., ["lm loss:"])
                    If None, defaults to ["lm loss:"]

    Returns:
        Dict with metric keys and their values list
    """
    if metric_keys is None:
        metric_keys = ["lm loss:"]

    results = {key: {"values": []} for key in metric_keys}

    for line in lines:
        # Skip non-iteration lines
        if "iteration" not in line:
            continue

        # Split by | and extract key-value pairs
        parts = line.split("|")
        for part in parts:
            part = part.strip()
            for key in metric_keys:
                # Match "lm loss: 1.161108E+01" format
                if part.startswith(key.rstrip(":")):
                    # Extract the value after the colon
                    match = re.search(r":\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", part)
                    if match:
                        try:
                            value = float(match.group(1))
                            results[key]["values"].append(value)
                        except ValueError:
                            continue

    return results


def find_latest_stdout_log(start_path):
    """
    Find the latest stdout.log file in the latest attempt directory.

    Directory structure:
        start_path/
            20260115_091249.893239/      # timestamp folders
                default_k2duk4a0/        # run folders
                    attempt_0/           # attempt folders
                        7/               # rank folders
                            stdout.log

    Finds the latest timestamp folder, then the latest attempt_x folder,
    then the latest rank folder containing stdout.log.
    """
    if not os.path.exists(start_path):
        return None, None

    # Step 1: Find all folders containing attempt_* directories
    folders_with_attempts = []
    for root, dirs, _ in os.walk(start_path):
        attempt_dirs = [d for d in dirs if d.startswith("attempt_")]
        if attempt_dirs:
            folders_with_attempts.append(root)

    if not folders_with_attempts:
        return None, None

    # Step 2: Sort by path (which includes timestamp) and get the latest
    folders_with_attempts.sort(reverse=True)
    latest_folder = folders_with_attempts[0]

    # Step 3: Find the latest attempt_x folder
    attempt_dirs = [d for d in os.listdir(latest_folder) if d.startswith("attempt_")]
    if not attempt_dirs:
        return None, None

    # Sort attempt directories numerically (attempt_0, attempt_1, ...)
    attempt_dirs.sort(
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else -1, reverse=True
    )
    latest_attempt = os.path.join(latest_folder, attempt_dirs[0])

    # Step 4: Find the latest rank directory with stdout.log
    try:
        rank_dirs = os.listdir(latest_attempt)
        # Sort numerically if possible
        rank_dirs.sort(key=lambda x: int(x) if x.isdigit() else float("inf"), reverse=True)

        for rank_dir in rank_dirs:
            log_path = os.path.join(latest_attempt, rank_dir, "stdout.log")
            if os.path.exists(log_path):
                return log_path, latest_attempt
    except OSError:
        pass

    return None, latest_attempt


@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_train_equal(test_path, test_type, test_task, test_case):
    """
    Compare training metrics from test run against gold values.

    This test extracts loss metrics from stdout.log and compares them
    against pre-recorded gold values using numpy.allclose for tolerance.
    """
    # Construct the test_result_path using the provided fixtures
    test_result_path = os.path.join(test_path, test_type, test_task, "test_results", test_case)
    start_path = os.path.join(test_result_path, "logs/details/host_0_localhost")

    # Find the latest stdout.log
    result_path, attempt_path = find_latest_stdout_log(start_path)

    assert attempt_path is not None, f"Failed to find any 'attempt_*' directory in {start_path}"
    assert result_path is not None, f"Failed to find 'stdout.log' in {attempt_path}"

    print(f"result_path: {result_path}")

    with open(result_path, "r") as file:
        lines = file.readlines()

    # Load gold values first to determine which metrics to extract
    gold_value_path = os.path.join(
        test_path, test_type, test_task, "test_results", test_case + ".json"
    )
    assert os.path.exists(gold_value_path), f"Failed to find gold result JSON at {gold_value_path}"

    with open(gold_value_path, "r") as f:
        gold_result_json = json.load(f)

    # Extract the metric keys from gold values
    metric_keys = list(gold_result_json.keys())

    # Extract metrics from log
    result_json = extract_metrics_from_log(lines, metric_keys)

    print("\nResult checking")
    print(f"Metric keys: {metric_keys}")
    print(f"Result: {result_json}")
    print(f"Gold Result: {gold_result_json}")

    # Compare each metric
    all_passed = True
    for key in metric_keys:
        result_values = result_json.get(key, {}).get("values", [])
        gold_values = gold_result_json.get(key, {}).get("values", [])

        if len(result_values) == 0:
            print(f"WARNING: No values extracted for metric '{key}'")
            all_passed = False
            continue

        if len(result_values) != len(gold_values):
            print(
                f"WARNING: Length mismatch for '{key}': got {len(result_values)}, expected {len(gold_values)}"
            )
            # Try to compare what we have
            min_len = min(len(result_values), len(gold_values))
            if min_len > 0:
                is_close = np.allclose(
                    gold_values[:min_len], result_values[:min_len], rtol=1e-3, atol=1e-5
                )
                print(f"Partial comparison ({min_len} values) for '{key}': {is_close}")
            all_passed = False
            continue

        is_close = np.allclose(gold_values, result_values, rtol=1e-3, atol=1e-5)
        print(f"Metric '{key}' comparison: {is_close}")

        if not is_close:
            # Print detailed diff for debugging
            diff = np.abs(np.array(gold_values) - np.array(result_values))
            print(f"  Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
            all_passed = False

    assert all_passed, "One or more metrics did not match gold values"

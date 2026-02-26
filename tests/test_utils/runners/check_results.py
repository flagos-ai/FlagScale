import json
import os
import re

import numpy as np
import pytest
import requests
from omegaconf import OmegaConf


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


@pytest.mark.usefixtures("path", "task", "model", "case")
def test_train_equal(path, task, model, case):
    """
    Compare training metrics from test run against gold values.

    This test extracts loss metrics from stdout.log and compares them
    against pre-recorded gold values using numpy.allclose for tolerance.
    """
    # Construct the test_result_path using the provided fixtures
    test_result_path = os.path.join(path, task, model, "test_results", case)
    start_path = os.path.join(test_result_path, "logs/details/host_0_localhost")

    # Find the latest stdout.log
    result_path, attempt_path = find_latest_stdout_log(start_path)

    assert attempt_path is not None, f"Failed to find any 'attempt_*' directory in {start_path}"
    assert result_path is not None, f"Failed to find 'stdout.log' in {attempt_path}"

    print(f"result_path: {result_path}")

    with open(result_path, "r") as file:
        lines = file.readlines()

    # Load gold values first to determine which metrics to extract
    gold_value_path = os.path.join(path, task, model, "gold_values", case + ".json")
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
    print(f"\n{'=' * 70}")
    print("DETAILED COMPARISON REPORT")
    print(f"{'=' * 70}")

    for key in metric_keys:
        result_values = result_json.get(key, {}).get("values", [])
        gold_values = gold_result_json.get(key, {}).get("values", [])

        print(f"\n{'=' * 70}")
        print(f"Metric: {key}")
        print(f"{'=' * 70}")
        print(f"GOLDEN VALUES ({len(gold_values)} values):")
        print(f"  {gold_values}")
        print(f"\nACTUAL VALUES ({len(result_values)} values):")
        print(f"  {result_values}")

        if len(result_values) == 0:
            print(f"❌ WARNING: No values extracted for metric '{key}'")
            all_passed = False
            continue

        if len(result_values) != len(gold_values):
            print(
                f"\n⚠️  WARNING: Length mismatch for '{key}': got {len(result_values)}, expected {len(gold_values)}"
            )
            # Try to compare what we have
            min_len = min(len(result_values), len(gold_values))
            if min_len > 0:
                is_close = np.allclose(gold_values[:min_len], result_values[:min_len])
                diff = np.abs(np.array(gold_values[:min_len]) - np.array(result_values[:min_len]))
                print(f"\nPartial comparison (first {min_len} values):")
                print(f"  Status: {'✅ PASS' if is_close else '❌ FAIL'}")
                print(f"  Max diff: {np.max(diff):.6e}")
                print(f"  Mean diff: {np.mean(diff):.6e}")
            all_passed = False
            continue

        # Calculate differences
        diff = np.abs(np.array(gold_values) - np.array(result_values))
        is_close = np.allclose(gold_values, result_values)

        print(f"\nComparison result: {'✅ PASS' if is_close else '❌ FAIL'}")
        print(f"  Max diff: {np.max(diff):.6e}")
        print(f"  Mean diff: {np.mean(diff):.6e}")

        if not is_close:
            all_passed = False

    print(f"\n{'=' * 70}")
    print(f"Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"{'=' * 70}\n")

    assert all_passed, "One or more metrics did not match gold values"


@pytest.mark.usefixtures("path", "task", "model", "case")
def test_inference_equal(path, task, model, case):
    """
    Verify the consistency between inference output results and golden reference results.

    Core functions of this test:
    1. Locate the inference output log file generated by the test run
    2. Extract inference output content within the specified marker range
    3. Load the pre-recorded golden reference result file
    4. Clean up trailing blank lines in the golden reference results to ensure comparison accuracy
    5. Compare the actual output content with the golden reference content line by line

    Parameter description for fixtures:
    - path: Base directory path
    - task: Task name
    - model: Model name
    - case: Test case name

    Exception checks:
    - Check for 'flag_gems' import failure error messages
    - Verify the existence of output files and golden reference files
    - Verify the consistency between the number of actual output lines and golden reference lines
    - Perform exact line-by-line content matching (ignoring newline character differences)
    """
    # Construct test result path: concatenate base path, task, model, test result directory and case name
    test_result_path = os.path.join(path, task, model, "results_test", case)
    # Locate the inference output log file (host_0_localhost.output)
    result_path = os.path.join(test_result_path, "inference_logs/host_0_localhost.output")

    print("result_path:", result_path)

    # Assertion check: ensure the inference output file exists
    assert os.path.exists(result_path), f"Failed to find 'host_0_localhost.output' at {result_path}"

    with open(result_path, "r") as file:
        lines = file.readlines()

    # Extract inference output content within the marker range
    result_lines = []
    output = False  # Flag to indicate whether to start collecting output content
    for line in lines:
        # Assertion check: ensure no 'flag_gems' import failure errors exist
        assert "Failed to import 'flag_gems'" not in line, "Failed to import 'flag_gems''"

        if line == "**************************************************\n":
            output = True
        if line == "##################################################\n":
            output = False
        if output:
            result_lines.append(line)

    # Construct the path to the golden reference result file
    gold_value_path = os.path.join(path, task, model, "results_gold", case)
    assert os.path.exists(gold_value_path), f"Failed to find gold result at {gold_value_path}"

    with open(gold_value_path, "r") as file:
        gold_value_lines = file.readlines()

    # Clean up trailing blank lines in the golden reference results: improve comparison robustness
    if gold_value_lines:
        # Find the index of the last non-blank line from the end
        last_non_empty = len(gold_value_lines) - 1
        while last_non_empty >= 0 and not gold_value_lines[last_non_empty].strip():
            last_non_empty -= 1
        # Truncate to the last non-blank line (inclusive)
        if last_non_empty >= 0:
            gold_value_lines = gold_value_lines[: last_non_empty + 1]
        else:
            gold_value_lines = []

    print("\nResult checking")
    print("Result: ", result_lines)
    print("Gold Result: ", gold_value_lines)
    print("len(result_lines), (gold_value_lines): ", len(result_lines), len(gold_value_lines))

    assert len(result_lines) == len(gold_value_lines)

    # Compare actual output and golden reference output line by line (ignoring newline character differences)
    for result_line, gold_value_line in zip(result_lines, gold_value_lines):
        print(result_line, gold_value_line)
        assert result_line.rstrip("\n") == gold_value_line.rstrip("\n")


@pytest.mark.usefixtures("path", "task", "model", "case")
def test_serve_equal(path, task, model, case):
    """
    Verify the output consistency of model serving deployment.

    This test case is mainly used to check the interface availability and basic response capability
    after the model is deployed as a service. It validates interface calls separately for two deployment modes:
    1. Composite Deployment Mode (enable_composition=True): Call the basic HTTP interface to verify non-empty responses.
    2. Standalone VLLM Deployment Mode: Call the OpenAI-compatible /completions interface to verify response status and content.

    Parameter Description (injected via pytest fixtures):
    - path: Base directory path
    - task: Task name (e.g., inference/training)
    - model: Model name
    - case: Test case name

    Exception Checks:
    - Configuration file loading failure
    - HTTP request returns non-200 status code
    - Response content is empty or does not meet the minimum length requirement
    - Missing required configuration items (e.g., port, model name)
    """
    # Construct the path to the configuration file corresponding to the test case
    config_path = os.path.join(path, task, model, "conf", f"{case}.yaml")
    print("[Serve] config_path ", config_path)

    # Load the original configuration file and locate the full service deployment configuration file
    with open(config_path, "r") as f:
        origin_config = OmegaConf.load(f)
        whole_config_path = os.path.join(
            origin_config["experiment"]["exp_dir"], "serve_logs/scripts/serve.yaml"
        )
        whole_config = OmegaConf.load(whole_config_path)

    print("[Serve] whole_config ", whole_config)

    # Extract deployment-related configuration items
    deploy_config = whole_config.experiment.runner.deploy

    # Scenario 1: Composite Deployment Mode (enable_composition=True)
    if deploy_config.get("enable_composition", False):
        # Concatenate the service invocation URL (port + service name)
        url = f"http://localhost:{deploy_config.port}" + deploy_config.get("name", "/")
        response = requests.post(url, json={"prompt": "Introduce Bruce Lee"})
        greeting = response.text
        print("[Serve] result ", greeting)
        assert len(greeting) > 5, "Response is empty."

    # Scenario 2: Non-composite Deployment Mode (mainly for VLLM engine deployment)
    else:
        # Extract the service configuration list
        serve_config = whole_config.get("serve", [])
        if not serve_config:
            raise ValueError(
                f"No 'serve_config' configuration found in task config: {whole_config}"
            )

        # Get the first service configuration item (default single-service deployment)
        serve_config = serve_config[0]

        # Check if the deployment uses the VLLM engine
        if serve_config.get("engine", "vllm") == "vllm":
            # Extract VLLM engine-related parameters
            engine_args = serve_config.get("engine_args", {})
            # Get the deployed model name (prioritize served_model_name, then model)
            model_name = engine_args.get("served_model_name", None) or engine_args.get(
                "model", None
            )
            if not model_name:
                raise ValueError(
                    f"Missing 'served_model_name' or 'model' argument in 'engine_args': {engine_args}"
                )

            # Concatenate the VLLM OpenAI-compatible interface URL
            url = f"http://localhost:{deploy_config.port}/v1/completions"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model_name,
                "prompt": "Introduce Bruce Lee in details",
                "max_tokens": 20,
                "temperature": 0,
                "stream": False,
            }
            response = requests.post(url, headers=headers, data=json.dumps(data))
            assert response.status_code == 200, "Request failed with status code: {}".format(
                response.status_code
            )
            print("[Serve] result ", response.json())


@pytest.mark.usefixtures("path", "task", "model", "case")
def test_rl_equal(path, task, model, case):
    """
    Verify the consistency of reward metrics during reinforcement learning (RL) training.

    Core functionalities of this test case:
    1. Locate and read the RL training output log file
    2. Extract the specified reward metric (val-core/openai/gsm8k/reward/mean@1) from logs
    3. Load pre-recorded golden reference metric data
    4. Compare the consistency between actual metrics and golden metrics using numpy.allclose
       with an absolute tolerance of 0.05

    Parameter Description (injected via pytest fixtures):
    - path: Base directory path
    - task: Task name (e.g., rl)
    - model: Model name
    - case: Test case name

    Core Verification Metric:
    - val-core/openai/gsm8k/reward/mean@1: Mean reward value on the GSM8K dataset

    Tolerance Setting:
    - Absolute tolerance (atol) = 0.05: Allows a maximum absolute error of 0.05
      between actual and golden values
    """
    # 1. Construct file paths and validate file existence
    test_result_path = os.path.join(path, task, model, "results_test", case)
    result_path = os.path.join(test_result_path, "logs/host_0_localhost.output")
    print("result_path:", result_path)

    assert os.path.exists(result_path), f"Failed to find 'host_0_localhost.output' at {result_path}"

    # 2. Read log file content
    with open(result_path, "r") as file:
        lines = file.readlines()

    # 3. Initialize result storage dictionary for extracted reward metrics
    result_json = {}
    result_json["val-core/openai/gsm8k/reward/mean@1"] = []

    # 4. Parse log file and extract the specified reward metric
    for line in lines:
        if "step" in line:
            line_split = line.strip().split(" ")
            for key_value in line_split:
                if key_value.startswith("val-core/openai/gsm8k/reward/mean"):
                    value = key_value.split(":")[-1]
                    result_json["val-core/openai/gsm8k/reward/mean@1"].append(float(value))

    # 5. Load golden reference metric data
    gold_value_path = os.path.join(path, task, model, "results_gold", case + ".json")
    assert os.path.exists(gold_value_path), f"Failed to find gold result JSON at {gold_value_path}"
    with open(gold_value_path, "r") as f:
        gold_result_json = json.load(f)

    # 6. Print debugging information for troubleshooting
    print("\nResult checking")
    print("Result: ", result_json)
    print("Gold Result: ", gold_result_json)
    print(
        "The results are basically equal: ",
        np.allclose(
            gold_result_json["val-core/openai/gsm8k/reward/mean@1"],
            result_json["val-core/openai/gsm8k/reward/mean@1"],
            atol=0.05,
        ),
    )

    # 7. Core assertion: Verify numerical consistency between actual and golden metrics
    assert np.allclose(
        gold_result_json["val-core/openai/gsm8k/reward/mean@1"],
        result_json["val-core/openai/gsm8k/reward/mean@1"],
        atol=0.05,
    ), "Result not close to gold result"

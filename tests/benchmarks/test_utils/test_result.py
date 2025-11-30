import json
import os
import re

import numpy as np
import pytest
import requests

from omegaconf import DictConfig, OmegaConf


def find_directory(start_path, target_dir_name):
    for root, dirs, files in os.walk(start_path):
        if target_dir_name in dirs:
            return os.path.join(root, target_dir_name)
    return None


@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_train_equal(test_path, test_type, test_task, test_case):
    test_result_path = os.path.join(test_path, test_type, test_task, "results_test", test_case)
    start_path = os.path.join(test_result_path, "logs/details/host_0_localhost")
    attempt_path = find_directory(start_path, "attempt_0")
    assert attempt_path is not None, f"Failed to find 'attempt_0' directory in {start_path}"

    results_path = os.listdir(attempt_path)
    results_path.sort()
    result_path = os.path.join(attempt_path, results_path[-1], "stdout.log")
    assert os.path.exists(result_path), f"Failed to find 'stdout.log' at {result_path}"

    tokens_sec_list = []
    iteration_time_list = []

    with open(result_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if " iteration" in line:
            match_time = re.search(r"elapsed time per iteration \(ms\):\s*([\d.]+)", line)
            match_batch = re.search(r"global batch size:\s*(\d+)", line)
            if match_time and match_batch:
                t_ms = float(match_time.group(1))
                b = int(match_batch.group(1))
                iteration_time_list.append(t_ms)
                tokens_sec_list.append(b / (t_ms / 1000.0))  # tokens/sec

    def remove_outliers(data, lower_q=10, upper_q=90):
        if len(data) < 3:
            return data
        low, high = np.percentile(data, [lower_q, upper_q])
        return [x for x in data if low <= x <= high]

    filtered_tokens = remove_outliers(tokens_sec_list)
    filtered_times = remove_outliers(iteration_time_list)

    median_tokens_sec = float(np.median(filtered_tokens))
    median_iter_time = float(np.median(filtered_times))

    result_json = {
        "tokens/sec": {"values": filtered_tokens, "median": median_tokens_sec},
        "iteration_time_ms": {"values": filtered_times, "median": median_iter_time},
    }

    gold_value_path = os.path.join(
        test_path, test_type, test_task, "results_gold", test_case + ".json"
    )
    assert os.path.exists(gold_value_path), f"Failed to find gold result JSON at {gold_value_path}"

    with open(gold_value_path, "r") as f:
        gold_result_json = json.load(f)

    gold_median_tokens = np.median(remove_outliers(gold_result_json["tokens/sec"]["values"]))
    gold_median_time = np.median(remove_outliers(gold_result_json["iteration_time_ms"]["values"]))

    print("\n===== Performance Metrics Checking (Median-based) =====")
    print(f"Median tokens/sec (Result): {median_tokens_sec:.2f}")
    print(f"Median tokens/sec (Gold):   {gold_median_tokens:.2f}")
    print(f"Median iteration time (Result): {median_iter_time:.2f} ms")
    print(f"Median iteration time (Gold):   {gold_median_time:.2f} ms")

    perf_threshold = 0.9
    time_threshold = 1.1
    assert (
        median_tokens_sec >= gold_median_tokens * perf_threshold
    ), f"Throughput dropped below {perf_threshold*100:.0f}% of gold baseline!"

    assert (
        median_iter_time <= gold_median_time * time_threshold
    ), f"Iteration time slower than {time_threshold:.1f}x of gold baseline!"


@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_inference_equal(test_path, test_type, test_task, test_case):
    # Paths
    test_result_path = os.path.join(test_path, test_type, test_task, "results_test", test_case)
    result_path = os.path.join(test_result_path, "inference_logs/host_0_localhost.output")
    assert os.path.exists(result_path), f"Failed to find result at {result_path}"

    gold_value_path = os.path.join(test_path, test_type, test_task, "results_gold", test_case)
    assert os.path.exists(gold_value_path), f"Failed to find gold JSON at {gold_value_path}"

    # Read gold JSON
    with open(gold_value_path, "r") as f:
        gold_data = json.load(f)

    gold_throughput = gold_data["throughput"]
    gold_avg_latency = gold_data["avg_latency"]

    # Read result file and extract numbers
    with open(result_path, "r") as f:
        text = f.read()

    match_throughput = re.search(r"Throughput:\s*([\d.]+)", text)
    match_avg_latency = re.search(r"Avg latency:\s*([\d.]+)", text)

    assert match_throughput is not None, "Failed to extract result throughput"
    assert match_avg_latency is not None, "Failed to extract result avg latency"

    result_throughput = float(match_throughput.group(1))
    result_avg_latency = float(match_avg_latency.group(1))

    # Allow 10% floating
    throughput_lower_bound = 0.9 * gold_throughput
    latency_upper_bound = 1.1 * gold_avg_latency

    print(
        f"Result Throughput: {result_throughput}, Gold: {gold_throughput}, Lower bound: {throughput_lower_bound}"
    )
    print(
        f"Result Avg Latency: {result_avg_latency}, Gold: {gold_avg_latency}, Upper bound: {latency_upper_bound}"
    )

    # Performance assertions
    assert (
        result_throughput >= throughput_lower_bound
    ), f"Throughput too low: {result_throughput} < {throughput_lower_bound} (10% below gold)"
    assert (
        result_avg_latency <= latency_upper_bound
    ), f"Avg latency too high: {result_avg_latency} > {latency_upper_bound} (10% above gold)"

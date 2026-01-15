import json
import os
import re

import numpy as np
import pytest
import requests
from omegaconf import OmegaConf


def find_directory(start_path, target_dir_name):
    for root, dirs, files in os.walk(start_path):
        if target_dir_name in dirs:
            return os.path.join(root, target_dir_name)
    return None


@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_train_equal(test_path, test_type, test_task, test_case):
    # Construct the test_result_path using the provided fixtures
    test_result_path = os.path.join(test_path, test_type, test_task, "results_test", test_case)
    start_path = os.path.join(test_result_path, "logs/details/host_0_localhost")

    attempt_path = find_directory(start_path, "attempt_0")
    assert attempt_path is not None, f"Failed to find 'attempt_0' directory in {start_path}"

    results_path = os.listdir(attempt_path)
    results_path.sort()
    result_path = os.path.join(attempt_path, results_path[-1], "stdout.log")

    print("result_path:", result_path)

    assert os.path.exists(result_path), f"Failed to find 'stdout.log' at {result_path}"

    with open(result_path, "r") as file:
        lines = file.readlines()

    result_json = {}
    result_json["lm loss:"] = {"values": []}

    for line in lines:
        if " iteration" in line:
            line_split = line.strip().split("|")
            for key_value in line_split:
                if key_value.startswith(" lm loss:"):
                    result_json["lm loss:"]["values"].append(float(key_value.split(":")[1]))

    gold_value_path = os.path.join(
        test_path, test_type, test_task, "results_gold", test_case + ".json"
    )
    assert os.path.exists(gold_value_path), f"Failed to find gold result JSON at {gold_value_path}"

    with open(gold_value_path, "r") as f:
        gold_result_json = json.load(f)

    print("\nResult checking")
    print("Result: ", result_json)
    print("Gold Result: ", gold_result_json)
    print(
        "The results are basically equal: ",
        np.allclose(gold_result_json["lm loss:"]["values"], result_json["lm loss:"]["values"]),
    )

    assert np.allclose(gold_result_json["lm loss:"]["values"], result_json["lm loss:"]["values"]), (
        "Result not close to gold result"
    )


@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_inference_equal(test_path, test_type, test_task, test_case):
    # Construct the test_result_path using the provided fixtures
    test_result_path = os.path.join(test_path, test_type, test_task, "results_test", test_case)
    result_path = os.path.join(test_result_path, "inference_logs/host_0_localhost.output")

    print("result_path:", result_path)

    assert os.path.exists(result_path), f"Failed to find 'host_0_localhost.output' at {result_path}"

    with open(result_path, "r") as file:
        lines = file.readlines()

    result_lines = []
    output = False
    for line in lines:
        assert "Failed to import 'flag_gems'" not in line, "Failed to import 'flag_gems''"
        if line == "**************************************************\n":
            output = True
        if line == "##################################################\n":
            output = False
        if output:
            result_lines.append(line)

    gold_value_path = os.path.join(test_path, test_type, test_task, "results_gold", test_case)
    assert os.path.exists(gold_value_path), f"Failed to find gold result at {gold_value_path}"

    with open(gold_value_path, "r") as file:
        gold_value_lines = file.readlines()

    # Remove the blank line at the end.
    if gold_value_lines:
        last_non_empty = len(gold_value_lines) - 1
        while last_non_empty >= 0 and not gold_value_lines[last_non_empty].strip():
            last_non_empty -= 1
        if last_non_empty >= 0:
            gold_value_lines = gold_value_lines[: last_non_empty + 1]
        else:
            gold_value_lines = []

    print("\nResult checking")
    print("Result: ", result_lines)
    print("Gold Result: ", gold_value_lines)

    print("len(result_lines), (gold_value_lines): ", len(result_lines), len(gold_value_lines))
    assert len(result_lines) == len(gold_value_lines)

    for result_line, gold_value_line in zip(result_lines, gold_value_lines):
        print(result_line, gold_value_line)
        assert result_line.rstrip("\n") == gold_value_line.rstrip("\n")


@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_inference_pipeline(test_path, test_type, test_task, test_case):
    # Construct the test_result_path using the provided fixtures
    test_result_path = os.path.join(test_path, test_type, test_task, "results_test", test_case)
    result_path = os.path.join(test_result_path, "inference_logs/host_0_localhost.output")

    print("result_path:", result_path)

    assert os.path.exists(result_path), f"Failed to find 'host_0_localhost.output' at {result_path}"

    with open(result_path, "r") as file:
        lines = file.readlines()

    result_lines = []
    output = False
    for line in lines:
        assert "Failed to import 'flag_gems'" not in line, "Failed to import 'flag_gems''"
        if line == "**************************************************\n":
            output = True
        if line == "##################################################\n":
            output = False
        if output:
            result_lines.append(line)

    gold_value_path = os.path.join(test_path, test_type, test_task, "results_gold", test_case)
    assert os.path.exists(gold_value_path), f"Failed to find gold result at {gold_value_path}"

    with open(gold_value_path, "r") as file:
        gold_value_lines = file.readlines()

    print("\nResult checking")
    print("Result: ", result_lines)
    print("Gold Result: ", gold_value_lines)

    print("len(result_lines), (gold_value_lines): ", len(result_lines), len(gold_value_lines))
    assert len(result_lines) == len(gold_value_lines)

    # Compare in groups of 4
    for i in range(0, len(result_lines), 4):
        # Get the next 4 lines for both result and gold values
        result_group = result_lines[i : i + 4]
        gold_group = gold_value_lines[i : i + 4]

        # Check the first line for strict equality
        assert result_group[0] == gold_group[0], (
            f"First line mismatch:\nResult: {result_group[0]}\nGold: {gold_group[0]}"
        )

        # Check the next three lines for equality before the '=' character
        for j in range(1, 4):
            result_parts = result_group[j].split("=")
            gold_parts = gold_group[j].split("=")

            # Check if both lines have an '=' and compare the keys (before '=')
            if len(result_parts) == 2 and len(gold_parts) == 2:
                result_key = result_parts[0].strip()
                gold_key = gold_parts[0].strip()

                assert result_key == gold_key, (
                    f"Line {j + 1} keys mismatch:\nResult: {result_group[j]}\nGold: {gold_group[j]}"
                )
            else:
                print(
                    f"Warning: One of the lines doesn't have an '=' character: \nResult: {result_group[j]}\nGold: {gold_group[j]}"
                )

        result_parts = result_group[1].split("=")
        assert len(result_parts) >= 2, "len(result_parts) should be 2"
        result_value = result_parts[1].strip()
        assert result_value[0] == "'" and result_value[-1] == "'" and result_value[1:-1], (
            f"String {result_value} should be wrapped in '' and not empty inside"
        )

        result_parts = result_group[2].split("=")
        assert len(result_parts) >= 2, "len(result_parts) should be 2"
        result_value = result_parts[1].strip()
        assert result_value[0] == "'" and result_value[-1] == "'" and result_value[1:-1], (
            f"String {result_value} should be wrapped in '' and not empty inside"
        )

        result_parts = result_group[3].split("=")
        assert len(result_parts) >= 2, "len(result_parts) should be 2"
        result_value = result_parts[1].strip()
        assert (
            (result_value[0] == "[" and result_value[-1] == "]")
            or (result_value[0] == "(" and result_value[-1] == ")")
            and result_value[1:-1]
        ), f"String {result_value} should be wrapped in [] or () and not empty inside"


@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_rl_equal(test_path, test_type, test_task, test_case):
    # Construct the test_result_path using the provided fixtures
    test_result_path = os.path.join(test_path, test_type, test_task, "results_test", test_case)
    result_path = os.path.join(test_result_path, "logs/host_0_localhost.output")

    print("result_path:", result_path)

    assert os.path.exists(result_path), f"Failed to find 'host_0_localhost.output' at {result_path}"

    with open(result_path, "r") as file:
        lines = file.readlines()

    result_json = {}
    result_json["val-aux/openai/gsm8k/reward/mean@1"] = []

    for line in lines:
        if "step" in line:
            line_split = line.strip().split(" - ")
            for key_value in line_split:
                if key_value.startswith("val-aux/openai/gsm8k/reward/mean@1"):
                    # Extract the value after the colon
                    value = key_value.split(":")[-1]
                    # Handle np.float64(...) format by extracting the number inside parentheses
                    match = re.search(r"np\.float64\(([\d.eE+-]+)\)", value)
                    if match:
                        value = match.group(1)
                    # Convert to float and append to the list
                    result_json["val-aux/openai/gsm8k/reward/mean@1"].append(float(value))

    gold_value_path = os.path.join(
        test_path, test_type, test_task, "results_gold", test_case + ".json"
    )
    assert os.path.exists(gold_value_path), f"Failed to find gold result JSON at {gold_value_path}"

    with open(gold_value_path, "r") as f:
        gold_result_json = json.load(f)

    print("\nResult checking")
    print("Result: ", result_json)
    print("Gold Result: ", gold_result_json)
    print(
        "The results are basically equal: ",
        np.allclose(
            gold_result_json["val-aux/openai/gsm8k/reward/mean@1"],
            result_json["val-aux/openai/gsm8k/reward/mean@1"],
            atol=0.05,
        ),
    )

    assert np.allclose(
        gold_result_json["val-aux/openai/gsm8k/reward/mean@1"],
        result_json["val-aux/openai/gsm8k/reward/mean@1"],
        atol=0.05,
    ), "Result not close to gold result"


@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_serve_equal(test_path, test_type, test_task, test_case):
    config_path = os.path.join(test_path, test_type, test_task, "conf", f"{test_case}.yaml")
    print("[Serve] config_path ", config_path)

    with open(config_path, "r") as f:
        origin_config = OmegaConf.load(f)
        whole_config_path = os.path.join(
            origin_config["experiment"]["exp_dir"], "serve_logs/scripts/serve.yaml"
        )
        whole_config = OmegaConf.load(whole_config_path)
    print("[Serve] whole_config ", whole_config)
    deploy_config = whole_config.experiment.runner.deploy
    if deploy_config.get("enable_composition", False):
        url = f"http://localhost:{deploy_config.port}" + deploy_config.get("name", "/")

        response = requests.post(url, json={"prompt": "Introduce Bruce Lee"})
        greeting = response.text
        print("[Serve] result ", greeting)
        assert len(greeting) > 5, "Response is empty."
    else:
        serve_config = whole_config.get("serve", [])
        if not serve_config:
            raise ValueError(
                f"No 'serve_config' configuration found in task config: {whole_config}"
            )
        serve_config = serve_config[0]
        if serve_config.get("engine", "vllm") == "vllm":
            engine_args = serve_config.get("engine_args", {})
            model_name = engine_args.get("served_model_name", None) or engine_args.get(
                "model", None
            )
            if not model_name:
                raise ValueError(
                    f"Missing 'served_model_name' or 'model' argument in 'engine_args': {engine_args}"
                )
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

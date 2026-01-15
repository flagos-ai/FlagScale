#!/usr/bin/env python3
import argparse
import json
import os
import sys

import yaml


def load_yaml(path):
    """Load and parse YAML file, raise error if not found or invalid."""
    if not os.path.isfile(path):
        raise OSError(f"Configuration file not found: {path}")
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")


def get_platform_config(platform="default"):
    """Load platform configuration YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    platform_map = {"default": "default.yaml", "a100": "cuda.yaml"}
    yaml_file = platform_map.get(platform, f"{platform}.yaml")
    config_file = os.path.join(script_dir, "../config/platforms", yaml_file)

    if not os.path.exists(config_file):
        raise OSError(f"Platform config not found: {config_file}")
    return load_yaml(config_file)


def get_platform_data(config, platform="default"):
    """Extract platform-specific data from config."""
    platform_key_map = {"default": "generic", "a100": "a100"}
    platform_key = platform_key_map.get(platform, platform)

    if platform_key not in config:
        raise ValueError(
            f"Platform '{platform_key}' not found in config. Available: {list(config.keys())}"
        )
    return config[platform_key]


def get_unit_tests_config(platform="default"):
    """Get unit test patterns from platform configuration."""
    try:
        config = get_platform_config(platform)
        platform_data = get_platform_data(config, platform)
        unit_tests = platform_data.get("tests", {}).get("unit", {})
        return {"include": unit_tests.get("include", "*"), "exclude": unit_tests.get("exclude", [])}
    except Exception as e:
        print(f"Error getting unit test config: {e}", file=sys.stderr)
        return {"include": "*", "exclude": []}


def get_functional_tests(platform="default", task=None, model=None, test_list=None):
    """Get functional tests from platform config, optionally filtered by task/model/list."""
    config = get_platform_config(platform)
    platform_data = get_platform_data(config, platform)
    functional_tests = platform_data.get("tests", {}).get("functional", {})

    result = {}

    # If task specified, filter by task
    if task:
        if task not in functional_tests:
            raise ValueError(f"Task '{task}' not found. Available: {list(functional_tests.keys())}")
        task_data = functional_tests[task]

        # If model specified, filter by model
        if model:
            if model not in task_data:
                raise ValueError(
                    f"Model '{model}' not found in task '{task}'. Available: {list(task_data.keys())}"
                )
            model_tests = task_data[model]

            # If list specified, filter by specific test names
            if test_list:
                test_names = [t.strip() for t in test_list.split(",")]
                model_tests = [t for t in model_tests if t in test_names]
                if not model_tests:
                    raise ValueError(f"No matching tests found in list for {task}/{model}")

            result[task] = {model: model_tests}
        else:
            # No model specified, return all models in task
            result[task] = task_data
    else:
        # No task specified, return all
        result = functional_tests

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse test configuration with platform support")
    parser.add_argument("--platform", default="default", help="Platform type (default, a100, etc)")
    parser.add_argument("--type", choices=["unit", "functional"], help="Test type")
    parser.add_argument("--task", help="Functional task name (train, hetero_train)")
    parser.add_argument("--model", help="Model name (aquila, mixtral, etc)")
    parser.add_argument("--list", dest="test_list", help="Comma-separated list of test names")

    args = parser.parse_args()

    try:
        if args.type == "unit" or (not args.type and not args.task):
            # Get unit test patterns
            config = get_unit_tests_config(args.platform)
            print(json.dumps(config))
        else:
            # Get functional tests
            tests = get_functional_tests(args.platform, args.task, args.model, args.test_list)
            print(json.dumps(tests))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

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


def get_platform_config(platform="default", device=None):
    """Load platform configuration YAML file.

    Args:
        platform: Platform name (e.g., 'cuda', 'default')
        device: Device type within platform (e.g., 'a100', 'a800', 'h100')
                If None, device will be derived from platform name

    Returns:
        Tuple of (config_dict, device_type)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Map platform names to config files
    platform_file_map = {
        "default": "default.yaml",
        "cuda": "cuda.yaml",
        "a100": "cuda.yaml",
        "a800": "cuda.yaml",
        "h100": "cuda.yaml",
    }

    # If platform is a device type (a100, a800, h100) and no device specified
    if platform in ["a100", "a800", "h100"] and device is None:
        device = platform
        platform = "cuda"

    yaml_file = platform_file_map.get(platform, f"{platform}.yaml")
    config_file = os.path.join(script_dir, "../config/platforms", yaml_file)

    if not os.path.exists(config_file):
        raise OSError(f"Platform config not found: {config_file}")

    config = load_yaml(config_file)

    # If device not specified, use default device from config
    if device is None:
        if platform == "default":
            device = "generic"
        elif config.get("device_types"):
            # Use first device type as default
            device = config["device_types"][0]
        else:
            device = platform

    return config, device


def get_platform_data(config, device):
    """Extract device-specific data from config.

    Args:
        config: Full platform configuration dict
        device: Device type (e.g., 'a100', 'generic')

    Returns:
        Device-specific configuration dict
    """
    if device not in config:
        available = [k for k in config.keys() if k != "device_types"]
        raise ValueError(f"Device '{device}' not found in config. Available devices: {available}")
    return config[device]


def get_device_types(platform="default"):
    """Get available device types for a platform.

    Args:
        platform: Platform name (e.g., 'cuda', 'default')

    Returns:
        List of device types available for the platform
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    platform_file_map = {"default": "default.yaml", "cuda": "cuda.yaml"}

    yaml_file = platform_file_map.get(platform, f"{platform}.yaml")
    config_file = os.path.join(script_dir, "../config/platforms", yaml_file)

    if not os.path.exists(config_file):
        raise OSError(f"Platform config not found: {config_file}")

    config = load_yaml(config_file)

    # Return device_types if defined, otherwise return all top-level keys except device_types
    if "device_types" in config:
        return config["device_types"]
    else:
        return [k for k in config.keys() if k != "device_types"]


def get_unit_tests_config(platform="default", device=None):
    """Get unit test patterns from platform configuration.

    Args:
        platform: Platform name (e.g., 'cuda', 'default')
        device: Device type (e.g., 'a100', 'a800', 'h100')

    Returns:
        Dict with 'include' and 'exclude' patterns
    """
    try:
        config, device = get_platform_config(platform, device)
        platform_data = get_platform_data(config, device)
        unit_tests = platform_data.get("tests", {}).get("unit", {})
        return {"include": unit_tests.get("include", "*"), "exclude": unit_tests.get("exclude", [])}
    except Exception as e:
        print(f"Error getting unit test config: {e}", file=sys.stderr)
        return {"include": "*", "exclude": []}


def get_functional_tests(platform="default", device=None, task=None, model=None, test_list=None):
    """Get functional tests from platform config, optionally filtered by task/model/list.

    Args:
        platform: Platform name (e.g., 'cuda', 'default')
        device: Device type (e.g., 'a100', 'a800', 'h100')
        task: Task name to filter (e.g., 'train', 'hetero_train')
        model: Model name to filter (e.g., 'aquila', 'mixtral')
        test_list: Comma-separated list of test names to filter

    Returns:
        Dict of functional tests configuration
    """
    config, device = get_platform_config(platform, device)
    platform_data = get_platform_data(config, device)
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
    parser = argparse.ArgumentParser(
        description="Parse test configuration with platform and device support"
    )
    parser.add_argument("--platform", default="default", help="Platform type (default, cuda, etc)")
    parser.add_argument("--device", help="Device type within platform (a100, a800, h100, etc)")
    parser.add_argument("--type", choices=["unit", "functional", "device_types"], help="Query type")
    parser.add_argument("--task", help="Functional task name (train, hetero_train)")
    parser.add_argument("--model", help="Model name (aquila, mixtral, etc)")
    parser.add_argument("--list", dest="test_list", help="Comma-separated list of test names")

    args = parser.parse_args()

    try:
        if args.type == "device_types":
            # Return available device types for the platform
            device_types = get_device_types(args.platform)
            print(json.dumps(device_types))
        elif args.type == "unit" or (not args.type and not args.task):
            # Get unit test patterns
            config = get_unit_tests_config(args.platform, args.device)
            print(json.dumps(config))
        else:
            # Get functional tests
            tests = get_functional_tests(
                args.platform, args.device, args.task, args.model, args.test_list
            )
            print(json.dumps(tests))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

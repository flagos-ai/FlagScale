#!/usr/bin/env python3
"""
Helper functions for FlagScale test runners.
Provides utilities for parsing test configurations and formatting output.
"""

import json
import sys


def extract_test_patterns(json_str: str) -> tuple[str, str]:
    """
    Extract include and exclude patterns from unit test configuration JSON.

    Args:
        json_str: JSON string containing 'include' and 'exclude' keys

    Returns:
        Tuple of (include_pattern, exclude_args_string)
    """
    try:
        data = json.loads(json_str)
        include = data.get("include", "*")
        exclude_list = data.get("exclude", [])

        # Convert exclude list to pytest --ignore arguments
        exclude_args = " ".join([f"--ignore={e}" for e in exclude_list]) if exclude_list else ""

        return include, exclude_args
    except (json.JSONDecodeError, KeyError) as e:
        sys.stderr.write(f"Error parsing test patterns: {e}\n")
        return "*", ""


def parse_test_cases(json_str: str) -> list[tuple[str, str, str]]:
    """
    Parse functional test configuration JSON and extract test cases.

    Args:
        json_str: JSON string containing test configuration

    Returns:
        List of tuples (task, model, config)
    """
    try:
        tests_config = json.loads(json_str)
        test_cases = []

        for task, models_data in tests_config.items():
            for model, test_configs in models_data.items():
                if isinstance(test_configs, list):
                    for config in test_configs:
                        test_cases.append((task, model, config))

        return test_cases
    except (json.JSONDecodeError, KeyError) as e:
        sys.stderr.write(f"Error parsing test configuration: {e}\n")
        return []


def parse_device_list(json_str: str) -> list[str]:
    """
    Parse device types JSON array into a list of device names.

    Args:
        json_str: JSON array string like '["a100", "a800", "h100"]'

    Returns:
        List of device names
    """
    try:
        devices = json.loads(json_str)
        if not isinstance(devices, list):
            sys.stderr.write(f"Error: Expected JSON array, got {type(devices).__name__}\n")
            return []
        return devices
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error parsing device list: {e}\n")
        return []


def main():
    """CLI interface for helper functions."""
    if len(sys.argv) < 2:
        print("Usage: helpers.py <command> [args]", file=sys.stderr)
        print("\nCommands:", file=sys.stderr)
        print(
            "  extract-patterns    - Extract include/exclude patterns from JSON stdin",
            file=sys.stderr,
        )
        print("  parse-test-cases    - Parse test cases from JSON stdin", file=sys.stderr)
        print("  parse-devices       - Parse device list from JSON stdin", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    json_input = sys.stdin.read().strip()

    if command == "extract-patterns":
        include, exclude = extract_test_patterns(json_input)
        print(f"INCLUDE={include}")
        print(f"EXCLUDE={exclude}")

    elif command == "parse-test-cases":
        test_cases = parse_test_cases(json_input)
        for task, model, config in test_cases:
            print(f"{task} {model} {config}")

    elif command == "parse-devices":
        devices = parse_device_list(json_input)
        print(" ".join(devices))

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

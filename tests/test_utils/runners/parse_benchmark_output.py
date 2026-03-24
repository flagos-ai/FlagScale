"""Parse host_0_localhost.output log file and produce benchmark.json.

Usage:
    python parse_benchmark_output.py <log_file> <gold_values_file> [output_json] [platform] [device]

The script extracts benchmark metrics from the training log using metric keys
defined in the gold values file, then writes a structured JSON report.
"""

import json
import re
import sys

# Default benchmark metric keys if no gold values file is provided
DEFAULT_METRIC_KEYS = [
    "elapsed time per iteration (ms):",
    "throughput per GPU (TFLOP/s/GPU):",
]


def extract_metrics_from_log(lines, metric_keys):
    """Extract metrics from training log lines.

    Log format (pipe-separated):
        " [2026-01-15 09:13:30] iteration 4/10 | ... | lm loss: 1.161108E+01 | ... |"
    """
    results = {key: [] for key in metric_keys}

    for line in lines:
        if "iteration" not in line:
            continue

        parts = line.split("|")
        for part in parts:
            part = part.strip()
            for key in metric_keys:
                if part.startswith(key.rstrip(":")):
                    match = re.search(r":\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", part)
                    if match:
                        try:
                            results[key].append(float(match.group(1)))
                        except ValueError:
                            continue

    return results


def main():
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <log_file> <gold_values_file> [output_json] [platform] [device]",
            file=sys.stderr,
        )
        sys.exit(1)

    log_file = sys.argv[1]
    gold_values_file = sys.argv[2]
    output_json = sys.argv[3] if len(sys.argv) > 3 else "benchmark.json"
    platform = sys.argv[4] if len(sys.argv) > 4 else None
    device = sys.argv[5] if len(sys.argv) > 5 else None

    # Read log file
    with open(log_file, "r") as f:
        lines = f.readlines()

    # Determine metric keys from gold values file
    try:
        with open(gold_values_file, "r") as f:
            gold_data = json.load(f)

        # Handle platform/device-classified gold values structure
        # Drill down through non-metric levels (platform -> device -> metrics)
        def _is_metric_keys(keys):
            return any(":" in k for k in keys)

        if platform and platform in gold_data:
            gold_data = gold_data[platform]
        elif not _is_metric_keys(gold_data.keys()):
            gold_data = gold_data[next(iter(gold_data))]

        if device and device in gold_data:
            gold_data = gold_data[device]
        elif not _is_metric_keys(gold_data.keys()):
            gold_data = gold_data[next(iter(gold_data))]
        metric_keys = list(gold_data.keys())
    except (FileNotFoundError, json.JSONDecodeError):
        print(
            f"Warning: Could not load gold values from {gold_values_file}, "
            f"using default metric keys",
            file=sys.stderr,
        )
        metric_keys = DEFAULT_METRIC_KEYS

    # Extract metrics
    metrics = extract_metrics_from_log(lines, metric_keys)

    # Build benchmark_metrics.json: { "metric_name": {"values": [...]}, ... }
    # Each value is an object whose keys match header_config field names
    benchmark = {}
    for key in metric_keys:
        benchmark[key] = {"values": metrics.get(key, [])}

    # Write output
    with open(output_json, "w") as f:
        json.dump(benchmark, f, indent=4)

    print(f"Benchmark results written to {output_json}")
    print(f"Metrics extracted: {list(benchmark.keys())}")
    for key, data in benchmark.items():
        print(f"  {key}: {len(data['values'])} values")


if __name__ == "__main__":
    main()

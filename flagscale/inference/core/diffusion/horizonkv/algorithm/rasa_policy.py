# Modified for the HorizonKV inference artifact; see NOTICE.
import json
import math
from pathlib import Path


def validate_policy(data):
    if data.get("schema_version") != 1 or data.get("policy") != "bounded_two_config":
        raise ValueError("Expected schema_version=1, policy=bounded_two_config")
    for key in ("best_config", "min_config"):
        point = data[key]
        if type(point["frames"]) is not int or not 5 <= point["frames"] <= 256:
            raise ValueError(
                f"{key}.frames must be an integer in [5, 256] (one sink + four recent frames)"
            )
        sp = point["block_sparsity"]
        if not isinstance(sp, (int, float)) or not math.isfinite(sp) or (not 0 <= sp < 1):
            raise ValueError(f"Invalid {key}.block_sparsity")
    best, minimum = (data["best_config"], data["min_config"])
    if minimum["frames"] > best["frames"]:
        raise ValueError("min_config.frames exceeds best_config.frames")
    budget = lambda p: p["frames"] * (1 - p["block_sparsity"])
    if not math.isclose(budget(best), budget(minimum), rel_tol=1e-06, abs_tol=1e-08):
        raise ValueError("The two configurations must have equal k * (1 - sparsity)")
    for key in ("switch_miss_items", "overlap_min_miss_items"):
        if type(data[key]) is not int or data[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if data["switch_miss_items"] < data["overlap_min_miss_items"]:
        raise ValueError("RASA gate must not be below the overlap gate")
    return data


def load_policy(path):
    return validate_policy(json.loads(Path(path).read_text()))

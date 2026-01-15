from typing import Any

import torch


def parse_torch_dtype(value: Any) -> torch.dtype | str | None:
    """Parse a YAML-provided dtype value into a torch.dtype.

    Accepts values like:
    - torch.bfloat16, bfloat16, bf16
    - torch.float16, float16, fp16, half
    - torch.float32, float32, fp32
    - torch.float64, float64, fp64

    Returns:
        - torch.dtype if recognized
        - None otherwise
    """

    if value is None:
        return None

    if isinstance(value, torch.dtype):
        return value

    if isinstance(value, str):
        s = value.strip()
        if s.startswith("torch."):
            s = s.split(".", 1)[1]
        alias_map = {
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp16": "float16",
            "half": "float16",
            "float16": "float16",
            "fp32": "float32",
            "float32": "float32",
            "fp64": "float64",
            "float64": "float64",
        }
        key = alias_map.get(s.lower(), s.lower())
        if hasattr(torch, key) and isinstance(getattr(torch, key), torch.dtype):
            return getattr(torch, key)

    return None

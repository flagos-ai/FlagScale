# Copyright (c) 2025, BAAI. All rights reserved.
#
# Copied from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

from pathlib import Path


def resolve_path(path: str) -> Path:
    """Resolve a path to an absolute path."""
    return Path(path).expanduser().absolute().resolve()

"""
MemRift: Memory-efficient weight and activation compression for LLM training.

This package provides:
- Offline weight compression (split_zstd format)
- Training-time dynamic weight loading/release (layer-level hooks)
- Optional activation compression (saved_tensors_hooks)
- Async decompression (threadpool + CUDA streams)

Usage:
    In YAML config, enable via memrift_* prefixed keys:
    - memrift_enable: true
    - memrift_weight_enable: true
    - memrift_compressed_weight_dir: /path/to/compressed_weights
"""

from . import ops
from .format import (
    load_index,
    read_compressed_weight,
    write_compressed_weight,
)
from .train_hooks import inject_memrift_if_configured

__all__ = [
    "read_compressed_weight",
    "write_compressed_weight",
    "load_index",
    "inject_memrift_if_configured",
    "ops",
]

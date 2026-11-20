# Modified for the HorizonKV inference artifact; see NOTICE.
"""Settings for pooled-QK retrieval, host KV history and 64x64 attention."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KVSparseConfig:
    max_kv_frames: int = 7
    gpu_cache_frames: int = 15
    block_target_density: float = 0.5
    overlap_block_sparsity: float = 0.5
    mandatory_recent_kv_frames: int = 4


def load_kv_sparse_config(path: str | Path) -> KVSparseConfig:
    data = json.loads(Path(path).read_text())
    return KVSparseConfig(
        max_kv_frames=data["max_stage1_kv_frames"],
        gpu_cache_frames=data["gpu_cache_frames"],
        block_target_density=data["block_target_density"],
        overlap_block_sparsity=data["overlap_block_sparsity"],
        mandatory_recent_kv_frames=data.get("mandatory_recent_kv_frames", 4),
    )


def attach_kv_sparse_to_model(model, path: str | Path) -> None:
    model.kv_sparse_config = load_kv_sparse_config(path)

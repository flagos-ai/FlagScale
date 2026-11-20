# Modified for the HorizonKV inference artifact; see NOTICE.
"""Allocate and reset bounded GPU slots and host KV history."""
from __future__ import annotations

from typing import Any

import torch

from .cache_slots import _kv_cache_wait_pending_slot_metadata_commit
from .config import KVSparseConfig
from .execution import _can_pin_cpu_memory, _synchronize_cuda_streams, _zeros_cpu
from .offload import _kv_cache_wait_pending_cpu_copy


def build_kv_cache_state(
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
    frame_capacity: int,
    frame_seqlen: int,
    num_heads: int,
    head_dim: int,
    kv_sparse_config: KVSparseConfig | None = None,
) -> dict[str, Any]:
    frame_capacity = max(1, int(frame_capacity))
    frame_seqlen = max(1, int(frame_seqlen))
    token_capacity = frame_capacity * frame_seqlen
    num_heads = max(1, int(num_heads))
    head_dim = max(1, int(head_dim))
    device = torch.device(device)
    gpu_cache_frames = getattr(kv_sparse_config, "gpu_cache_frames", None)
    gpu_cache_frames = max(0, int(gpu_cache_frames or 0))
    base: dict[str, Any] = {
        "cache_mode": "cpu_offload",
        "cpu_offload_kv": True,
        "frame_capacity": int(frame_capacity),
        "frame_seqlen": int(frame_seqlen),
        "token_capacity": int(token_capacity),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
    }
    pin_memory = _can_pin_cpu_memory(device)
    cpu_kv_shape = [batch_size, num_heads, token_capacity, head_dim]
    base.update(
        {
            "cpu_k": _zeros_cpu(cpu_kv_shape, dtype=dtype, pin_memory=pin_memory),
            "cpu_v": _zeros_cpu(cpu_kv_shape, dtype=dtype, pin_memory=pin_memory),
            "frame_k_mean": torch.zeros(
                [batch_size, frame_capacity, num_heads, head_dim], dtype=dtype, device=device
            ),
            "gpu_cache_frames": int(gpu_cache_frames),
            "gpu_frame_to_slot_dense": (
                torch.full((num_heads, frame_capacity), -1, dtype=torch.long)
                if gpu_cache_frames > 0
                else None
            ),
            "gpu_slot_frame_ids_tensor_cpu": (
                torch.full((num_heads, gpu_cache_frames), -1, dtype=torch.long)
                if gpu_cache_frames > 0
                else None
            ),
            "gpu_slot_frame_lens_tensor_cpu": (
                torch.zeros((num_heads, gpu_cache_frames), dtype=torch.long)
                if gpu_cache_frames > 0
                else None
            ),
            "gpu_slot_last_used_tensor_cpu": (
                torch.full((num_heads, gpu_cache_frames), -1, dtype=torch.long)
                if gpu_cache_frames > 0
                else None
            ),
            "gpu_identity_filled_frames": 0,
            "_kv_sparse_clean_gpu_identity_fill_scheduled_frames": 0,
            "gpu_clock": 0,
            "_kv_sparse_overlap_frame_indices_cpu_workspace": {
                "buffer": None,
                "event": None,
            },
        }
    )
    if gpu_cache_frames > 0:
        gpu_k_slots = torch.empty(
            [batch_size, num_heads, gpu_cache_frames, frame_seqlen, head_dim],
            dtype=dtype,
            device=device,
        )
        gpu_v_slots = torch.empty_like(gpu_k_slots)
        base["gpu_k_slots"] = gpu_k_slots
        base["gpu_v_slots"] = gpu_v_slots
        base["gpu_k"] = gpu_k_slots.reshape(
            batch_size, num_heads * gpu_cache_frames * frame_seqlen, 1, head_dim
        )
        base["gpu_v"] = gpu_v_slots.reshape(
            batch_size, num_heads * gpu_cache_frames * frame_seqlen, 1, head_dim
        )
        gpu_frame_to_slot_device = torch.full(
            (num_heads, frame_capacity), -1, dtype=torch.long, device=device
        )
        identity_count = min(frame_capacity, gpu_cache_frames)
        if identity_count > 0:
            identity_slots = torch.arange(identity_count, dtype=torch.long, device=device)
            gpu_frame_to_slot_device[:, :identity_count] = identity_slots.view(1, -1)
        base["gpu_frame_to_slot_device"] = gpu_frame_to_slot_device
    else:
        base["gpu_k"] = None
        base["gpu_v"] = None
        base["gpu_k_slots"] = None
        base["gpu_v_slots"] = None
        base["gpu_frame_to_slot_device"] = None
    return base


def reset_kv_cache_state(kv_cache: dict[str, Any]) -> None:
    _kv_cache_wait_pending_slot_metadata_commit(kv_cache)
    _kv_cache_wait_pending_cpu_copy(kv_cache)
    if "global_end_index" in kv_cache:
        kv_cache["global_end_index"].fill_(0)
    if "local_end_index" in kv_cache:
        kv_cache["local_end_index"].fill_(0)
    gpu_k = kv_cache.get("gpu_k", None)
    if isinstance(gpu_k, torch.Tensor) and gpu_k.is_cuda:
        _synchronize_cuda_streams(gpu_k.device)
    dense_frame_to_slot = kv_cache.get("gpu_frame_to_slot_dense", None)
    if isinstance(dense_frame_to_slot, torch.Tensor):
        dense_frame_to_slot.fill_(-1)
    slot_frame_ids_tensor = kv_cache.get("gpu_slot_frame_ids_tensor_cpu", None)
    if isinstance(slot_frame_ids_tensor, torch.Tensor):
        slot_frame_ids_tensor.fill_(-1)
    slot_frame_lens_tensor = kv_cache.get("gpu_slot_frame_lens_tensor_cpu", None)
    if isinstance(slot_frame_lens_tensor, torch.Tensor):
        slot_frame_lens_tensor.zero_()
    slot_last_used_tensor = kv_cache.get("gpu_slot_last_used_tensor_cpu", None)
    if isinstance(slot_last_used_tensor, torch.Tensor):
        slot_last_used_tensor.fill_(-1)
    kv_cache["gpu_clock"] = 0
    kv_cache["gpu_identity_filled_frames"] = 0
    kv_cache["_kv_sparse_clean_gpu_identity_fill_scheduled_frames"] = 0
    gpu_frame_to_slot_device = kv_cache.get("gpu_frame_to_slot_device", None)
    if isinstance(gpu_frame_to_slot_device, torch.Tensor):
        gpu_frame_to_slot_device.fill_(-1)
        identity_count = min(
            int(gpu_frame_to_slot_device.shape[1]), int(kv_cache.get("gpu_cache_frames", 0) or 0)
        )
        if identity_count > 0:
            identity_slots = torch.arange(
                identity_count, dtype=torch.long, device=gpu_frame_to_slot_device.device
            )
            gpu_frame_to_slot_device[:, :identity_count] = identity_slots.view(1, -1)
    frame_indices_workspace = kv_cache.get("_kv_sparse_overlap_frame_indices_cpu_workspace", None)
    if isinstance(frame_indices_workspace, dict):
        frame_indices_workspace["event"] = None

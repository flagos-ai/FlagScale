# Modified for the HorizonKV inference artifact; see NOTICE.
"""CUDA streams, pinned control buffers and execution switches."""
from __future__ import annotations

import os
from typing import Any

import torch

_ALL_HEAD_INDICES_DEVICE_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


_SHARED_CUDA_STREAMS: dict[tuple[str, int, str], torch.cuda.Stream] = {}


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _overlap_attn_enabled_by_env() -> bool:
    return _truthy_env("HORIZONKV_OVERLAP_ATTN", "0")


def _overlap_flash_split_attention_enabled_by_env() -> bool:
    return _truthy_env("HORIZONKV_OVERLAP_FLASH_SPLIT_ATTN", "1")


def _overlap_split_min_miss_items_by_env() -> int:
    return int(os.environ["HORIZONKV_OVERLAP_SPLIT_MIN_MISS_ITEMS"])


def _can_pin_cpu_memory(device: torch.device | str) -> bool:
    dev = torch.device(device)
    return dev.type == "cuda" and torch.cuda.is_available()


def _zeros_cpu(
    shape: tuple[int, ...] | list[int], *, dtype: torch.dtype, pin_memory: bool
) -> torch.Tensor:
    try:
        return torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
    except RuntimeError:
        return torch.zeros(shape, dtype=dtype, device="cpu")


def _pin_cpu_tensor_if_available(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        return tensor
    try:
        if tensor.is_pinned() or not torch.cuda.is_available():
            return tensor
    except Exception:
        return tensor
    try:
        return tensor.pin_memory()
    except Exception:
        return tensor


def _shared_cuda_stream(*, device: torch.device, name: str) -> torch.cuda.Stream | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    device_index = (
        int(device.index) if device.index is not None else int(torch.cuda.current_device())
    )
    key = ("cuda", device_index, str(name))
    stream = _SHARED_CUDA_STREAMS.get(key, None)
    if stream is not None:
        return stream
    stream = torch.cuda.Stream(device=torch.device("cuda", device_index))
    _SHARED_CUDA_STREAMS[key] = stream
    return stream


def _stage_overlap_frame_indices_to_cpu_async(
    kv_cache: dict[str, Any], frame_indices: torch.Tensor
) -> dict[str, Any] | None:
    if (
        frame_indices.device.type != "cuda"
        or frame_indices.dtype != torch.long
        or frame_indices.ndim != 2
        or (not torch.cuda.is_available())
    ):
        return None
    copy_stream = _shared_cuda_stream(device=frame_indices.device, name="overlap_frame_indices_d2h")
    if copy_stream is None:
        return None
    workspace = kv_cache.setdefault(
        "_kv_sparse_overlap_frame_indices_cpu_workspace", {"buffer": None, "event": None}
    )
    pending = workspace.setdefault("pending", [])
    if pending:
        pending[:] = [(buf, ev) for buf, ev in pending if not ev.query()]
    prior_event = workspace.get("event", None)
    buffer = workspace.get("buffer", None)
    if prior_event is not None:
        if not prior_event.query():
            pending.append((buffer, prior_event))
            buffer = None
        else:
            workspace["event"] = None
    required = int(frame_indices.numel())
    if buffer is None or buffer.numel() < required:
        capacity = 1 << max(0, required - 1).bit_length()
        try:
            buffer = torch.empty((capacity,), dtype=torch.long, device="cpu", pin_memory=True)
        except RuntimeError:
            return None
    workspace["buffer"] = buffer
    cpu_view = buffer.reshape(-1).narrow(0, 0, required).view(frame_indices.shape)
    try:
        with torch.cuda.device(frame_indices.device):
            producer_stream = torch.cuda.current_stream(frame_indices.device)
            producer_event = torch.cuda.Event()
            producer_event.record(producer_stream)
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(producer_event)
                cpu_view.copy_(frame_indices.detach(), non_blocking=True)
                done_event = torch.cuda.Event()
                done_event.record(copy_stream)
        frame_indices.detach().record_stream(copy_stream)
    except Exception:
        return None
    workspace["event"] = done_event
    return {"tensor": cpu_view, "event": done_event, "workspace": workspace}


def _resolve_staged_overlap_frame_indices_cpu(staged: dict[str, Any] | None) -> torch.Tensor | None:
    if staged is None:
        return None
    event = staged["event"]
    if not event.query():
        event.synchronize()
    workspace = staged["workspace"]
    if workspace["event"] is event:
        workspace["event"] = None
    return staged["tensor"]


def _all_head_indices_device(num_heads: int, device: torch.device) -> torch.Tensor:
    num_heads = int(num_heads)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device("cuda", int(torch.cuda.current_device()))
    device_index = int(device.index) if device.index is not None else -1
    key = (str(device.type), device_index, num_heads)
    cached = _ALL_HEAD_INDICES_DEVICE_CACHE.get(key, None)
    if cached is not None:
        return cached
    tensor = torch.arange(num_heads, device=device, dtype=torch.long)
    _ALL_HEAD_INDICES_DEVICE_CACHE[key] = tensor
    return tensor


def _kv_cache_stream_ready_event(device: torch.device) -> torch.cuda.Event | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


def _synchronize_cuda_streams(device: torch.device) -> None:
    """Drain this device's producer streams before resetting request state."""
    for (_, index, _), stream in _SHARED_CUDA_STREAMS.items():
        if index == device.index:
            stream.synchronize()

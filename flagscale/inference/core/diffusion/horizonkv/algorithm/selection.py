# Modified for the HorizonKV inference artifact; see NOTICE.
"""Compulsory frames and pooled-QK frame retrieval."""
from __future__ import annotations
from typing import Any
import os
from .transfer import select_pooled_qk_topk
import torch
from .execution import _all_head_indices_device, _can_pin_cpu_memory


_COMPULSORY_INDICES_DEVICE_READY: dict[
    tuple[str, int, int, tuple[int, ...]], tuple[torch.cuda.Event, int]
] = {}


_COMPULSORY_INDICES_DEVICE_CACHE: dict[tuple[str, int, int, tuple[int, ...]], torch.Tensor] = {}


def _sink_frame_indices(Lk: int, kv_start: int, frame_seqlen: int, sink_tokens: int) -> set[int]:
    if sink_tokens <= 0:
        return set()
    sink_len = min(Lk, max(0, int(sink_tokens) - int(kv_start)))
    if sink_len <= 0:
        return set()
    last_frame = (sink_len - 1) // frame_seqlen
    return set(range(last_frame + 1))


def _mandatory_kv_frame_indices(
    Lk: int,
    frame_seqlen: int,
    kv_start: int,
    local_end_index: int,
    current_start: int,
    current_end: int,
    num_frame_per_block: int,
) -> set[int]:
    if Lk <= 0 or frame_seqlen <= 0:
        return set()
    chunk_tokens = max(1, int(num_frame_per_block) * int(frame_seqlen))
    chunk_lo = int(current_start) // chunk_tokens * chunk_tokens
    chunk_hi = chunk_lo + chunk_tokens
    cs, ce = (int(current_start), int(current_end))
    base = int(current_end) - int(local_end_index) + int(kv_start)
    intervals = [(chunk_lo - base, chunk_hi - base), (cs - base, ce - base)]
    out: set[int] = set()
    for a, b in intervals:
        lo = max(0, int(a))
        hi = min(Lk, int(b))
        if lo >= hi:
            continue
        f0 = lo // frame_seqlen
        f1 = (hi - 1) // frame_seqlen
        out.update(range(f0, f1 + 1))
    return out


def _preceding_kv_frame_indices(
    Lk: int,
    frame_seqlen: int,
    kv_start: int,
    local_end_index: int,
    current_start: int,
    current_end: int,
    num_frame_per_block: int,
    preceding_frames: int | None,
) -> set[int]:
    if Lk <= 0 or frame_seqlen <= 0 or preceding_frames is None or (int(preceding_frames) <= 0):
        return set()
    chunk_tokens = max(1, int(num_frame_per_block) * int(frame_seqlen))
    chunk_start = int(current_start) // chunk_tokens * chunk_tokens
    window_global_start = int(current_end) - int(local_end_index) + int(kv_start)
    local_chunk_start = int(chunk_start) - int(window_global_start)
    end_frame = local_chunk_start // int(frame_seqlen)
    n_frames = max(1, (int(Lk) + int(frame_seqlen) - 1) // int(frame_seqlen))
    end_frame = max(0, min(int(end_frame), n_frames))
    start_frame = max(0, end_frame - int(preceding_frames))
    return set(range(start_frame, end_frame))


def _recent_kv_frame_indices(Lk: int, frame_seqlen: int, recent_frames: int | None) -> set[int]:
    if Lk <= 0 or frame_seqlen <= 0 or recent_frames is None or (int(recent_frames) <= 0):
        return set()
    n_frames = max(1, (int(Lk) + int(frame_seqlen) - 1) // int(frame_seqlen))
    start_frame = max(0, n_frames - int(recent_frames))
    return set(range(start_frame, n_frames))


def _overlay_live_frame_stats(
    cached_stats: torch.Tensor | None,
    live_stats: torch.Tensor | None,
    *,
    live_start_index: int | None,
    frame_seqlen: int,
    n_frames: int,
) -> torch.Tensor | None:
    if cached_stats is None:
        return None
    out = cached_stats[:, :n_frames]
    if live_stats is None or live_start_index is None or live_stats.shape[1] == 0:
        return out
    start_frame = int(live_start_index) // int(frame_seqlen)
    end_frame = min(int(n_frames), start_frame + int(live_stats.shape[1]))
    if start_frame < 0 or start_frame >= end_frame:
        return out
    out = out.clone()
    out[:, start_frame:end_frame] = live_stats[:, : end_frame - start_frame]
    return out


def _cached_compulsory_indices_device(
    kv_cache: dict[str, Any],
    compulsory_frames: set[int] | list[int] | tuple[int, ...],
    n_frames: int,
    device: torch.device,
) -> torch.Tensor:
    values = tuple(sorted({int(v) for v in compulsory_frames if 0 <= int(v) < int(n_frames)}))
    device_index = int(device.index) if device.index is not None else -1
    key = (str(device.type), device_index, int(n_frames), values)
    cache = kv_cache.setdefault("_kv_sparse_compulsory_indices_device_cache", {})
    tensor = _COMPULSORY_INDICES_DEVICE_CACHE.get(key, None)
    if tensor is None and isinstance(cache, dict):
        tensor = cache.get(key, None)
    if (
        isinstance(tensor, torch.Tensor)
        and tensor.device == device
        and (tensor.dtype == torch.long)
        and (int(tensor.numel()) == len(values))
    ):
        ready = _COMPULSORY_INDICES_DEVICE_READY.get(key)
        if ready is not None and torch.cuda.is_available():
            current_stream = torch.cuda.current_stream(device)
            if int(current_stream.cuda_stream) != int(ready[1]):
                current_stream.wait_event(ready[0])
        return tensor
    tensor = torch.empty((len(values),), device=device, dtype=torch.long)
    if values:
        cpu_cache = kv_cache.setdefault("_kv_sparse_compulsory_indices_cpu_cache", {})
        cpu_tensor = cpu_cache.get(values, None) if isinstance(cpu_cache, dict) else None
        if (
            not isinstance(cpu_tensor, torch.Tensor)
            or cpu_tensor.device.type != "cpu"
            or cpu_tensor.dtype != torch.long
            or (int(cpu_tensor.numel()) != len(values))
        ):
            cpu_tensor = torch.tensor(values, device="cpu", dtype=torch.long)
            if _can_pin_cpu_memory(device):
                try:
                    cpu_tensor = cpu_tensor.pin_memory()
                except Exception:
                    pass
            if isinstance(cpu_cache, dict):
                cpu_cache[values] = cpu_tensor
        tensor.copy_(
            cpu_tensor, non_blocking=bool(getattr(cpu_tensor, "is_pinned", lambda: False)())
        )
    if torch.cuda.is_available() and device.type == "cuda":
        producer_stream = torch.cuda.current_stream(device)
        ready_event = torch.cuda.Event()
        ready_event.record(producer_stream)
        _COMPULSORY_INDICES_DEVICE_READY[key] = (ready_event, int(producer_stream.cuda_stream))
    if isinstance(cache, dict):
        cache[key] = tensor
    _COMPULSORY_INDICES_DEVICE_CACHE[key] = tensor
    return tensor


def _select_kv_frame_indices_max_kv_frames_batched_fused(
    kv_cache: dict[str, Any],
    roped_query: torch.Tensor,
    frame_k_index_window: torch.Tensor | None,
    *,
    n_frames: int,
    frame_seqlen: int,
    max_kv_frames: int,
    compulsory_frames: set[int],
) -> torch.Tensor | None:
    if frame_k_index_window is None or not isinstance(frame_k_index_window, torch.Tensor):
        return None
    if roped_query.device.type != "cuda" or frame_k_index_window.device != roped_query.device:
        return None
    num_heads = int(roped_query.shape[2])
    if num_heads <= 0:
        return None
    head_indices = _all_head_indices_device(num_heads, roped_query.device)
    if int(max_kv_frames) < 0 or int(max_kv_frames) > 256:
        return None
    if int(n_frames) <= 0 or int(frame_k_index_window.shape[1]) < int(n_frames):
        return None
    compulsory_indices = _cached_compulsory_indices_device(
        kv_cache, compulsory_frames, n_frames, roped_query.device
    )
    try:
        selected = select_pooled_qk_topk(
            q=roped_query,
            frame_k=frame_k_index_window[:, : int(n_frames)],
            head_indices=head_indices,
            compulsory_indices=compulsory_indices,
            frame_seqlen=int(frame_seqlen),
            max_kv_frames=int(max_kv_frames),
        )
    except Exception as exc:
        if str(os.getenv("HORIZONKV_FUSED_SELECTION_VERBOSE", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            print(f"[horizonkv] fused selection fallback after failure: {exc}")
        return None
    if not isinstance(selected, torch.Tensor) or selected.device != roped_query.device:
        return None
    if selected.ndim != 2 or int(selected.shape[0]) != num_heads:
        return None
    return selected

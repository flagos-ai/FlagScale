# Modified for the HorizonKV inference artifact; see NOTICE.
"""Clean KV offload, GPU slot fill and transfer completion tracking."""
from __future__ import annotations

from typing import Any

import torch

from .cache_slots import (
    _kv_cache_queue_slot_metadata_commit,
    _kv_cache_slot_state_tensors,
    _kv_cache_update_slot_metadata_fast,
)
from .execution import _kv_cache_stream_ready_event, _shared_cuda_stream, _truthy_env
from .transfer import copy_kv_frame_plan_to_gpu_slots, copy_kv_span_to_cpu_cache


def update_gpu_kv_cache_frames(
    kv_cache: dict[str, Any],
    *,
    start_frame: int,
    end_frame: int,
    source_start_index: int,
    current_start: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    frame_k: torch.Tensor,
    frame_v: torch.Tensor,
    producer_event: torch.cuda.Event | None = None,
) -> None:
    _ = (current_start, num_frame_per_block)
    if not isinstance(kv_cache, dict):
        return
    gpu_cache_frames = int(kv_cache.get("gpu_cache_frames", 0) or 0)
    if gpu_cache_frames <= 0:
        return
    committed_identity = max(0, int(kv_cache.get("gpu_identity_filled_frames", 0) or 0))
    scheduled_identity = max(
        committed_identity,
        int(
            kv_cache.get("_kv_sparse_clean_gpu_identity_fill_scheduled_frames", committed_identity)
            or 0
        ),
    )
    identity_start = min(scheduled_identity, gpu_cache_frames)
    if identity_start >= gpu_cache_frames:
        return
    gpu_k = kv_cache.get("gpu_k", None)
    gpu_v = kv_cache.get("gpu_v", None)
    if (
        not isinstance(gpu_k, torch.Tensor)
        or not isinstance(gpu_v, torch.Tensor)
        or (not isinstance(frame_k, torch.Tensor))
        or (not isinstance(frame_v, torch.Tensor))
        or (frame_k.device != gpu_k.device)
        or (frame_v.device != gpu_v.device)
    ):
        return
    slot_state = _kv_cache_slot_state_tensors(kv_cache)
    if slot_state is None:
        return
    cpu_k = kv_cache.get("cpu_k", None)
    if not isinstance(cpu_k, torch.Tensor):
        return
    frame_seqlen = int(frame_seqlen)
    if frame_seqlen <= 0 or frame_k.ndim != 4 or frame_v.ndim != 4:
        return
    if int(frame_k.shape[1]) <= 0 or int(frame_k.shape[2]) <= 0:
        return
    start = max(0, int(start_frame))
    end = max(start, int(end_frame))
    if start >= end:
        return
    total_tokens = int(cpu_k.shape[2])
    source_start = int(source_start_index)
    source_token_count = int(frame_k.shape[1])
    num_heads = min(
        int(kv_cache.get("num_heads", 0) or 0),
        int(frame_k.shape[2]),
        int(frame_v.shape[2]),
        int(slot_state[1].shape[0]),
    )
    if num_heads <= 0:
        return
    frame_capacity = int(slot_state[0].shape[1])
    fill_start = int(identity_start)
    if start > fill_start:
        return
    fill_end = min(end, gpu_cache_frames, frame_capacity)
    if fill_start >= fill_end:
        return
    abs_start = fill_start * frame_seqlen
    rel_start = abs_start - source_start
    source_token_count = min(source_token_count, int(frame_v.shape[1]))
    if rel_start < 0 or rel_start >= source_token_count:
        return
    frame_count = min(fill_end - fill_start, (source_token_count - rel_start) // frame_seqlen)
    if frame_count <= 0:
        return
    fill_end = fill_start + frame_count
    rel_end = rel_start + frame_count * frame_seqlen
    identity_fill_end = fill_end
    frame_ids_cpu = torch.arange(fill_start, fill_end, dtype=torch.long)
    heads_cpu = torch.arange(num_heads, dtype=torch.long).repeat(frame_count)
    frames_cpu = frame_ids_cpu.repeat_interleave(num_heads)
    slots_cpu = frames_cpu.clone()
    copy_stream = _shared_cuda_stream(device=gpu_k.device, name="clean_kv_gpu_fill")
    copy_done: torch.cuda.Event | None = None
    gpu_k_slots = kv_cache.get("gpu_k_slots", None)
    gpu_v_slots = kv_cache.get("gpu_v_slots", None)
    use_slot_layout = (
        isinstance(gpu_k_slots, torch.Tensor)
        and isinstance(gpu_v_slots, torch.Tensor)
        and (gpu_k_slots.device == gpu_k.device)
        and (gpu_v_slots.device == gpu_v.device)
        and (gpu_k_slots.ndim == 5)
        and (gpu_v_slots.ndim == 5)
    )

    def _record_scheduled_identity_fill() -> None:
        kv_cache["_kv_sparse_clean_gpu_identity_fill_scheduled_frames"] = max(
            int(kv_cache.get("_kv_sparse_clean_gpu_identity_fill_scheduled_frames", 0) or 0),
            int(identity_fill_end),
        )

    def _copy_identity_frames() -> None:
        if use_slot_layout:
            assert isinstance(gpu_k_slots, torch.Tensor)
            assert isinstance(gpu_v_slots, torch.Tensor)
            batch_size = int(frame_k.shape[0])
            k_src = (
                frame_k[:, rel_start:rel_end, :num_heads, :]
                .reshape(batch_size, frame_count, frame_seqlen, num_heads, int(frame_k.shape[-1]))
                .permute(0, 3, 1, 2, 4)
            )
            v_src = (
                frame_v[:, rel_start:rel_end, :num_heads, :]
                .reshape(batch_size, frame_count, frame_seqlen, num_heads, int(frame_v.shape[-1]))
                .permute(0, 3, 1, 2, 4)
            )
            gpu_k_slots[:, :num_heads, fill_start:fill_end].copy_(k_src, non_blocking=True)
            gpu_v_slots[:, :num_heads, fill_start:fill_end].copy_(v_src, non_blocking=True)
            return
        slot_len = frame_count * frame_seqlen
        for head_id in range(num_heads):
            slot_start = (head_id * gpu_cache_frames + fill_start) * frame_seqlen
            gpu_k[:, slot_start : slot_start + slot_len].copy_(
                frame_k[:, rel_start:rel_end, head_id : head_id + 1, :], non_blocking=True
            )
            gpu_v[:, slot_start : slot_start + slot_len].copy_(
                frame_v[:, rel_start:rel_end, head_id : head_id + 1, :], non_blocking=True
            )

    if copy_stream is not None and torch.cuda.is_available() and (gpu_k.device.type == "cuda"):
        if producer_event is None:
            producer_event = _kv_cache_stream_ready_event(frame_k.device)
        with torch.cuda.device(gpu_k.device):
            with torch.cuda.stream(copy_stream):
                if producer_event is not None:
                    copy_stream.wait_event(producer_event)
                _copy_identity_frames()
                copy_done = torch.cuda.Event()
                copy_done.record(copy_stream)
        frame_k.record_stream(copy_stream)
        frame_v.record_stream(copy_stream)
        _record_scheduled_identity_fill()
        if _kv_cache_queue_slot_metadata_commit(
            kv_cache,
            head_ids=heads_cpu,
            slot_ids=slots_cpu,
            frame_ids=frames_cpu,
            frame_seqlen=frame_seqlen,
            total_tokens=total_tokens,
            ready_event=copy_done,
            device=gpu_k.device,
            identity_fill_end=identity_fill_end,
        ):
            return
    _copy_identity_frames()
    metadata_committed = _kv_cache_update_slot_metadata_fast(
        kv_cache,
        head_ids=heads_cpu,
        slot_ids=slots_cpu,
        frame_ids=frames_cpu,
        frame_seqlen=frame_seqlen,
        total_tokens=total_tokens,
    )
    if metadata_committed and identity_fill_end is not None:
        kv_cache["gpu_identity_filled_frames"] = max(
            int(kv_cache.get("gpu_identity_filled_frames", 0) or 0), identity_fill_end
        )
        _record_scheduled_identity_fill()


def _kv_cache_pending_cpu_copy_records(
    kv_cache: dict[str, Any]
) -> list[tuple[torch.cuda.Event, tuple[int, int]]]:
    records = [
        (event, span) for event, span in kv_cache.get("cpu_copy_records", []) if not event.query()
    ]
    kv_cache["cpu_copy_records"] = records
    return records


def _kv_cache_wait_pending_cpu_copy(kv_cache: dict[str, Any]) -> None:
    records = _kv_cache_pending_cpu_copy_records(kv_cache)
    if records:
        stream = torch.cuda.current_stream(kv_cache["frame_k_mean"].device)
        for event, _ in records:
            stream.wait_event(event)
        kv_cache["cpu_copy_records"] = []


def copy_kv_to_cpu_cache_async(
    kv_cache: dict[str, Any],
    *,
    local_start_index: int,
    local_end_index: int,
    key: torch.Tensor,
    value: torch.Tensor,
    producer_event: torch.cuda.Event | None = None,
) -> None:
    start = int(local_start_index)
    end = int(local_end_index)
    if start >= end:
        return
    cpu_k = kv_cache["cpu_k"]
    cpu_v = kv_cache["cpu_v"]

    def copy_fallback() -> None:
        cpu_k[:, :, start:end].copy_(key.detach().permute(0, 2, 1, 3), non_blocking=True)
        cpu_v[:, :, start:end].copy_(value.detach().permute(0, 2, 1, 3), non_blocking=True)

    if (
        key.device.type != "cuda"
        or value.device != key.device
        or (not torch.cuda.is_available())
        or (not (cpu_k.is_pinned() and cpu_v.is_pinned()))
    ):
        copy_fallback()
        kv_cache["cpu_copy_records"] = []
        return
    stream = _shared_cuda_stream(device=key.device, name="cpu_copy")
    if stream is None:
        copy_fallback()
        kv_cache["cpu_copy_records"] = []
        return
    if producer_event is None:
        producer_event = _kv_cache_stream_ready_event(key.device)
    key_detached = key.detach()
    value_detached = value.detach()
    if producer_event is not None:
        stream.wait_event(producer_event)
    with torch.cuda.stream(stream):
        copied = 0
        try:
            copied = copy_kv_span_to_cpu_cache(
                gpu_k_span=key_detached,
                gpu_v_span=value_detached,
                cpu_k=cpu_k,
                cpu_v=cpu_v,
                start_token=start,
            )
        except Exception:
            copied = 0
        if copied <= 0:
            cpu_k[:, :, start:end].copy_(key_detached.permute(0, 2, 1, 3), non_blocking=True)
            cpu_v[:, :, start:end].copy_(value_detached.permute(0, 2, 1, 3), non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
    key_detached.record_stream(stream)
    value_detached.record_stream(stream)
    records = _kv_cache_pending_cpu_copy_records(kv_cache)
    records.append((event, (start, end)))
    kv_cache["cpu_copy_records"] = records


def queue_deferred_clean_kv_to_cpu_cache(
    kv_cache: dict[str, Any],
    next_kv_cache: dict[str, Any] | None,
    *,
    local_start_index: int,
    local_end_index: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    queue_kv_cache = next_kv_cache if isinstance(next_kv_cache, dict) else kv_cache
    if (
        not isinstance(queue_kv_cache, dict)
        or key.device.type != "cuda"
        or value.device != key.device
        or (not torch.cuda.is_available())
    ):
        return False
    start = int(local_start_index)
    end = int(local_end_index)
    if start >= end:
        return True
    producer_event = _kv_cache_stream_ready_event(key.device)
    record = {
        "target_kv_cache": kv_cache,
        "local_start_index": start,
        "local_end_index": end,
        "key": key.detach(),
        "value": value.detach(),
        "producer_event": producer_event,
    }
    queue_kv_cache.setdefault("_deferred_clean_kv_offloads", []).append(record)
    return True


def flush_deferred_clean_kv_offloads(kv_cache: dict[str, Any] | None) -> None:
    if not isinstance(kv_cache, dict):
        return
    records = kv_cache.pop("_deferred_clean_kv_offloads", None)
    if not records:
        return
    for record in records:
        copy_kv_to_cpu_cache_async(
            record["target_kv_cache"],
            local_start_index=record["local_start_index"],
            local_end_index=record["local_end_index"],
            key=record["key"],
            value=record["value"],
            producer_event=record["producer_event"],
        )


def _kv_cache_copy_frame_tensor_plan_from_cpu_to_slots(
    kv_cache: dict[str, Any],
    *,
    frame_indices_cpu: torch.Tensor,
    slot_plan_cpu: torch.Tensor,
    copy_plan_cpu: torch.Tensor,
    expected_copies: int | None = None,
    frame_seqlen: int,
    stream: torch.cuda.Stream,
) -> bool:
    gpu_k = kv_cache["gpu_k"]
    gpu_v = kv_cache["gpu_v"]
    cpu_k = kv_cache["cpu_k"]
    cpu_v = kv_cache["cpu_v"]
    frame_seqlen = int(frame_seqlen)
    frame_indices_cpu = frame_indices_cpu.contiguous()
    slot_plan_cpu = slot_plan_cpu.contiguous()
    copy_plan_cpu = copy_plan_cpu.contiguous()
    if expected_copies is None:
        expected_copies = int(copy_plan_cpu.ne(0).sum().item())
    else:
        expected_copies = int(expected_copies)
    if expected_copies <= 0:
        return True
    try:
        with torch.cuda.stream(stream):
            copied = copy_kv_frame_plan_to_gpu_slots(
                cpu_k=cpu_k,
                cpu_v=cpu_v,
                gpu_k=gpu_k,
                gpu_v=gpu_v,
                frame_indices=frame_indices_cpu,
                slot_plan=slot_plan_cpu,
                copy_plan=copy_plan_cpu,
                frame_seqlen=frame_seqlen,
                gpu_cache_frames=kv_cache["gpu_cache_frames"],
            )
        return copied == expected_copies
    except Exception as exc:
        if _truthy_env("HORIZONKV_FUSED_ONLOAD_VERBOSE"):
            print(f"[horizonkv] frame-plan H2D failed: {exc}")
        return False


def _kv_cache_wait_pending_cpu_copy_for_frames_on_stream(
    kv_cache: dict[str, Any],
    frame_ids: list[int] | torch.Tensor,
    frame_seqlen: int,
    stream: torch.cuda.Stream,
) -> None:
    records = _kv_cache_pending_cpu_copy_records(kv_cache)
    if not records:
        return
    targets = [
        (int(frame_id) * int(frame_seqlen), (int(frame_id) + 1) * int(frame_seqlen))
        for frame_id in frame_ids
        if int(frame_id) >= 0
    ]
    if not targets:
        return
    for event, span in records:
        p0, p1 = (int(span[0]), int(span[1]))
        if any((start < p1 and p0 < end for start, end in targets)):
            stream.wait_event(event)

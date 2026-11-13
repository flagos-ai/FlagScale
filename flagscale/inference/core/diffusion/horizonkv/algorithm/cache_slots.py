# Modified for the HorizonKV inference artifact; see NOTICE.
"""CPU slot metadata and deferred residency commits."""
from __future__ import annotations

from typing import Any

import torch

from .execution import _pin_cpu_tensor_if_available
from .transfer import update_frame_slot_map


def _kv_cache_slot_state_tensors(
    kv_cache: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        (
            kv_cache[k]
            for k in (
                "gpu_frame_to_slot_dense",
                "gpu_slot_frame_ids_tensor_cpu",
                "gpu_slot_frame_lens_tensor_cpu",
                "gpu_slot_last_used_tensor_cpu",
            )
        )
    )


def _kv_cache_wait_pending_slot_metadata_commit(kv_cache: dict[str, Any]) -> bool:
    ok = True
    for item in kv_cache.pop("_kv_sparse_pending_slot_metadata_commits", []):
        if item["ready_event"] is not None:
            torch.cuda.current_stream(item["device"]).wait_event(item["ready_event"])
        committed = _kv_cache_commit_compact_stage_slots(
            kv_cache,
            item["stage"],
            frame_seqlen=item["frame_seqlen"],
            total_tokens=item["total_tokens"],
        )
        ok = ok and committed
        identity_end = item["identity_fill_end"]
        if committed and identity_end is not None:
            for key in (
                "gpu_identity_filled_frames",
                "_kv_sparse_clean_gpu_identity_fill_scheduled_frames",
            ):
                kv_cache[key] = max(kv_cache[key], identity_end)
    return ok


def _kv_cache_commit_incremental_tensor_slot_plan_state(
    kv_cache: dict[str, Any],
    plan_result: dict[str, Any],
    next_clock: int,
    *,
    frame_seqlen: int,
    total_tokens: int,
) -> bool:
    """Commit the planner's unique fills; hit timestamps remain request-local."""
    committed = _kv_cache_update_slot_metadata_fast(
        kv_cache,
        head_ids=plan_result["fill_head_ids"],
        slot_ids=plan_result["fill_slot_ids"],
        frame_ids=plan_result["fill_frame_ids"],
        last_used=plan_result["fill_last_used"],
        frame_seqlen=frame_seqlen,
        total_tokens=total_tokens,
    )
    kv_cache["gpu_clock"] = max(kv_cache["gpu_clock"], next_clock)
    return committed


def _kv_cache_update_slot_metadata_fast(
    kv_cache: dict[str, Any],
    *,
    head_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_seqlen: int,
    total_tokens: int,
    last_used: torch.Tensor | None = None,
) -> bool:
    slot_state = _kv_cache_slot_state_tensors(kv_cache)
    dense, slot_frame_ids_t, slot_frame_lens_t, slot_last_used_t = slot_state
    num_heads = int(slot_frame_ids_t.shape[0])
    gpu_cache_frames = int(slot_frame_ids_t.shape[1])
    frame_capacity = int(dense.shape[1])
    n = int(head_ids.numel())
    if n <= 0:
        return True

    def _keep_last_by_head_slot(
        heads_t: torch.Tensor, slots_t: torch.Tensor, *payload_tensors: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        if int(heads_t.numel()) <= 1:
            return (heads_t, slots_t, *payload_tensors)
        keys = (heads_t * gpu_cache_frames + slots_t).tolist()
        last_positions: dict[int, int] = {}
        for pos, key in enumerate(keys):
            last_positions[int(key)] = pos
        if len(last_positions) == len(keys):
            return (heads_t, slots_t, *payload_tensors)
        keep_positions = torch.tensor(sorted(last_positions.values()), dtype=torch.long)
        tensors = (heads_t, slots_t, *payload_tensors)
        return tuple((tensor[keep_positions] for tensor in tensors))

    clock = int(kv_cache.get("gpu_clock", 0) or 0)
    heads = head_ids.contiguous()
    slots = slot_ids.contiguous()
    frames = frame_ids.contiguous()
    head_slot_valid = heads.ge(0) & heads.lt(num_heads) & slots.ge(0) & slots.lt(gpu_cache_frames)
    frame_valid = head_slot_valid & frames.ge(0) & frames.lt(frame_capacity)
    if not bool(frame_valid.any().item()):
        kv_cache["gpu_clock"] = max(clock, clock + n)
        return True
    valid_heads = heads[frame_valid]
    valid_slots = slots[frame_valid]
    valid_frames = frames[frame_valid]
    valid_orig_indices = torch.arange(n, dtype=torch.long)[frame_valid]
    timestamps = clock + valid_orig_indices if last_used is None else last_used[frame_valid]
    valid_heads, valid_slots, valid_frames, timestamps = _keep_last_by_head_slot(
        valid_heads, valid_slots, valid_frames, timestamps
    )
    old_frames = slot_frame_ids_t[valid_heads, valid_slots].clone()
    old_valid = old_frames.ge(0) & old_frames.lt(frame_capacity)
    if bool(old_valid.any().item()):
        old_heads = valid_heads[old_valid]
        old_slots = valid_slots[old_valid]
        old_frames_valid = old_frames[old_valid]
        old_matches = dense[old_heads, old_frames_valid].eq(old_slots)
        if bool(old_matches.any().item()):
            dense[old_heads[old_matches], old_frames_valid[old_matches]] = -1
    frame_lens = (int(total_tokens) - valid_frames * int(frame_seqlen)).clamp(
        min=0, max=int(frame_seqlen)
    )
    nonempty = frame_lens.gt(0)
    if bool(nonempty.any().item()):
        write_heads = valid_heads[nonempty]
        write_slots = valid_slots[nonempty]
        write_frames = valid_frames[nonempty]
        write_lens = frame_lens[nonempty]
        write_last_used = timestamps[nonempty]
        dense[write_heads, write_frames] = write_slots
        slot_frame_ids_t[write_heads, write_slots] = write_frames
        slot_frame_lens_t[write_heads, write_slots] = write_lens
        slot_last_used_t[write_heads, write_slots] = write_last_used
        gpu_frame_to_slot_device = kv_cache.get("gpu_frame_to_slot_device", None)
        if (
            isinstance(gpu_frame_to_slot_device, torch.Tensor)
            and gpu_frame_to_slot_device.device.type == "cuda"
        ):
            write_heads = _pin_cpu_tensor_if_available(write_heads.contiguous())
            write_frames = _pin_cpu_tensor_if_available(write_frames.contiguous())
            write_slots = _pin_cpu_tensor_if_available(write_slots.contiguous())
            write_old_frames = _pin_cpu_tensor_if_available(old_frames[nonempty].contiguous())
            try:
                update_frame_slot_map(
                    frame_to_slot=gpu_frame_to_slot_device,
                    head_ids=write_heads.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                    frame_ids=write_frames.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                    slot_ids=write_slots.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                    evicted_frame_ids=write_old_frames.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                )
            except Exception:
                gpu_frame_to_slot_device.fill_(-1)
    kv_cache["gpu_clock"] = max(int(kv_cache.get("gpu_clock", 0) or 0), clock + n)
    return True


def _kv_cache_commit_compact_stage_slots(
    kv_cache: dict[str, Any], stage: dict[str, Any], *, frame_seqlen: int, total_tokens: int
) -> bool:
    return _kv_cache_update_slot_metadata_fast(
        kv_cache,
        head_ids=stage["scatter_head_ids_cpu"],
        slot_ids=stage["scatter_slot_ids_cpu"],
        frame_ids=stage["scatter_frame_ids_cpu"],
        frame_seqlen=frame_seqlen,
        total_tokens=total_tokens,
    )


def _kv_cache_queue_compact_stage_slots_commit(
    kv_cache: dict[str, Any],
    stage: dict[str, Any],
    *,
    frame_seqlen: int,
    total_tokens: int,
    ready_event: torch.cuda.Event | None = None,
    device: torch.device | None = None,
) -> bool:
    return _kv_cache_queue_slot_metadata_commit(
        kv_cache,
        head_ids=stage["scatter_head_ids_cpu"],
        slot_ids=stage["scatter_slot_ids_cpu"],
        frame_ids=stage["scatter_frame_ids_cpu"],
        frame_seqlen=frame_seqlen,
        total_tokens=total_tokens,
        ready_event=ready_event,
        device=device,
    )


def _kv_cache_queue_slot_metadata_commit(
    kv_cache: dict[str, Any],
    *,
    head_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_seqlen: int,
    total_tokens: int,
    ready_event: torch.cuda.Event | None = None,
    device: torch.device | None = None,
    identity_fill_end: int | None = None,
) -> bool:
    pending = kv_cache.setdefault("_kv_sparse_pending_slot_metadata_commits", [])
    pending.append(
        {
            "stage": {
                "scatter_head_ids_cpu": head_ids.contiguous(),
                "scatter_slot_ids_cpu": slot_ids.contiguous(),
                "scatter_frame_ids_cpu": frame_ids.contiguous(),
            },
            "frame_seqlen": int(frame_seqlen),
            "total_tokens": int(total_tokens),
            "ready_event": ready_event,
            "device": device,
            "identity_fill_end": identity_fill_end,
        }
    )
    return True

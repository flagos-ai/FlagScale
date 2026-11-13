# Modified for the HorizonKV inference artifact; see NOTICE.
"""Compact row plans, GPU controls and compacted RoPE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .execution import _pin_cpu_tensor_if_available
from .sparse_kernels import compacted_rope_apply_compact, compacted_rope_apply_packed_rows
from .transfer import fused_kv_frame_materialize_varlen


@dataclass(slots=True)
class _OverlapRowPlan:
    active_rows_cpu: torch.Tensor
    frame_plan_cpu: torch.Tensor
    slot_plan_cpu: torch.Tensor
    k_lens_cpu: torch.Tensor
    item_count: int
    cu_seqlens_cpu: torch.Tensor
    frame_positions_cpu: torch.Tensor
    all_full_length: bool
    covers_all_heads: bool
    control_long_cpu: torch.Tensor
    control_int_cpu: torch.Tensor


@dataclass(slots=True)
class _OverlapSlotPlan:
    slot_plan_cpu: torch.Tensor
    copy_plan_cpu: torch.Tensor
    fill_count: int
    hit_row_plan: _OverlapRowPlan | None = None
    miss_row_plan: _OverlapRowPlan | None = None


def _row_plan_from_cpp_dict(raw: Any) -> _OverlapRowPlan | None:
    if raw is None:
        return None
    active_count = raw["active_rows"].numel()
    shape = raw["frame_plan"].shape
    plan_items = raw["frame_plan"].numel()
    control_long = _pin_cpu_tensor_if_available(raw["control_long"])
    control_int = _pin_cpu_tensor_if_available(raw["control_int"])
    return _OverlapRowPlan(
        active_rows_cpu=control_long[:active_count],
        frame_plan_cpu=control_long.narrow(0, active_count, plan_items).view(shape),
        slot_plan_cpu=control_long.narrow(0, active_count + plan_items, plan_items).view(shape),
        frame_positions_cpu=control_long.narrow(0, active_count + 2 * plan_items, plan_items).view(
            shape
        ),
        k_lens_cpu=control_int[:active_count],
        cu_seqlens_cpu=control_int[active_count:],
        item_count=raw["item_count"],
        all_full_length=raw["all_full_length"],
        covers_all_heads=raw["covers_all_heads"],
        control_long_cpu=control_long,
        control_int_cpu=control_int,
    )


def _build_overlap_slot_plan(*, plan_result: dict[str, Any]) -> _OverlapSlotPlan | None:
    return _OverlapSlotPlan(
        slot_plan_cpu=_pin_cpu_tensor_if_available(plan_result["slot_plan"]),
        copy_plan_cpu=_pin_cpu_tensor_if_available(plan_result["copy_plan"]),
        fill_count=plan_result["copy_item_count"],
        hit_row_plan=_row_plan_from_cpp_dict(plan_result["hit_row_plan"]),
        miss_row_plan=_row_plan_from_cpp_dict(plan_result["miss_row_plan"]),
    )


def _apply_fast_lf_rope_to_compact_k(
    k_compact: torch.Tensor,
    *,
    num_heads: int,
    frame_seqlen: int,
    selected_frames: int,
    rope_grid_sizes: torch.Tensor,
    rope_freqs: torch.Tensor,
    start_frame: int = 0,
) -> torch.Tensor | None:
    return compacted_rope_apply_compact(
        k_compact,
        num_heads=num_heads,
        selected_frames=selected_frames,
        frame_seqlen=frame_seqlen,
        grid_sizes=rope_grid_sizes,
        freqs=rope_freqs,
        start_frame=start_frame,
    )


def _apply_fast_lf_rope_to_row_materialized_k(
    k_compact: torch.Tensor,
    controls: dict[str, Any],
    *,
    frame_seqlen: int,
    rope_grid_sizes: torch.Tensor,
    rope_freqs: torch.Tensor,
    inplace: bool = False,
    start_frame: int = 0,
) -> torch.Tensor | None:
    return compacted_rope_apply_packed_rows(
        k_compact,
        controls["frame_positions"],
        controls["k_lens"],
        controls["cu_seqlens_k"],
        frame_seqlen=frame_seqlen,
        grid_sizes=rope_grid_sizes,
        freqs=rope_freqs,
        start_frame=start_frame,
        inplace=inplace,
    )


def _copy_overlap_control_to_device(
    kv_cache: dict[str, Any], *, workspace_name: str, cpu_tensor: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, tuple[Any, ...]]:
    cpu_tensor = _pin_cpu_tensor_if_available(cpu_tensor.contiguous())
    cache = kv_cache.setdefault("_kv_sparse_overlap_control_workspaces", {})
    device_index = (
        int(device.index) if device.index is not None else int(torch.cuda.current_device())
    )
    key = (str(workspace_name), device_index, cpu_tensor.dtype)
    entry = cache.get(key, None)
    current_stream = torch.cuda.current_stream(device)
    if entry is not None:
        last_use_event = entry.get("last_use_event", None)
        if last_use_event is not None:
            current_stream.wait_event(last_use_event)
    else:
        entry = {}
    required = int(cpu_tensor.numel())
    buffer = entry.get("buffer", None)
    if buffer is None or buffer.numel() < required:
        capacity = 1 << max(0, required - 1).bit_length()
        buffer = torch.empty((capacity,), device=device, dtype=cpu_tensor.dtype)
        entry = {"buffer": buffer, "last_use_event": None}
        cache[key] = entry
    view = buffer.narrow(0, 0, required)
    view.copy_(cpu_tensor, non_blocking=True)
    return (view, key)


def _mark_overlap_control_workspaces_used(
    kv_cache: dict[str, Any], controls: dict[str, Any], stream: torch.cuda.Stream
) -> None:
    for name in ("control_long", "control_int"):
        controls[name].record_stream(stream)
    event = torch.cuda.Event()
    event.record(stream)
    cache = kv_cache["_kv_sparse_overlap_control_workspaces"]
    for key in controls["workspace_keys"]:
        cache[key]["last_use_event"] = event


def _prepare_compact_slot_plan_controls(
    kv_cache: dict[str, Any],
    *,
    device: torch.device,
    row_plan: _OverlapRowPlan,
    frame_seqlen: int,
    workspace_name: str = "default",
) -> dict[str, Any] | None:
    if row_plan.item_count <= 0:
        return None
    active_count = row_plan.active_rows_cpu.numel()
    max_selected = row_plan.frame_plan_cpu.shape[1]
    plan_items = active_count * max_selected
    control_long, long_key = _copy_overlap_control_to_device(
        kv_cache,
        workspace_name=f"{workspace_name}.long",
        cpu_tensor=row_plan.control_long_cpu,
        device=device,
    )
    control_int, int_key = _copy_overlap_control_to_device(
        kv_cache,
        workspace_name=f"{workspace_name}.int",
        cpu_tensor=row_plan.control_int_cpu,
        device=device,
    )
    active_rows = control_long[:active_count]
    shape = (active_count, max_selected)
    return dict(
        active_count=active_count,
        max_selected=max_selected,
        token_count=max_selected * frame_seqlen,
        head_indices=active_rows,
        frame_indices=control_long.narrow(0, active_count, plan_items).view(shape),
        slot_indices=control_long.narrow(0, active_count + plan_items, plan_items).view(shape),
        frame_positions=control_long.narrow(0, active_count + 2 * plan_items, plan_items).view(
            shape
        ),
        full_row_indices=None if row_plan.covers_all_heads else active_rows,
        full_q_rows=row_plan.covers_all_heads,
        k_lens=control_int[:active_count],
        cu_seqlens_k=control_int[active_count:],
        all_full_length=row_plan.all_full_length,
        control_long=control_long,
        control_int=control_int,
        workspace_keys=(long_key, int_key),
    )


def _materialize_compact_slot_plan(
    kv_cache: dict[str, Any],
    row_plan: _OverlapRowPlan,
    controls: dict[str, Any],
    *,
    frame_seqlen: int,
    live_k: torch.Tensor,
    live_v: torch.Tensor,
    live_start_index: int,
) -> dict[str, Any]:
    gpu_k, gpu_v = kv_cache["gpu_k"], kv_cache["gpu_v"]
    _, num_heads, total_tokens, dim = kv_cache["cpu_k"].shape
    device = gpu_k.device
    torch.cuda.current_stream(device).wait_event(controls["ready_event"])
    k_compact = torch.empty(
        (row_plan.item_count * frame_seqlen, 1, dim), device=device, dtype=gpu_k.dtype
    )
    v_compact = torch.empty_like(k_compact)
    if not fused_kv_frame_materialize_varlen(
        gpu_k=gpu_k,
        gpu_v=gpu_v,
        k_out=k_compact,
        v_out=v_compact,
        head_indices=controls["head_indices"],
        frame_indices=controls["frame_indices"],
        slot_indices=controls["slot_indices"],
        cu_seqlens_k=controls["cu_seqlens_k"],
        frame_seqlen=frame_seqlen,
        total_tokens=total_tokens,
        num_heads=num_heads,
        gpu_cache_frames=kv_cache["gpu_cache_frames"],
        live_k=live_k,
        live_v=live_v,
        live_start_index=live_start_index,
    ):
        raise RuntimeError("packed KV materialization extension unavailable")
    _mark_overlap_control_workspaces_used(kv_cache, controls, torch.cuda.current_stream(device))
    return dict(
        controls=controls, frame_seqlen=frame_seqlen, k_compact=k_compact, v_compact=v_compact
    )

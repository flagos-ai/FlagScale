# Modified for the HorizonKV inference artifact; see NOTICE.
"""Fetch missing frames, apply RoPE and refill cache slots asynchronously."""
from __future__ import annotations

import os
from typing import Any

import torch

from .attention import _expand_attention_part, _run_packed_attention_part
from .config import KVSparseConfig
from .execution import _pin_cpu_tensor_if_available, _shared_cuda_stream, _truthy_env
from .materialize import (
    _apply_fast_lf_rope_to_row_materialized_k,
    _mark_overlap_control_workspaces_used,
    _OverlapRowPlan,
)
from .offload import _kv_cache_wait_pending_cpu_copy_for_frames_on_stream
from .transfer import copy_compact_to_gpu_slots_d2d, copy_kv_frames_to_gpu_compact


def _build_compact_miss_stage(
    *,
    miss_row_plan: _OverlapRowPlan,
    compact_scatter_plan: dict[str, Any] | None,
    B: int,
    D: int,
    frame_seqlen: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | None:
    if B != 1 or compact_scatter_plan is None or not compact_scatter_plan["all_frames_full"]:
        return None
    plan = compact_scatter_plan
    cu_seqlens_cpu = _pin_cpu_tensor_if_available(plan["cu_seqlens_long"])
    packed_tokens = int(cu_seqlens_cpu[-1])
    compact_k = torch.empty((packed_tokens, 1, D), device=device, dtype=dtype)
    stage = dict(
        compact_k=compact_k,
        compact_v=torch.empty_like(compact_k),
        frame_seqlen=frame_seqlen,
        active_head_ids_cpu=miss_row_plan.active_rows_cpu,
        frame_plan_cpu=miss_row_plan.frame_plan_cpu,
        cu_seqlens_cpu=cu_seqlens_cpu,
        item_count=miss_row_plan.item_count,
        frame_ids_for_wait=_pin_cpu_tensor_if_available(plan["frame_ids_for_wait"]),
        copy_done=None,
    )
    for name in ("head_ids", "slot_ids", "src_rows", "src_cols", "frame_ids"):
        stage[f"scatter_{name}_cpu"] = _pin_cpu_tensor_if_available(plan[f"scatter_{name}"])
    return stage


def _submit_compact_miss_h2d(
    kv_cache: dict[str, Any],
    *,
    stage: dict[str, Any],
    cpu_k: torch.Tensor,
    cpu_v: torch.Tensor,
    frame_seqlen: int,
    stream: torch.cuda.Stream,
) -> bool:
    copy_done: torch.cuda.Event | None = None
    try:
        with torch.cuda.stream(stream):
            _kv_cache_wait_pending_cpu_copy_for_frames_on_stream(
                kv_cache, stage["frame_ids_for_wait"], frame_seqlen, stream
            )
            copied = copy_kv_frames_to_gpu_compact(
                cpu_k=cpu_k,
                cpu_v=cpu_v,
                compact_k=stage["compact_k"],
                compact_v=stage["compact_v"],
                head_ids=stage["active_head_ids_cpu"],
                frame_ids=stage["frame_plan_cpu"],
                cu_seqlens=stage["cu_seqlens_cpu"],
                frame_seqlen=frame_seqlen,
            )
            copy_done = torch.cuda.Event()
            copy_done.record(stream)
        stage["copy_done"] = copy_done
        stage["compact_k"].record_stream(stream)
        stage["compact_v"].record_stream(stream)
        expected = int(cpu_k.shape[0]) * int(stage["item_count"])
        return int(copied) == expected
    except Exception as exc:
        if copy_done is None:
            try:
                copy_done = torch.cuda.Event()
                copy_done.record(stream)
                stage["copy_done"] = copy_done
                stage["compact_k"].record_stream(stream)
                stage["compact_v"].record_stream(stream)
            except Exception:
                pass
        if str(os.getenv("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            print(f"[horizonkv] compact miss H2D fallback after failure: {exc}")
        return False


def _submit_compact_miss_lf_rope(
    kv_cache: dict[str, Any],
    *,
    stage: dict[str, Any],
    controls: dict[str, Any],
    rope_grid_sizes: torch.Tensor | None,
    rope_freqs: torch.Tensor | None,
    stream: torch.cuda.Stream,
    producer_event: torch.cuda.Event | None = None,
    lf_k_start_frame: int = 0,
) -> bool:
    copy_done = stage.get("copy_done", None)
    if copy_done is None:
        return False
    try:
        with torch.cuda.stream(stream):
            if producer_event is not None:
                stream.wait_event(producer_event)
            stream.wait_event(copy_done)
            controls_ready_event = controls.get("ready_event", None)
            if controls_ready_event is not None:
                stream.wait_event(controls_ready_event)
            attention_k = _apply_fast_lf_rope_to_row_materialized_k(
                stage["compact_k"],
                controls=controls,
                frame_seqlen=int(stage.get("frame_seqlen", 0) or 0),
                rope_grid_sizes=rope_grid_sizes,
                rope_freqs=rope_freqs,
                inplace=False,
                start_frame=int(lf_k_start_frame),
            )
            if not isinstance(attention_k, torch.Tensor):
                return False
            _mark_overlap_control_workspaces_used(kv_cache, controls, stream)
            attention_k.record_stream(stream)
            stage["compact_k"].record_stream(stream)
            rope_done = torch.cuda.Event()
            rope_done.record(stream)
        stage["attention_k"] = attention_k
        stage["attention_k_ready"] = rope_done
        return True
    except Exception as exc:
        if _truthy_env("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0"):
            print(f"[horizonkv] async compact miss LF-RoPE skipped after failure: {exc}")
        return False


def _run_compact_miss_attention_part(
    kv_cache: dict[str, Any],
    *,
    stage: dict[str, Any],
    controls: dict[str, Any],
    q_group: torch.Tensor,
    out_dtype: torch.dtype,
    kv_sparse_config: KVSparseConfig,
    rope_grid_sizes: torch.Tensor,
    rope_freqs: torch.Tensor,
    lf_k_start_frame: int = 0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    current = torch.cuda.current_stream(q_group.device)
    current.wait_event(controls["ready_event"])
    current.wait_event(stage["copy_done"])
    rope_ready = stage.get("attention_k_ready")
    if rope_ready is not None:
        current.wait_event(rope_ready)
    k = stage.get("attention_k")
    if k is None:
        k = _apply_fast_lf_rope_to_row_materialized_k(
            stage["compact_k"],
            controls,
            frame_seqlen=stage["frame_seqlen"],
            rope_grid_sizes=rope_grid_sizes,
            rope_freqs=rope_freqs,
            start_frame=lf_k_start_frame,
        )
        if k is None:
            return None
        stage["attention_k"] = k
    indices = controls["full_row_indices"]
    q = q_group if indices is None else q_group.index_select(0, indices)
    part = _run_packed_attention_part(
        kv_cache,
        q=q,
        k=k,
        v=stage["compact_v"],
        controls=controls,
        config=kv_sparse_config,
        label="overlap_miss",
    )
    for tensor in (stage["compact_k"], stage["compact_v"], k):
        tensor.record_stream(current)
    _mark_overlap_control_workspaces_used(kv_cache, controls, current)
    return None if part is None else _expand_attention_part(q_group, part, indices, out_dtype)


def _schedule_compact_miss_slot_fill(
    *,
    stage: dict[str, Any],
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    frame_seqlen: int,
    gpu_cache_frames: int,
    source_ready_event: torch.cuda.Event | None = None,
) -> bool:
    fill_stream = _shared_cuda_stream(device=gpu_k.device, name="overlap_compact_slot_fill")
    copy_done = stage.get("copy_done", None)
    if fill_stream is None or copy_done is None:
        return False
    expected = int(stage["scatter_head_ids_cpu"].numel())
    if expected <= 0:
        return False
    with torch.cuda.stream(fill_stream):
        ready_event = source_ready_event if source_ready_event is not None else copy_done
        fill_stream.wait_event(ready_event)
        copied = copy_compact_to_gpu_slots_d2d(
            compact_k=stage["compact_k"],
            compact_v=stage["compact_v"],
            gpu_k=gpu_k,
            gpu_v=gpu_v,
            head_ids=stage["scatter_head_ids_cpu"],
            slot_ids=stage["scatter_slot_ids_cpu"],
            src_rows=stage["scatter_src_rows_cpu"],
            src_cols=stage["scatter_src_cols_cpu"],
            cu_seqlens=stage["cu_seqlens_cpu"],
            frame_seqlen=frame_seqlen,
            gpu_cache_frames=gpu_cache_frames,
        )
        slot_fill_done = torch.cuda.Event()
        slot_fill_done.record(fill_stream)
    if int(copied) != expected:
        raise RuntimeError(f"compact miss D2D slot fill copied {int(copied)} of {expected} items")
    stage["compact_k"].record_stream(fill_stream)
    stage["compact_v"].record_stream(fill_stream)
    stage["slot_fill_done"] = slot_fill_done
    stage["slot_fill_scheduled"] = True
    return True

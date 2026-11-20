# Modified for the HorizonKV inference artifact; see NOTICE.
"""Overlap hit attention with miss transfer; expose sparse self-attention."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from .attention import (
    _run_compact_slot_plan_flash_attention_part_from_materialized,
    _run_materialized_slot_plan_flash_attention,
)
from .cache import build_kv_cache_state, reset_kv_cache_state
from .cache_slots import (
    _kv_cache_commit_incremental_tensor_slot_plan_state,
    _kv_cache_queue_compact_stage_slots_commit,
    _kv_cache_slot_state_tensors,
    _kv_cache_wait_pending_slot_metadata_commit,
)
from .config import KVSparseConfig
from .execution import (
    _overlap_attn_enabled_by_env,
    _overlap_flash_split_attention_enabled_by_env,
    _overlap_split_min_miss_items_by_env,
    _resolve_staged_overlap_frame_indices_cpu,
    _shared_cuda_stream,
    _stage_overlap_frame_indices_to_cpu_async,
)
from .lse import _merge_attention_partials
from .materialize import (
    _build_overlap_slot_plan,
    _materialize_compact_slot_plan,
    _prepare_compact_slot_plan_controls,
)
from .miss import (
    _build_compact_miss_stage,
    _run_compact_miss_attention_part,
    _schedule_compact_miss_slot_fill,
    _submit_compact_miss_h2d,
    _submit_compact_miss_lf_rope,
)
from .offload import (
    _kv_cache_copy_frame_tensor_plan_from_cpu_to_slots,
    copy_kv_to_cpu_cache_async,
    flush_deferred_clean_kv_offloads,
    queue_deferred_clean_kv_to_cpu_cache,
    update_gpu_kv_cache_frames,
)
from .selection import (
    _mandatory_kv_frame_indices,
    _overlay_live_frame_stats,
    _preceding_kv_frame_indices,
    _recent_kv_frame_indices,
    _select_kv_frame_indices_max_kv_frames_batched_fused,
    _sink_frame_indices,
)
from .sparse_kernels import compacted_rope_apply
from .transfer import plan_kv_frame_slots_tensor

__all__ = [
    "build_kv_cache_state",
    "reset_kv_cache_state",
    "copy_kv_to_cpu_cache_async",
    "queue_deferred_clean_kv_to_cpu_cache",
    "update_gpu_kv_cache_frames",
    "sparse_kv_self_attention",
]


def _identity_slot_indices_if_capacity_resident(
    kv_cache: dict[str, Any],
    *,
    frame_indices: torch.Tensor,
    frame_seqlen: int,
    gpu_cache_frames: int,
    live_start_index: int | None,
    live_len: int,
) -> torch.Tensor | None:
    if (
        not isinstance(frame_indices, torch.Tensor)
        or frame_indices.device.type != "cuda"
        or frame_indices.dtype != torch.long
        or (frame_indices.ndim != 2)
        or (int(frame_indices.shape[0]) <= 0)
        or (int(frame_indices.shape[1]) <= 0)
        or (int(frame_seqlen) <= 0)
        or (int(gpu_cache_frames) <= 0)
        or (live_start_index is None)
        or (int(live_len) <= 0)
    ):
        return None
    live_start = int(live_start_index)
    live_len = int(live_len)
    frame_seqlen = int(frame_seqlen)
    if live_start < 0 or live_start % frame_seqlen != 0 or live_len % frame_seqlen != 0:
        return None
    history_frames = live_start // frame_seqlen
    if history_frames > int(gpu_cache_frames):
        return None
    identity_filled = kv_cache.get("gpu_identity_filled_frames", None)
    if not isinstance(identity_filled, int) or int(identity_filled) < history_frames:
        return None
    frame_indices = frame_indices.contiguous()
    live_first = history_frames
    live_last = live_first + live_len // frame_seqlen
    live_mask = frame_indices.ge(0) & frame_indices.ge(live_first) & frame_indices.lt(live_last)
    return torch.where(
        live_mask, torch.full((), -2, dtype=torch.long, device=frame_indices.device), frame_indices
    ).contiguous()


@dataclass
class _OverlapExecution:
    """One attention invocation; all rows share the same cache and RoPE policy."""

    kv_cache: dict[str, Any]
    q: torch.Tensor
    frames: torch.Tensor
    frame_seqlen: int
    live_k: torch.Tensor
    live_v: torch.Tensor
    live_start: int
    out_dtype: torch.dtype
    config: KVSparseConfig
    rope_grid: torch.Tensor
    rope_freqs: torch.Tensor
    rope_start: int

    def __post_init__(self):
        self.gpu_k = self.kv_cache["gpu_k"]
        self.gpu_v = self.kv_cache["gpu_v"]
        self.device = self.gpu_k.device
        self.batch, self.heads, self.total_tokens, self.dim = self.kv_cache["cpu_k"].shape
        self.capacity = self.kv_cache["gpu_cache_frames"]
        self.q = self.q.contiguous()
        self.miss_copy_done = None
        self.controls_ready = None

    def _attention(self, slots):
        return _run_materialized_slot_plan_flash_attention(
            self.kv_cache,
            q_group=self.q,
            frame_indices=self.frames,
            slot_indices=slots,
            frame_seqlen=self.frame_seqlen,
            live_k=self.live_k,
            live_v=self.live_v,
            live_start_index=self.live_start,
            out_dtype=self.out_dtype,
            kv_sparse_config=self.config,
            rope_grid_sizes=self.rope_grid,
            rope_freqs=self.rope_freqs,
            lf_k_start_frame=self.rope_start,
        )

    def _controls(self, row_plan, name):
        stream = _shared_cuda_stream(device=self.device, name=f"overlap_flash_{name}_control_h2d")
        with torch.cuda.stream(stream):
            controls = _prepare_compact_slot_plan_controls(
                self.kv_cache,
                device=self.device,
                row_plan=row_plan,
                frame_seqlen=self.frame_seqlen,
                workspace_name=name,
            )
            ready = torch.cuda.Event()
            ready.record(stream)
        controls["ready_event"] = ready
        return controls

    def _split_attention(self, plan, raw_plan):
        """Enqueue resident attention before miss H2D, then merge using LSE."""
        hit_rows, miss_rows = plan.hit_row_plan, plan.miss_row_plan
        if hit_rows is None or miss_rows is None:
            return None
        hit_controls = self._controls(hit_rows, "hit")
        self.controls_ready = hit_controls["ready_event"]
        stage = _build_compact_miss_stage(
            miss_row_plan=miss_rows,
            compact_scatter_plan=raw_plan.get("compact_scatter_plan"),
            B=self.batch,
            D=self.dim,
            frame_seqlen=self.frame_seqlen,
            device=self.device,
            dtype=self.gpu_k.dtype,
        )
        if stage is None:
            return None
        miss_controls = self._controls(miss_rows, "miss")
        onload_stream = _shared_cuda_stream(device=self.device, name="overlap_compact_onload")
        rope_stream = _shared_cuda_stream(device=self.device, name="overlap_compact_miss_rope")
        producer = torch.cuda.Event()
        producer.record(torch.cuda.current_stream(self.device))

        materialized = _materialize_compact_slot_plan(
            self.kv_cache,
            hit_rows,
            hit_controls,
            frame_seqlen=self.frame_seqlen,
            live_k=self.live_k,
            live_v=self.live_v,
            live_start_index=self.live_start,
        )
        if materialized is None:
            return None
        hit = _run_compact_slot_plan_flash_attention_part_from_materialized(
            self.kv_cache,
            q_group=self.q,
            materialized_part=materialized,
            out_dtype=self.out_dtype,
            kv_sparse_config=self.config,
            rope_grid_sizes=self.rope_grid,
            rope_freqs=self.rope_freqs,
            lf_k_start_frame=self.rope_start,
        )
        if hit is None:
            return None
        submitted = _submit_compact_miss_h2d(
            self.kv_cache,
            stage=stage,
            cpu_k=self.kv_cache["cpu_k"],
            cpu_v=self.kv_cache["cpu_v"],
            frame_seqlen=self.frame_seqlen,
            stream=onload_stream,
        )
        self.miss_copy_done = stage.get("copy_done")
        if not submitted:
            return None
        _submit_compact_miss_lf_rope(
            self.kv_cache,
            stage=stage,
            controls=miss_controls,
            rope_grid_sizes=self.rope_grid,
            rope_freqs=self.rope_freqs,
            stream=rope_stream,
            producer_event=producer,
            lf_k_start_frame=self.rope_start,
        )
        miss = _run_compact_miss_attention_part(
            self.kv_cache,
            stage=stage,
            controls=miss_controls,
            q_group=self.q,
            out_dtype=self.out_dtype,
            kv_sparse_config=self.config,
            rope_grid_sizes=self.rope_grid,
            rope_freqs=self.rope_freqs,
            lf_k_start_frame=self.rope_start,
        )
        if miss is None:
            return None
        read_done = torch.cuda.Event()
        read_done.record(torch.cuda.current_stream(self.device))
        if _schedule_compact_miss_slot_fill(
            stage=stage,
            gpu_k=self.gpu_k,
            gpu_v=self.gpu_v,
            frame_seqlen=self.frame_seqlen,
            gpu_cache_frames=self.capacity,
            source_ready_event=read_done,
        ):
            _kv_cache_queue_compact_stage_slots_commit(
                self.kv_cache,
                stage,
                frame_seqlen=self.frame_seqlen,
                total_tokens=self.total_tokens,
                ready_event=stage["slot_fill_done"],
                device=self.device,
            )
        result = _merge_attention_partials(*hit, *miss, out_dtype=self.out_dtype)
        flush_deferred_clean_kv_offloads(self.kv_cache)
        return result

    def _onload_and_attend(self, frames_cpu, plan, raw_plan):
        """The no-overlap path also handles all-hit/all-miss selections."""
        stream = _shared_cuda_stream(device=self.device, name="overlap_onload")
        current = torch.cuda.current_stream(self.device)
        if self.miss_copy_done is not None:
            stream.wait_event(self.miss_copy_done)
        if self.controls_ready is not None:
            stream.wait_event(self.controls_ready)
        if plan.fill_count:
            copied = _kv_cache_copy_frame_tensor_plan_from_cpu_to_slots(
                self.kv_cache,
                frame_indices_cpu=frames_cpu,
                slot_plan_cpu=plan.slot_plan_cpu,
                copy_plan_cpu=plan.copy_plan_cpu,
                expected_copies=plan.fill_count,
                frame_seqlen=self.frame_seqlen,
                stream=stream,
            )
            if not copied:
                return None
            done = torch.cuda.Event()
            done.record(stream)
            current.wait_event(done)
        result = self._attention(plan.slot_plan_cpu)
        if result is not None:
            _kv_cache_commit_incremental_tensor_slot_plan_state(
                self.kv_cache,
                raw_plan,
                raw_plan["gpu_clock"],
                frame_seqlen=self.frame_seqlen,
                total_tokens=self.total_tokens,
            )
        return result

    def run(self, staged_frames):
        slots = _identity_slot_indices_if_capacity_resident(
            self.kv_cache,
            frame_indices=self.frames,
            frame_seqlen=self.frame_seqlen,
            gpu_cache_frames=self.capacity,
            live_start_index=self.live_start,
            live_len=self.live_k.shape[1],
        )
        if slots is not None:
            result = self._attention(slots)
            if result is not None:
                return result
        if staged_frames is None:
            staged_frames = _stage_overlap_frame_indices_to_cpu_async(self.kv_cache, self.frames)
        frames_cpu = _resolve_staged_overlap_frame_indices_cpu(staged_frames)
        if frames_cpu is None:
            frames_cpu = self.frames.cpu()
        lookup, ids, lens, used = _kv_cache_slot_state_tensors(self.kv_cache)
        raw_plan = plan_kv_frame_slots_tensor(
            frame_indices=frames_cpu,
            frame_to_slot=lookup,
            slot_frame_ids=ids,
            slot_frame_lens=lens,
            slot_last_used=used,
            total_tokens=self.total_tokens,
            frame_seqlen=self.frame_seqlen,
            num_heads=self.heads,
            gpu_cache_frames=self.capacity,
            live_start_index=self.live_start,
            live_len=self.live_k.shape[1],
            has_live=True,
            gpu_clock=self.kv_cache["gpu_clock"],
            build_compact_plans=_overlap_flash_split_attention_enabled_by_env(),
            split_min_miss_items=_overlap_split_min_miss_items_by_env(),
        )
        if raw_plan is None:
            return None
        plan = _build_overlap_slot_plan(plan_result=raw_plan)
        if plan.hit_row_plan is not None and plan.miss_row_plan is not None:
            result = self._split_attention(plan, raw_plan)
            if result is not None:
                return result
        return self._onload_and_attend(frames_cpu, plan, raw_plan)


def sparse_kv_self_attention(
    *,
    roped_query: torch.Tensor,
    kv_cache: dict[str, Any],
    frame_seqlen: int,
    kv_start: int,
    sink_tokens: int,
    current_start: int,
    current_end: int,
    local_end_index: int,
    num_frame_per_block: int,
    layer_idx: int,
    kv_sparse_config: KVSparseConfig,
    out_dtype: torch.dtype,
    rope_freqs: torch.Tensor | None = None,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_frame_k_mean: torch.Tensor | None = None,
    live_start_index: int | None = None,
    rope_grid_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    _kv_cache_wait_pending_slot_metadata_commit(kv_cache)
    B, Lq, H, D = roped_query.shape
    cfg = kv_sparse_config
    Lk = int(local_end_index)
    if Lk % frame_seqlen:
        raise ValueError("HorizonKV attention requires complete KV frames")
    n_frames = Lk // frame_seqlen
    sink_frames = _sink_frame_indices(Lk, kv_start, frame_seqlen, sink_tokens)
    forced_frames = _mandatory_kv_frame_indices(
        Lk, frame_seqlen, kv_start, local_end_index, current_start, current_end, num_frame_per_block
    )
    forced_frames.update(
        _preceding_kv_frame_indices(
            Lk,
            frame_seqlen,
            kv_start,
            local_end_index,
            current_start,
            current_end,
            num_frame_per_block,
            None,
        )
    )
    forced_frames.update(
        _recent_kv_frame_indices(Lk, frame_seqlen, getattr(cfg, "mandatory_recent_kv_frames", None))
    )
    compulsory_frames = set(sink_frames) | set(forced_frames)
    adaptive_metadata = None
    frame_k_index_window = _overlay_live_frame_stats(
        kv_cache.get("frame_k_mean", None),
        live_frame_k_mean,
        live_start_index=live_start_index,
        frame_seqlen=frame_seqlen,
        n_frames=n_frames,
    )
    if os.getenv("HORIZONKV_ADAPTIVE_CONFIG"):
        from .rasa import select_adaptive_frames_if_supported

        adaptive_metadata = select_adaptive_frames_if_supported(
            kv_cache=kv_cache,
            cfg=cfg,
            q=roped_query,
            frame_means=frame_k_index_window,
            compulsory_frames=compulsory_frames,
            frame_seqlen=frame_seqlen,
            n_frames=n_frames,
            live_start_index=live_start_index,
            live_k=live_k,
        )
    if adaptive_metadata is not None:
        cfg = adaptive_metadata["cfg"]
        selected_frame_idx = adaptive_metadata["selected"]
    else:
        selected_frame_idx = _select_kv_frame_indices_max_kv_frames_batched_fused(
            kv_cache,
            roped_query,
            frame_k_index_window,
            n_frames=n_frames,
            frame_seqlen=frame_seqlen,
            max_kv_frames=cfg.max_kv_frames,
            compulsory_frames=compulsory_frames,
        )
    if selected_frame_idx is None:
        raise RuntimeError("HorizonKV pooled-QK frame selection failed")
    staged_frame_indices = None
    if (
        _overlap_attn_enabled_by_env()
        and live_start_index is not None
        and (int(live_start_index) > int(kv_cache.get("gpu_cache_frames", 0) or 0) * frame_seqlen)
    ):
        if adaptive_metadata is not None:
            staged_frame_indices = adaptive_metadata["staged"]
        else:
            staged_frame_indices = _stage_overlap_frame_indices_to_cpu_async(
                kv_cache, selected_frame_idx
            )
    if rope_grid_sizes is None or rope_freqs is None:
        raise ValueError("fast LF-RoPE requires rope_grid_sizes and rope_freqs")
    selected_frames = int(selected_frame_idx.shape[1])
    query_frames = max(1, int(math.ceil(Lq / float(frame_seqlen))))
    current_start_frame = int(current_start) // int(frame_seqlen)
    lf_k_start_frame = max(0, current_start_frame - max(0, selected_frames - query_frames))
    q_for_attention = compacted_rope_apply(
        roped_query, rope_grid_sizes, rope_freqs, start_frame=current_start_frame
    ).to(dtype=live_v.dtype if isinstance(live_v, torch.Tensor) else roped_query.dtype)
    q_attn = q_for_attention.permute(0, 2, 1, 3).reshape(B * H, Lq, 1, D)
    overlap_result = _OverlapExecution(
        kv_cache=kv_cache,
        q=q_attn,
        frames=selected_frame_idx,
        frame_seqlen=frame_seqlen,
        live_k=live_k,
        live_v=live_v,
        live_start=live_start_index,
        out_dtype=out_dtype,
        config=cfg,
        rope_grid=rope_grid_sizes,
        rope_freqs=rope_freqs,
        rope_start=lf_k_start_frame,
    ).run(staged_frame_indices)
    if overlap_result is not None:
        return overlap_result.reshape(B, H, Lq, D).permute(0, 2, 1, 3)
    raise RuntimeError(
        "HorizonKV materialized attention failed; check the sparse and transfer extensions"
    )

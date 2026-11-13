# Modified for the HorizonKV inference artifact; see NOTICE.
"""Materialized attention and packed hit/miss partial outputs."""
from __future__ import annotations

from typing import Any

import flash_attn
import torch

from .config import KVSparseConfig
from .execution import _all_head_indices_device, _truthy_env
from .materialize import (
    _apply_fast_lf_rope_to_compact_k,
    _apply_fast_lf_rope_to_row_materialized_k,
    _mark_overlap_control_workspaces_used,
)
from .offload import flush_deferred_clean_kv_offloads
from .transfer import fused_kv_frame_materialize
from .triton_kernels import blocksparse_attention_pooled_triton


class SparseAttentionError(RuntimeError):
    pass


@torch.no_grad()
def _run_materialized_slot_plan_flash_attention(
    kv_cache: dict[str, Any],
    *,
    q_group: torch.Tensor,
    frame_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    frame_seqlen: int,
    live_k: torch.Tensor,
    live_v: torch.Tensor,
    live_start_index: int,
    out_dtype: torch.dtype,
    kv_sparse_config: KVSparseConfig,
    rope_grid_sizes: torch.Tensor,
    rope_freqs: torch.Tensor,
    lf_k_start_frame: int = 0,
) -> torch.Tensor | None:
    gpu_k, gpu_v = kv_cache["gpu_k"], kv_cache["gpu_v"]
    _, num_heads, total_tokens, dim = kv_cache["cpu_k"].shape
    device = gpu_k.device
    rows = q_group.shape[0]
    selected_frames = frame_indices.shape[1]
    token_count = selected_frames * frame_seqlen
    try:
        frame_indices = frame_indices.to(device=device, non_blocking=True).contiguous()
        slot_indices = slot_indices.to(device=device, non_blocking=True).contiguous()
        k_compact = torch.empty((rows, token_count, 1, dim), device=device, dtype=gpu_k.dtype)
        v_compact = torch.empty_like(k_compact)
        if not fused_kv_frame_materialize(
            gpu_k=gpu_k,
            gpu_v=gpu_v,
            k_out=k_compact,
            v_out=v_compact,
            head_indices=_all_head_indices_device(num_heads, device),
            frame_indices=frame_indices,
            slot_indices=slot_indices,
            frame_seqlen=frame_seqlen,
            total_tokens=total_tokens,
            num_heads=num_heads,
            gpu_cache_frames=kv_cache["gpu_cache_frames"],
            live_k=live_k,
            live_v=live_v,
            live_start_index=live_start_index,
        ):
            return None
        k_compact = _apply_fast_lf_rope_to_compact_k(
            k_compact,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            selected_frames=selected_frames,
            rope_grid_sizes=rope_grid_sizes,
            rope_freqs=rope_freqs,
            start_frame=lf_k_start_frame,
        )
        if k_compact is None:
            return None
        flush_deferred_clean_kv_offloads(kv_cache)
        sparsity = kv_sparse_config.overlap_block_sparsity
        if sparsity > 0:
            out = _sparse_attention(
                q=q_group,
                k=k_compact,
                v=v_compact,
                sparsity=sparsity,
                label="materialized",
                return_lse=False,
            )
        else:
            out = flash_attn.flash_attn_func(q_group, k_compact, v_compact)
        return out.to(out_dtype)
    except SparseAttentionError:
        raise
    except Exception as exc:
        if _truthy_env("HORIZONKV_FUSED_ONLOAD_VERBOSE"):
            print(f"[horizonkv] materialized attention failed: {exc}")
        return None


def _sparse_attention(*, q, k, v, sparsity, label, return_lse=True, **packed_controls):
    """A configured sparse path must never silently fall back to dense attention."""
    failure = f"{label}: sparse attention failed (sparsity={sparsity}, q={tuple(q.shape)}, k={tuple(k.shape)}); refusing dense fallback"
    try:
        result = blocksparse_attention_pooled_triton(
            q=q,
            k=k,
            v=v,
            sparsity=sparsity,
            return_lse=return_lse,
            **packed_controls,
        )
    except Exception as exc:
        raise SparseAttentionError(failure) from exc
    if result is None:
        raise SparseAttentionError(failure)
    return result


def _run_packed_attention_part(
    kv_cache: dict[str, Any],
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    controls: dict[str, Any],
    config: KVSparseConfig,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The planner supplies packed KV, int32 lengths and cumulative lengths on the GPU."""
    token_count = controls["token_count"]
    sparsity = config.overlap_block_sparsity
    if sparsity > 0:
        if controls["all_full_length"]:
            k = k.view(q.shape[0], token_count, 1, k.shape[-1])
            v = v.view_as(k)
            packed_controls = {}
        else:
            packed_controls = dict(
                k_lens=controls["k_lens"],
                cu_seqlens_k=controls["cu_seqlens_k"],
                max_seqlen_k=token_count,
            )
        return _sparse_attention(
            q=q,
            k=k,
            v=v,
            sparsity=sparsity,
            label=label,
            **packed_controls,
        )
    rows, query_tokens, query_heads, _ = q.shape
    cache = kv_cache.setdefault("_kv_sparse_flash_varlen_q_controls", {})
    key = (q.device, rows, query_tokens)
    if key not in cache:
        cache[key] = torch.arange(rows + 1, dtype=torch.int32, device=q.device) * query_tokens
    try:
        out, lse, _ = flash_attn.flash_attn_varlen_func(
            q=q.flatten(0, 1),
            k=k,
            v=v,
            cu_seqlens_q=cache[key],
            cu_seqlens_k=controls["cu_seqlens_k"],
            max_seqlen_q=query_tokens,
            max_seqlen_k=token_count,
            dropout_p=0.0,
            causal=False,
            return_attn_probs=True,
        )
        out = out.unflatten(0, (rows, query_tokens))
        lse = (
            lse.transpose(0, 1).contiguous().view(rows, query_tokens, query_heads).permute(0, 2, 1)
        )
        return out, lse
    except Exception:
        return None


def _expand_attention_part(q, part, row_indices, out_dtype):
    """Absent hit/miss rows contribute zero output and negative infinite LSE."""
    out, lse = part
    out, lse = out.to(out_dtype), lse.float()
    if row_indices is None:
        return out, lse
    full_out = torch.zeros(q.shape, device=q.device, dtype=out_dtype)
    full_lse = torch.full(
        (q.shape[0], q.shape[2], q.shape[1]), -torch.inf, device=q.device, dtype=torch.float32
    )
    full_out.index_copy_(0, row_indices, out)
    full_lse.index_copy_(0, row_indices, lse)
    return full_out, full_lse


def _run_compact_slot_plan_flash_attention_part_from_materialized(
    kv_cache: dict[str, Any],
    *,
    q_group: torch.Tensor,
    materialized_part: dict[str, Any],
    out_dtype: torch.dtype,
    kv_sparse_config: KVSparseConfig,
    rope_grid_sizes: torch.Tensor,
    rope_freqs: torch.Tensor,
    lf_k_start_frame: int = 0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    controls = materialized_part["controls"]
    k = _apply_fast_lf_rope_to_row_materialized_k(
        materialized_part["k_compact"],
        controls,
        frame_seqlen=materialized_part["frame_seqlen"],
        rope_grid_sizes=rope_grid_sizes,
        rope_freqs=rope_freqs,
        inplace=True,
        start_frame=lf_k_start_frame,
    )
    if k is None:
        return None
    indices = controls["full_row_indices"]
    q = q_group if indices is None else q_group.index_select(0, indices)
    part = _run_packed_attention_part(
        kv_cache,
        q=q,
        k=k,
        v=materialized_part["v_compact"],
        controls=controls,
        config=kv_sparse_config,
        label="overlap_hit",
    )
    _mark_overlap_control_workspaces_used(kv_cache, controls, torch.cuda.current_stream(q.device))
    return None if part is None else _expand_attention_part(q_group, part, indices, out_dtype)

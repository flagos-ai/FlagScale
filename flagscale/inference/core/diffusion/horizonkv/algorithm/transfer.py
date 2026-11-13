# Modified for the HorizonKV inference artifact; see NOTICE.
"""CUDA transfer, slot planning and materialization used by the host cache."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


@lru_cache(maxsize=1)
def _load_module() -> Any | None:
    if not torch.cuda.is_available():
        return None
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        return None
    src_dir = Path(__file__).resolve().parent / "csrc"
    build_dir = Path(os.getenv("HORIZONKV_FUSED_ONLOAD_BUILD_DIR", "/tmp/horizonkv_transfer"))
    build_dir.mkdir(parents=True, exist_ok=True)
    verbose = str(os.getenv("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    try:
        return load(
            name="horizonkv_transfer",
            sources=[
                str(src_dir / "kv_sparse_onload.cpp"),
                str(src_dir / "kv_slot_plan.cpp"),
                str(src_dir / "kv_sparse_onload_cuda.cu"),
                str(src_dir / "kv_materialize.cu"),
                str(src_dir / "kv_selection.cu"),
            ],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=verbose,
        )
    except Exception as exc:
        if verbose:
            print(f"[horizonkv] fused onload extension disabled after build failure: {exc}")
        return None


def fused_kv_frame_materialize(
    *,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    head_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    frame_seqlen: int,
    total_tokens: int,
    num_heads: int,
    gpu_cache_frames: int,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_start_index: int = 0,
) -> bool:
    module = _load_module()
    if module is None:
        return False
    device = k_out.device
    empty_live = torch.empty(0, device=device, dtype=k_out.dtype)
    has_live = live_k is not None and live_v is not None
    module.kv_sparse_frame_materialize(
        gpu_k,
        gpu_v,
        k_out,
        v_out,
        head_indices,
        frame_indices,
        slot_indices,
        live_k if has_live else empty_live,
        live_v if has_live else empty_live,
        int(frame_seqlen),
        int(total_tokens),
        int(num_heads),
        int(gpu_cache_frames),
        int(live_start_index),
        bool(has_live),
    )
    return True


def fused_kv_frame_materialize_varlen(
    *,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    head_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    frame_seqlen: int,
    total_tokens: int,
    num_heads: int,
    gpu_cache_frames: int,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_start_index: int = 0,
) -> bool:
    module = _load_module()
    if module is None:
        return False
    device = k_out.device
    empty_live = torch.empty(0, device=device, dtype=k_out.dtype)
    has_live = live_k is not None and live_v is not None
    module.kv_sparse_frame_materialize_varlen(
        gpu_k,
        gpu_v,
        k_out,
        v_out,
        head_indices,
        frame_indices,
        slot_indices,
        cu_seqlens_k,
        live_k if has_live else empty_live,
        live_v if has_live else empty_live,
        int(frame_seqlen),
        int(total_tokens),
        int(num_heads),
        int(gpu_cache_frames),
        int(live_start_index),
        bool(has_live),
    )
    return True


def select_pooled_qk_topk(
    *,
    q: torch.Tensor,
    frame_k: torch.Tensor,
    head_indices: torch.Tensor,
    compulsory_indices: torch.Tensor,
    frame_seqlen: int,
    max_kv_frames: int,
) -> torch.Tensor | None:
    module = _load_module()
    if module is None:
        return None
    return module.kv_sparse_select_pooled_qk_topk(
        q, frame_k, head_indices, compulsory_indices, int(frame_seqlen), int(max_kv_frames)
    )


def update_frame_slot_map(
    *,
    frame_to_slot: torch.Tensor,
    head_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    evicted_frame_ids: torch.Tensor,
) -> bool:
    module = _load_module()
    if module is None:
        return False
    module.kv_sparse_update_frame_slot_map(
        frame_to_slot, head_ids, frame_ids, slot_ids, evicted_frame_ids
    )
    return True


def copy_kv_frame_plan_to_gpu_slots(
    *,
    cpu_k: torch.Tensor,
    cpu_v: torch.Tensor,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    frame_indices: torch.Tensor,
    slot_plan: torch.Tensor,
    copy_plan: torch.Tensor,
    frame_seqlen: int,
    gpu_cache_frames: int,
) -> int:
    module = _load_module()
    if module is None:
        return 0
    return int(
        module.kv_sparse_copy_frame_plan_to_slots(
            cpu_k,
            cpu_v,
            gpu_k,
            gpu_v,
            frame_indices,
            slot_plan,
            copy_plan,
            int(frame_seqlen),
            int(gpu_cache_frames),
        )
    )


def copy_kv_frames_to_gpu_compact(
    *,
    cpu_k: torch.Tensor,
    cpu_v: torch.Tensor,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    head_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    frame_seqlen: int,
) -> int:
    module = _load_module()
    if module is None:
        return 0
    return int(
        module.kv_sparse_copy_frames_to_compact(
            cpu_k,
            cpu_v,
            compact_k,
            compact_v,
            head_ids,
            frame_ids,
            cu_seqlens,
            int(frame_seqlen),
        )
    )


def copy_compact_to_gpu_slots_d2d(
    *,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    head_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    src_rows: torch.Tensor,
    src_cols: torch.Tensor,
    cu_seqlens: torch.Tensor,
    frame_seqlen: int,
    gpu_cache_frames: int,
) -> int:
    module = _load_module()
    if module is None:
        return 0
    return int(
        module.kv_sparse_copy_compact_to_slots_d2d(
            compact_k,
            compact_v,
            gpu_k,
            gpu_v,
            head_ids,
            slot_ids,
            src_rows,
            src_cols,
            cu_seqlens,
            int(frame_seqlen),
            int(gpu_cache_frames),
        )
    )


def copy_kv_span_to_cpu_cache(
    *,
    gpu_k_span: torch.Tensor,
    gpu_v_span: torch.Tensor,
    cpu_k: torch.Tensor,
    cpu_v: torch.Tensor,
    start_token: int,
) -> int:
    module = _load_module()
    if module is None:
        return 0
    return int(
        module.kv_sparse_copy_kv_span_to_cpu(gpu_k_span, gpu_v_span, cpu_k, cpu_v, int(start_token))
    )


def plan_kv_frame_slots_tensor(
    *,
    frame_indices: torch.Tensor,
    frame_to_slot: torch.Tensor,
    slot_frame_ids: torch.Tensor,
    slot_frame_lens: torch.Tensor,
    slot_last_used: torch.Tensor,
    total_tokens: int,
    frame_seqlen: int,
    num_heads: int,
    gpu_cache_frames: int,
    live_start_index: int,
    live_len: int,
    has_live: bool,
    gpu_clock: int,
    build_compact_plans: bool = True,
    split_min_miss_items: int = 0,
) -> dict[str, Any] | None:
    module = _load_module()
    if module is None:
        return None
    return module.kv_sparse_plan_slots_tensor(
        frame_indices,
        frame_to_slot,
        slot_frame_ids,
        slot_frame_lens,
        slot_last_used,
        int(total_tokens),
        int(frame_seqlen),
        int(num_heads),
        int(gpu_cache_frames),
        int(live_start_index),
        int(live_len),
        bool(has_live),
        int(gpu_clock),
        bool(build_compact_plans),
        int(split_min_miss_items),
    )

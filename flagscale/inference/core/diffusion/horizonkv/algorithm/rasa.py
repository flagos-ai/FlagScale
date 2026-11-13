# Modified for the HorizonKV inference artifact; see NOTICE.
from copy import copy
from functools import lru_cache
import os
import torch
from .rasa_policy import load_policy


@lru_cache(maxsize=8)
def _settings(path):
    return load_policy(path)


def select_adaptive_frames_if_supported(
    *,
    kv_cache,
    cfg,
    q,
    frame_means,
    compulsory_frames,
    frame_seqlen,
    n_frames,
    live_start_index,
    live_k,
):
    from .selection import _select_kv_frame_indices_max_kv_frames_batched_fused
    from .cache_slots import _kv_cache_slot_state_tensors
    from .execution import (
        _stage_overlap_frame_indices_to_cpu_async,
        _resolve_staged_overlap_frame_indices_cpu,
    )
    from .transfer import _load_module

    policy = _settings(os.environ["HORIZONKV_ADAPTIVE_CONFIG"])
    primary_topk = policy["best_config"]["frames"]
    fallback_topk = policy["min_config"]["frames"]
    primary_sparsity = policy["best_config"]["block_sparsity"]
    fallback_sparsity = policy["min_config"]["block_sparsity"]
    if (
        primary_topk != cfg.max_kv_frames
        or abs(primary_sparsity - cfg.overlap_block_sparsity) > 1e-08
    ):
        raise ValueError("bounded primary must match the generated KV config topk/sparsity")
    if (
        live_start_index is None
        or live_start_index <= int(kv_cache["gpu_cache_frames"]) * frame_seqlen
    ):
        return None
    if not (
        frame_means is not None
        and isinstance(live_k, torch.Tensor)
        and (q.dtype == torch.bfloat16)
        and (q.device.type == "cuda")
    ):
        raise RuntimeError(
            "RASA requires the fast pooled-QK materializer with fixed endpoint block density"
        )
    if len(compulsory_frames) > fallback_topk:
        raise ValueError("bounded fallback topk cannot cover all compulsory frames")
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_adaptive_count_misses"):
        raise RuntimeError("bounded adaptive requires the rebuilt optional onload extension")
    selected = _select_kv_frame_indices_max_kv_frames_batched_fused(
        kv_cache,
        q,
        frame_means,
        n_frames=n_frames,
        frame_seqlen=frame_seqlen,
        max_kv_frames=primary_topk,
        compulsory_frames=compulsory_frames,
    )
    if selected is None:
        raise RuntimeError("bounded primary frame selection failed")
    state = _kv_cache_slot_state_tensors(kv_cache)
    if state is None:
        raise RuntimeError("bounded slot metadata unavailable")
    staged = _stage_overlap_frame_indices_to_cpu_async(kv_cache, selected)
    frames_cpu = _resolve_staged_overlap_frame_indices_cpu(staged)
    if frames_cpu is None:
        raise RuntimeError("bounded primary frame staging failed")
    live_first = int(live_start_index) // frame_seqlen
    live_end = live_first + int(live_k.shape[1]) // frame_seqlen
    misses = int(
        module.kv_sparse_adaptive_count_misses(
            frames_cpu, state[0], state[1], state[2], live_first, live_end, frame_seqlen
        )
    )
    if misses < 0:
        raise RuntimeError("bounded encountered invalid selected frame IDs")
    effective_cfg = cfg
    if misses >= policy["switch_miss_items"]:
        selected_fallback = _select_kv_frame_indices_max_kv_frames_batched_fused(
            kv_cache,
            q,
            frame_means,
            n_frames=n_frames,
            frame_seqlen=frame_seqlen,
            max_kv_frames=fallback_topk,
            compulsory_frames=compulsory_frames,
        )
        if selected_fallback is None:
            raise RuntimeError("bounded fallback frame selection failed")
        selected = selected_fallback
        staged = _stage_overlap_frame_indices_to_cpu_async(kv_cache, selected)
        if staged is None:
            raise RuntimeError("bounded fallback frame staging failed")
        effective_cfg = copy(cfg)
        effective_cfg.max_kv_frames = fallback_topk
        effective_cfg.overlap_block_sparsity = fallback_sparsity
        effective_cfg.block_target_density = 1 - fallback_sparsity
    return {
        "selected": selected,
        "staged": staged,
        "cfg": effective_cfg,
    }

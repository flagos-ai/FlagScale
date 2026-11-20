# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Cache eviction and hit/miss attention checks independent of model weights."""

import pytest
import torch


def make_cache(device="cpu", *, frames=8, capacity=4, heads=2, frame_seqlen=64, dim=64):
    from flagscale.inference.core.diffusion.horizonkv.algorithm.cache import build_kv_cache_state
    from flagscale.inference.core.diffusion.horizonkv.algorithm.config import KVSparseConfig

    return build_kv_cache_state(
        batch_size=1, dtype=torch.bfloat16, device=device,
        frame_capacity=frames, frame_seqlen=frame_seqlen, num_heads=heads, head_dim=dim,
        kv_sparse_config=KVSparseConfig(gpu_cache_frames=capacity),
    )


def test_eviction_clears_old_frame_and_preserves_planner_timestamps():
    from flagscale.inference.core.diffusion.horizonkv.algorithm.cache import reset_kv_cache_state
    from flagscale.inference.core.diffusion.horizonkv.algorithm.cache_slots import (
        _kv_cache_commit_incremental_tensor_slot_plan_state,
    )

    cache = make_cache(frame_seqlen=2, dim=4)
    cache["gpu_frame_to_slot_dense"][0, 0] = 1
    cache["gpu_slot_frame_ids_tensor_cpu"][0, 1] = 0
    cache["gpu_slot_frame_lens_tensor_cpu"][0, 1] = 2
    plan = {"fill_head_ids": torch.tensor([0, 1]), "fill_slot_ids": torch.tensor([1, 2]),
            "fill_frame_ids": torch.tensor([5, 6]), "fill_last_used": torch.tensor([13, 15])}
    assert _kv_cache_commit_incremental_tensor_slot_plan_state(
        cache, plan, 16, frame_seqlen=2, total_tokens=16,
    )
    assert cache["gpu_frame_to_slot_dense"][0, 0] == -1
    assert cache["gpu_frame_to_slot_dense"][0, 5] == 1
    assert cache["gpu_frame_to_slot_dense"][1, 6] == 2
    assert cache["gpu_slot_last_used_tensor_cpu"][0, 1] == 13
    assert cache["gpu_slot_last_used_tensor_cpu"][1, 2] == 15
    assert cache["gpu_clock"] == 16
    reset_kv_cache_state(cache)
    assert cache["gpu_frame_to_slot_dense"].eq(-1).all()
    assert cache["gpu_slot_frame_ids_tensor_cpu"].eq(-1).all()
    assert cache["gpu_slot_frame_lens_tensor_cpu"].eq(0).all()
    assert cache["gpu_clock"] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lse_merge_handles_empty_rows_and_extreme_logits(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from flagscale.inference.core.diffusion.horizonkv.algorithm.lse import _merge_attention_partials

    a = torch.full((3, 2, 1, 8), 2.0, device=device)
    b = torch.full_like(a, 7.0)
    la = torch.tensor([[1000., -1000.], [-torch.inf, -torch.inf], [0., 0.]], device=device)[:, None]
    lb = torch.tensor([[999., -999.], [-torch.inf, -torch.inf], [-torch.inf, -torch.inf]], device=device)[:, None]
    merged = _merge_attention_partials(a, la, b, lb, out_dtype=a.dtype)
    torch.testing.assert_close(merged[0, 0], torch.full_like(a[0, 0], (2 + 7 / torch.e) / (1 + 1 / torch.e)))
    torch.testing.assert_close(merged[0, 1], torch.full_like(a[0, 1], (2 / torch.e + 7) / (1 + 1 / torch.e)))
    assert merged[1].eq(0).all()
    torch.testing.assert_close(merged[2], a[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and HorizonKV extensions")
def test_overlap_matches_dense_attention_and_refills_evicted_slots(monkeypatch):
    from flagscale.inference.core.diffusion.horizonkv.algorithm.cache import reset_kv_cache_state
    from flagscale.inference.core.diffusion.horizonkv.algorithm.cache_slots import (
        _kv_cache_update_slot_metadata_fast, _kv_cache_wait_pending_slot_metadata_commit,
    )
    from flagscale.inference.core.diffusion.horizonkv.algorithm.config import KVSparseConfig
    from flagscale.inference.core.diffusion.horizonkv.algorithm.oae import _OverlapExecution
    from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_kernels import compacted_rope_apply
    from flagscale.inference.core.diffusion.horizonkv.wan.model import rope_params

    torch.manual_seed(31)
    frame_length, dim, heads = 64, 64, 2
    keys = torch.randn(1, 8 * frame_length, heads, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    query = torch.randn(2, 2 * frame_length, 1, dim, device="cuda", dtype=torch.bfloat16)
    # The first head misses frame 4; the second head is entirely resident/live.
    # This exercises unequal packed hit rows and an absent miss row.
    frames = torch.tensor([[0, 4, 6, 7], [1, 2, 6, 7]], device="cuda")
    grid = torch.tensor([[2, 8, 8]])
    freqs = rope_params(128, dim).cuda()
    selected_keys, selected_values = [], []
    for head in range(heads):
        ids = frames[head].repeat_interleave(frame_length) * frame_length
        ids += torch.arange(frame_length, device="cuda").repeat(4)
        k = keys[:, ids, head:head + 1]
        k = compacted_rope_apply(k, torch.tensor([[4, 8, 8]]), freqs)
        selected_keys.append(k)
        selected_values.append(values[:, ids, head:head + 1])
    k = torch.cat(selected_keys)
    v = torch.cat(selected_values)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
    ).transpose(1, 2)

    outputs = []
    for overlap in (False, True):
        monkeypatch.setenv("HORIZONKV_OVERLAP_ATTN", str(int(overlap)))
        monkeypatch.setenv("HORIZONKV_OVERLAP_FLASH_SPLIT_ATTN", str(int(overlap)))
        monkeypatch.setenv("HORIZONKV_OVERLAP_SPLIT_MIN_MISS_ITEMS", "1")
        cache = make_cache("cuda")
        cache["cpu_k"].copy_(keys.permute(0, 2, 1, 3).cpu())
        cache["cpu_v"].copy_(values.permute(0, 2, 1, 3).cpu())
        cache["gpu_k_slots"].copy_(keys[:, :4 * frame_length].reshape(1, 4, frame_length, heads, dim).permute(0, 3, 1, 2, 4))
        cache["gpu_v_slots"].copy_(values[:, :4 * frame_length].reshape(1, 4, frame_length, heads, dim).permute(0, 3, 1, 2, 4))
        _kv_cache_update_slot_metadata_fast(
            cache, head_ids=torch.arange(heads).repeat_interleave(4),
            slot_ids=torch.arange(4).repeat(heads), frame_ids=torch.arange(4).repeat(heads),
            frame_seqlen=frame_length, total_tokens=8 * frame_length,
        )
        execution = _OverlapExecution(
            cache, query, frames, frame_length, keys[:, 6 * frame_length:].contiguous(),
            values[:, 6 * frame_length:].contiguous(), 6 * frame_length, torch.bfloat16,
            KVSparseConfig(max_kv_frames=4, gpu_cache_frames=4, overlap_block_sparsity=0),
            grid, freqs, 0,
        )
        result = execution.run(None)
        assert result is not None
        torch.testing.assert_close(result.float(), expected, rtol=0.02, atol=0.004)
        _kv_cache_wait_pending_slot_metadata_commit(cache)
        slot = int(cache["gpu_frame_to_slot_dense"][0, 4])
        assert 0 <= slot < 4
        torch.testing.assert_close(cache["gpu_k_slots"][0, 0, slot], keys[0, 4 * frame_length:5 * frame_length, 0])
        outputs.append(result)
        reset_kv_cache_state(cache)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.02, atol=0.004)

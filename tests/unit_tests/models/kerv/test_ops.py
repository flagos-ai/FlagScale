# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys

import pytest
import torch

from flagscale.models.kerv.ops import (
    bundled_runtime_path,
    configure_kerv_ops,
    kerv_action_projection_select,
    kerv_action_verify_accept,
    kerv_add_rms_norm,
    kerv_down_proj_residual_rms_norm,
    kerv_draft_action_topk,
    kerv_kv_accept_commit,
    kerv_kv_commit,
    kerv_logical_kv_commit,
    kerv_o_proj_residual_rms_norm,
    kerv_rope_kv_store,
    kerv_silu_mul,
    kerv_static_tree_pack,
    kerv_tree_embed_pack,
    kerv_value_cache_store,
    kerv_verify_accept_control,
    kerv_vision_add_layer_norm,
    kerv_vision_bias_gelu,
    static_tree_attention,
    static_tree_attention_reference,
)


def test_complete_runtime_imports_without_triton():
    modules = (
        "adaptive_linear_fusion",
        "adaptive_rms_norm",
        "adaptive_rotary_fusion",
        "embodied_ops",
        "fused_logsoftmax_topk",
        "rotary_cache",
        "tree_attention_mask",
    )
    script = (
        "import importlib, sys; "
        "sys.modules['triton'] = None; "
        f"modules = {modules!r}; "
        "[importlib.import_module('KERVRuntimeOptimization.' + name) for name in modules]"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(bundled_runtime_path()), environment.get("PYTHONPATH", ""))
    )

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_public_namespace_and_cpu_fallbacks():
    gate = torch.randn(2, 8)
    up = torch.randn_like(gate)
    expected = torch.nn.functional.silu(gate.clone()) * up
    result = kerv_silu_mul(gate, up)
    torch.testing.assert_close(result, expected)

    hidden = torch.randn(2, 8)
    residual = torch.randn_like(hidden)
    weight = torch.randn(8)
    expected = torch.nn.functional.rms_norm(hidden + residual, (8,), weight, 1e-6)
    result = kerv_add_rms_norm(hidden, residual, weight, 1e-6)
    torch.testing.assert_close(result, expected)

    logits = torch.randn(3, 5, 17)
    candidates = torch.randint(0, 17, (3, 5))
    best, length = kerv_verify_accept_control(logits, candidates, 0.0, 0)
    predicted = logits[:, :-1].argmax(-1)
    accepted = torch.cumprod((candidates[:, 1:] == predicted).int(), dim=1).sum(dim=1)
    assert torch.equal(best, accepted.argmax())
    assert torch.equal(length, accepted.max())

    draft = torch.arange(12).reshape(1, 12)
    kept = torch.tensor([0, 3, 7])
    packed = kerv_static_tree_pack(draft, kept, 5)
    assert packed.tolist() == [[0, 3, 7, 0, 0]]

    hidden = torch.randn(2, 8)
    weight = torch.randn(11, 8)
    expected_ids = torch.nn.functional.linear(hidden, weight).argmax(dim=-1) + 100
    torch.testing.assert_close(kerv_action_projection_select(hidden, weight, 100), expected_ids)

    destination = torch.zeros(1, 2, 8, 4)
    source = torch.ones(1, 2, 3, 4)
    kerv_kv_commit(destination, source, 2)
    assert torch.equal(destination[..., 2:5, :], source)

    cache = torch.zeros(1, 2, 8, 4)
    kerv_value_cache_store(cache, source, 1)
    assert torch.equal(cache[..., 1:4, :], source)


def test_static_tree_attention_cpu_reference():
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)
    ancestors = torch.tensor([[0, 1], [0, 2], [0, 4]])
    result = static_tree_attention(query, key, value, ancestors, 2)
    expected = static_tree_attention_reference(query, key, value, ancestors, 2)
    torch.testing.assert_close(result, expected)


def test_phase2_public_interfaces_cpu_native():
    """Exercise every phase-two schema without requiring CUDA/Triton."""
    configure_kerv_ops(
        enabled=True,
        backend="native",
        include=(
            "kerv_tree_embed_pack,kerv_rope_kv_store,kerv_action_verify_accept,"
            "kerv_draft_action_topk,kerv_kv_accept_commit,kerv_vision_add_layer_norm,"
            "kerv_vision_bias_gelu,kerv_o_proj_residual_rms_norm,"
            "kerv_down_proj_residual_rms_norm,kerv_logical_kv_commit"
        ),
    )

    tokens = torch.tensor([[2, 5, 7, 11]])
    keep = torch.tensor([0, 2])
    embedding = torch.randn(16, 8)
    packed = kerv_tree_embed_pack(tokens, keep, embedding, 3)
    expected_ids = torch.tensor([[2, 7, 2]])
    torch.testing.assert_close(packed, embedding[expected_ids])

    query = torch.randn(1, 2, 3, 128)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cos = torch.randn(1, 3, 128)
    sin = torch.randn_like(cos)
    key_cache = torch.zeros(1, 2, 12, 128)
    value_cache = torch.zeros_like(key_cache)
    rotated = kerv_rope_kv_store(
        query, key, value, cos, sin, torch.empty(0, dtype=torch.long), key_cache, value_cache, 4
    )
    assert rotated.shape == query.shape
    torch.testing.assert_close(value_cache[:, :, 4:7], value)

    hidden = torch.randn(6, 8)
    action_weight = torch.randn(32, 8)
    candidates = torch.randint(0, 32, (2, 3))
    best, length, next_token = kerv_action_verify_accept(hidden, action_weight, candidates, 0.0, 0)
    logits = torch.nn.functional.linear(hidden, action_weight).reshape(2, 3, 32)
    predicted = logits.argmax(-1)
    accepted = torch.cumprod((predicted[:, :-1] == candidates[:, 1:]).to(torch.int32), dim=1).sum(
        -1
    )
    assert torch.equal(best, accepted.argmax())
    assert torch.equal(length, accepted[best])
    assert torch.equal(next_token, predicted[best, length.clamp_max(2)])

    values, indices = kerv_draft_action_topk(hidden, action_weight, 100, 4)
    ref_values, ref_indices = torch.sort(
        torch.nn.functional.log_softmax(logits.reshape(-1, 32)[0].float(), dim=-1),
        dim=-1,
        descending=True,
        stable=True,
    )
    torch.testing.assert_close(values[0], ref_values[:4].to(values.dtype))
    assert torch.equal(indices[0], ref_indices[:4] + 100)

    storage = torch.zeros(2, 2, 1, 2, 16, 4)
    source = torch.randn_like(storage[..., :3, :])
    storage[..., 6:9, :].copy_(source)
    indices = torch.tensor([2, 0, 1])
    expected = storage[..., 6 + indices, :].clone()
    scratch = torch.empty_like(expected)
    kerv_kv_accept_commit(storage, indices, 6, scratch=scratch)
    torch.testing.assert_close(storage[..., 6:9, :], expected)

    x = torch.randn(4, 8)
    residual = torch.randn_like(x)
    norm_weight = torch.randn(8)
    bias = torch.randn(8)
    norm = kerv_vision_add_layer_norm(x, residual, norm_weight, bias, 1e-5)
    expected_norm = torch.nn.functional.layer_norm(x + residual, (8,), norm_weight, bias, 1e-5)
    torch.testing.assert_close(norm, expected_norm)
    gelu = kerv_vision_bias_gelu(x, bias)
    torch.testing.assert_close(gelu, torch.nn.functional.gelu(x + bias, approximate="none"))

    proj = torch.randn(8, 8)
    rms = torch.randn(8)
    for op in (kerv_o_proj_residual_rms_norm, kerv_down_proj_residual_rms_norm):
        value = op(x, proj, residual, rms, 1e-5, bias)
        expected_value = torch.nn.functional.rms_norm(
            torch.nn.functional.linear(x, proj, bias) + residual,
            (8,),
            rms,
            1e-5,
        )
        torch.testing.assert_close(value, expected_value)

    logical = torch.zeros(12, dtype=torch.long)
    result = kerv_logical_kv_commit(logical, torch.tensor([8, 4]), 3)
    assert result.tolist() == [0, 0, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_verify_accept_cuda_matches_aten():
    logits = torch.randn(49, 9, 256, device="cuda", dtype=torch.bfloat16)
    candidates = torch.randint(0, 256, (49, 9), device="cuda")
    result = kerv_verify_accept_control(logits, candidates, 0.0, 0)
    predicted = logits[:, :-1].argmax(-1)
    accepted = torch.cumprod((candidates[:, 1:] == predicted).int(), dim=1).sum(dim=1)
    assert torch.equal(result[0], accepted.argmax())
    assert torch.equal(result[1], accepted.max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_tree_attention_cuda_matches_reference():
    # Build a duplicate-free binary ancestor chain for the tree portion.  The
    # production KERV templates have the same invariant.
    from flagscale.models.kerv.ops import configure_kerv_ops

    configure_kerv_ops(
        enabled=True,
        include="kerv_static_tree_attention",
        backend="triton",
    )
    for depth in (4, 5, 8):
        query = torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        ancestors = torch.full((64, depth), -1, device="cuda", dtype=torch.long)
        for row in range(32, 64):
            chain = []
            node = row
            while node >= 32 and len(chain) < depth:
                chain.append(node)
                node = 32 + (node - 33) // 2 if node >= 33 else -1
            ancestors[row, : len(chain)] = torch.tensor(chain, device="cuda")
        result = static_tree_attention(query, key, value, ancestors, 32)
        expected = static_tree_attention_reference(query, key, value, ancestors, 32)
        torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_tree_attention_fixed_prefix_gap_matches_reference():
    """Unused fixed-prefix slots must never participate in tree attention."""
    configure_kerv_ops(
        enabled=True,
        include="kerv_static_tree_attention",
        backend="triton",
    )
    tree_start, valid_prefix, tree_nodes = 288, 279, 64
    query = torch.randn(1, 2, tree_nodes, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(
        1,
        2,
        tree_start + tree_nodes,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    ancestors = torch.full((tree_nodes, 8), -1, device="cuda", dtype=torch.long)
    for row in range(tree_nodes):
        start = max(0, row - 7)
        nodes = torch.arange(start, row + 1, device="cuda") + tree_start
        ancestors[row, : nodes.numel()] = nodes
    valid_prefix_tensor = torch.tensor(valid_prefix, device="cuda", dtype=torch.int32)
    result = static_tree_attention(
        query,
        key,
        value,
        ancestors,
        tree_start,
        valid_prefix_tokens=valid_prefix_tensor,
    )
    expected = static_tree_attention_reference(
        query,
        key,
        value,
        ancestors,
        tree_start,
        valid_prefix_tokens=valid_prefix_tensor,
    )
    torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)

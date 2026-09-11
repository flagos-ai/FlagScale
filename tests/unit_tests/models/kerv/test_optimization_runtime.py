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

import importlib

import pytest
import torch
import torch.nn as nn

from flagscale.models.kerv.ops import load_embodied_ops

load_embodied_ops()

fuse_transformer_linears = importlib.import_module(
    "KERVRuntimeOptimization.adaptive_linear_fusion"
).fuse_transformer_linears
rms_norm_inference = importlib.import_module(
    "KERVRuntimeOptimization.adaptive_rms_norm"
).rms_norm_inference
fused_rotary_qk = importlib.import_module(
    "KERVRuntimeOptimization.adaptive_rotary_fusion"
).fused_rotary_qk
fused_logsoftmax_topk = importlib.import_module(
    "KERVRuntimeOptimization.fused_logsoftmax_topk"
).fused_logsoftmax_topk
build_tree_causal_mask = importlib.import_module(
    "KERVRuntimeOptimization.tree_attention_mask"
).build_tree_causal_mask


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_adaptive_rms_norm_matches_kerv_rounding():
    value = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    result = rms_norm_inference(value, (4096,), weight, 1e-6)
    inverse_rms = torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (value.float() * inverse_rms).to(value.dtype) * weight

    torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)


def test_adaptive_rotary_matches_reference():
    query = torch.randn(1, 32, 8, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16)
    cosine = torch.randn(1, 1, 64, 128, device="cuda", dtype=torch.bfloat16)
    sine = torch.randn_like(cosine)
    positions = torch.arange(8, device="cuda").unsqueeze(0)

    result_query, result_key = fused_rotary_qk(query, key, cosine, sine, positions)
    selected_cosine = cosine[0, 0].index_select(0, positions[0]).unsqueeze(0).unsqueeze(1)
    selected_sine = sine[0, 0].index_select(0, positions[0]).unsqueeze(0).unsqueeze(1)

    def rotate_half(value):
        half = value.shape[-1] // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    expected_query = query * selected_cosine + rotate_half(query) * selected_sine
    expected_key = key * selected_cosine + rotate_half(key) * selected_sine
    torch.testing.assert_close(result_query, expected_query, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(result_key, expected_key, rtol=2e-2, atol=2e-2)


def test_fused_logsoftmax_topk_matches_pytorch():
    logits = torch.randn(2, 32064, device="cuda", dtype=torch.bfloat16)
    # BF16 random logits contain many ties, and CUDA topk does not guarantee
    # the order of equal elements across separate launches. Keep the tested
    # top-k unique so this comparison checks the operator rather than tie
    # scheduling in two independent native topk calls.
    top_indices = torch.tensor([3, 17, 101, 509, 1021, 4093, 8191, 16381], device="cuda")
    top_values = torch.tensor([32, 30, 28, 26, 24, 22, 20, 18], device="cuda", dtype=logits.dtype)
    logits[:, top_indices] = top_values

    values, indices = fused_logsoftmax_topk(logits, 8)
    expected_values, expected_indices = torch.topk(
        torch.nn.functional.log_softmax(logits, dim=-1), 8, dim=-1
    )

    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(values, expected_values, rtol=2e-2, atol=2e-2)


def test_tree_attention_mask_matches_reference():
    tree_length = 48
    past_length = 19
    tree_mask = torch.tril(torch.ones(tree_length, tree_length, device="cuda", dtype=torch.bool))

    result = build_tree_causal_mask(tree_mask, past_length, torch.bfloat16)
    expected = torch.zeros_like(result)
    expected[..., past_length:] = torch.where(
        tree_mask,
        torch.zeros((), device="cuda", dtype=torch.bfloat16),
        torch.full((), torch.finfo(torch.bfloat16).min, device="cuda", dtype=torch.bfloat16),
    )

    assert torch.equal(result, expected)


def test_packed_qkv_projection_preserves_outputs():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16, bias=True, device="cuda", dtype=torch.bfloat16)
            self.k_proj = nn.Linear(16, 8, bias=True, device="cuda", dtype=torch.bfloat16)
            self.v_proj = nn.Linear(16, 8, bias=True, device="cuda", dtype=torch.bfloat16)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()

    model = Model()
    value = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    expected = tuple(
        projection(value)
        for projection in (
            model.attention.q_proj,
            model.attention.k_proj,
            model.attention.v_proj,
        )
    )

    manifest = fuse_transformer_linears(
        model,
        qkv_rows=(4,),
        qkv_input_sizes=(16,),
    )
    result = tuple(
        projection(value)
        for projection in (
            model.attention.q_proj,
            model.attention.k_proj,
            model.attention.v_proj,
        )
    )

    assert manifest["qkv_group_count"] == 1
    for candidate, reference in zip(result, expected):
        torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-2)

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

"""Microbenchmark the public KERV FlagOS operator namespace.

Example:
    PYTHONPATH=. python examples/kerv/benchmark_ops.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from flagscale.models.kerv.ops import (
    configure_kerv_ops,
    kerv_action_projection_select,
    kerv_action_verify_accept,
    kerv_add_rms_norm,
    kerv_draft_action_topk,
    kerv_kv_accept_commit,
    kerv_kv_commit,
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


def _time_cuda(call, warmup: int, iterations: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)) / iterations)
    return float(statistics.median(samples))


def _time_cpu(call, warmup: int, iterations: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter()
        for _ in range(iterations):
            call()
        samples.append((time.perf_counter() - begin) * 1000.0 / iterations)
    return float(statistics.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="independent timing windows; report the median to suppress GPU jitter",
    )
    parser.add_argument("--output", default="outputs/kerv_embodied_ops_benchmark.json")
    parser.add_argument(
        "--backend",
        choices=("auto", "native", "triton"),
        default="auto",
        help="backend selection; auto keeps unproven tiny paths on ATen",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    configure_kerv_ops(
        enabled=True,
        include=None,
        backend=args.backend,
        record=False,
        strict=True,
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cases: list[dict[str, object]] = []

    gate = torch.randn(8, 11008, device=device, dtype=dtype)
    up = torch.randn_like(gate)
    reference_gate = gate.clone()
    native_silu = lambda: torch.nn.functional.silu(reference_gate.clone()).mul(up)
    fused_silu = lambda: kerv_silu_mul(gate.clone(), up)
    cases.append(_case("kerv_silu_mul", native_silu, fused_silu, device, args))

    hidden = torch.randn(8, 4096, device=device, dtype=dtype)
    residual = torch.randn_like(hidden)
    weight = torch.randn(4096, device=device, dtype=dtype)
    native_rms = lambda: torch.nn.functional.rms_norm(hidden + residual, (4096,), weight, 1e-6)
    fused_rms = lambda: kerv_add_rms_norm(hidden.clone(), residual, weight, 1e-6)
    cases.append(_case("kerv_add_rms_norm", native_rms, fused_rms, device, args))

    logits = torch.randn(49, 9, 256, device=device, dtype=dtype)
    candidates = torch.randint(0, 256, (49, 9), device=device)
    native_verify = lambda: _native_verify(logits, candidates)
    fused_verify = lambda: kerv_verify_accept_control(logits, candidates, 0.0, 0)
    cases.append(_case("kerv_verify_accept_control", native_verify, fused_verify, device, args))

    draft = torch.randint(0, 32000, (1, 320), device=device)
    kept = torch.tensor([0, 3, 7, 15, 28, 42, 55, 63], device=device)
    native_pack = lambda: torch.cat(
        (draft.index_select(1, kept), draft[:, :1].expand(-1, 256 - kept.numel())), dim=1
    )
    fused_pack = lambda: kerv_static_tree_pack(draft, kept, 256)
    cases.append(_case("kerv_static_tree_pack", native_pack, fused_pack, device, args))

    hidden = torch.randn(8, 4096, device=device, dtype=dtype)
    action_weight = torch.randn(256, 4096, device=device, dtype=dtype)
    native_action = lambda: torch.nn.functional.linear(hidden, action_weight).argmax(-1) + 31744
    fused_action = lambda: kerv_action_projection_select(hidden, action_weight, 31744)
    cases.append(_case("kerv_action_projection_select", native_action, fused_action, device, args))

    # Phase-two action path: keep the verifier control outputs integer-exact.
    verify_hidden = torch.randn(49 * 9, 4096, device=device, dtype=dtype)
    verify_weight = torch.randn(256, 4096, device=device, dtype=dtype)
    verify_candidates = torch.randint(0, 256, (49, 9), device=device)
    native_action_verify = lambda: _native_action_verify(
        verify_hidden, verify_weight, verify_candidates
    )
    fused_action_verify = lambda: kerv_action_verify_accept(
        verify_hidden, verify_weight, verify_candidates, 0.0, 0
    )
    cases.append(
        _case("kerv_action_verify_accept", native_action_verify, fused_action_verify, device, args)
    )

    draft_hidden = torch.randn(8, 1152, device=device, dtype=dtype)
    draft_weight = torch.randn(256, 1152, device=device, dtype=dtype)
    native_draft_topk = lambda: _native_draft_topk(draft_hidden, draft_weight, 8, 0)
    fused_draft_topk = lambda: kerv_draft_action_topk(draft_hidden, draft_weight, 0, 8)
    cases.append(_case("kerv_draft_action_topk", native_draft_topk, fused_draft_topk, device, args))

    # Embedding pack uses a compact benchmark vocabulary while retaining the
    # real KERV target width and padding behavior.
    tree_source = torch.randint(0, 2048, (1, 320), device=device)
    tree_kept = torch.tensor([0, 3, 7, 15, 28, 42, 55, 63], device=device)
    tree_embedding = torch.randn(2048, 1152, device=device, dtype=dtype)
    native_tree_embed = lambda: _native_tree_embed(tree_source, tree_kept, tree_embedding, 256)
    fused_tree_embed = lambda: kerv_tree_embed_pack(tree_source, tree_kept, tree_embedding, 256)
    cases.append(_case("kerv_tree_embed_pack", native_tree_embed, fused_tree_embed, device, args))

    query = torch.randn(1, 4, 8, 128, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cos = torch.randn(1, 8, 128, device=device, dtype=dtype)
    sin = torch.randn_like(cos)
    empty_positions = torch.empty(0, dtype=torch.long, device=device)
    native_rope = lambda: _native_rope_store(
        query, key, value, cos, sin, empty_positions, 16, device, dtype
    )
    fused_rope = lambda: _fused_rope_store(query, key, value, cos, sin, empty_positions, 16)
    cases.append(_case("kerv_rope_kv_store", native_rope, fused_rope, device, args))

    source = torch.randn(1, 4, 8, 128, device=device, dtype=dtype)
    native_value_cache = lambda: _native_cache_store(
        torch.zeros(1, 4, 64, 128, device=device, dtype=dtype), source, 16
    )
    fused_value_cache = lambda: kerv_value_cache_store(
        torch.zeros(1, 4, 64, 128, device=device, dtype=dtype), source, 16
    )
    cases.append(
        _case("kerv_value_cache_store", native_value_cache, fused_value_cache, device, args)
    )

    native_commit = lambda: _native_cache_store(
        torch.zeros(1, 4, 64, 128, device=device, dtype=dtype), source, 16
    )
    fused_commit = lambda: kerv_kv_commit(
        torch.zeros(1, 4, 64, 128, device=device, dtype=dtype), source, 16
    )
    cases.append(_case("kerv_kv_commit", native_commit, fused_commit, device, args))

    commit_storage = torch.zeros(2, 2, 1, 4, 64, 128, device=device, dtype=dtype)
    commit_indices = torch.tensor([9, 4, 6, 2], device=device)
    commit_storage[..., 24 + commit_indices, :] = torch.randn(
        2, 2, 1, 4, 4, 128, device=device, dtype=dtype
    )
    native_accept_commit = lambda: _native_accept_commit(commit_storage.clone(), commit_indices, 24)
    fused_accept_commit = lambda: kerv_kv_accept_commit(commit_storage.clone(), commit_indices, 24)
    cases.append(
        _case("kerv_kv_accept_commit", native_accept_commit, fused_accept_commit, device, args)
    )

    vision_hidden = torch.randn(102, 1024, device=device, dtype=dtype)
    vision_residual = torch.randn_like(vision_hidden)
    vision_weight = torch.randn(1024, device=device, dtype=dtype)
    vision_bias = torch.randn(1024, device=device, dtype=dtype)
    native_vision_norm = lambda: torch.nn.functional.layer_norm(
        vision_hidden + vision_residual, (1024,), vision_weight, vision_bias, 1e-5
    )
    fused_vision_norm = lambda: kerv_vision_add_layer_norm(
        vision_hidden, vision_residual, vision_weight, vision_bias, 1e-5
    )
    cases.append(
        _case("kerv_vision_add_layer_norm", native_vision_norm, fused_vision_norm, device, args)
    )
    native_vision_gelu = lambda: torch.nn.functional.gelu(
        vision_hidden + vision_bias, approximate="none"
    )
    fused_vision_gelu = lambda: kerv_vision_bias_gelu(vision_hidden, vision_bias)
    cases.append(
        _case("kerv_vision_bias_gelu", native_vision_gelu, fused_vision_gelu, device, args)
    )

    query = torch.randn(1, 4, 64, 128, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    ancestors = torch.full((64, 5), -1, device=device, dtype=torch.long)
    ancestors[32:, 0] = torch.arange(32, 64, device=device)
    ancestors[33:, 1] = 32 + (torch.arange(33, 64, device=device) - 33) // 2
    native_attention = lambda: static_tree_attention_reference(query, key, value, ancestors, 32)
    fused_attention = lambda: static_tree_attention(query, key, value, ancestors, 32)
    cases.append(
        _case("kerv_static_tree_attention", native_attention, fused_attention, device, args)
    )

    result = {
        "torch": torch.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _native_verify(logits, candidates):
    predicted = logits[:, :-1].argmax(-1)
    accepted = torch.cumprod((candidates[:, 1:] == predicted).int(), dim=1).sum(dim=1)
    best = accepted.argmax()
    return best, accepted[best]


def _native_action_verify(hidden, weight, candidates):
    logits = torch.nn.functional.linear(hidden, weight).reshape(*candidates.shape, weight.shape[0])
    predicted = logits.argmax(-1)
    matches = predicted[:, :-1] == candidates[:, 1:]
    accepted = torch.cumprod(matches.to(torch.int32), dim=1).sum(-1)
    best = accepted.argmax()
    length = accepted[best]
    next_token = predicted[best, length.clamp_max(predicted.shape[1] - 1)]
    return best, length, next_token


def _native_draft_topk(hidden, weight, k, offset):
    logits = torch.nn.functional.linear(hidden, weight)
    values, indices = torch.sort(
        torch.nn.functional.log_softmax(logits.float(), dim=-1),
        dim=-1,
        descending=True,
        stable=True,
    )
    return values[:, :k].to(logits.dtype), indices[:, :k].long() + offset


def _native_tree_embed(tokens, kept, embedding, target_nodes):
    selected = tokens.index_select(1, kept)
    selected = torch.cat(
        (selected, selected[:, :1].expand(-1, target_nodes - selected.shape[1])), dim=1
    )
    return torch.nn.functional.embedding(selected, embedding)


def _native_rope_store(query, key, value, cos, sin, positions, write_start, device, dtype):
    del positions, device, dtype
    half = query.shape[-1] // 2
    rotate = lambda x: torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    q_out = query * cos.unsqueeze(1) + rotate(query) * sin.unsqueeze(1)
    k_out = key * cos.unsqueeze(1) + rotate(key) * sin.unsqueeze(1)
    key_cache = torch.zeros(
        1, key.shape[1], 64, key.shape[-1], device=query.device, dtype=query.dtype
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache[..., write_start : write_start + key.shape[2], :].copy_(k_out)
    value_cache[..., write_start : write_start + value.shape[2], :].copy_(value)
    return q_out, key_cache, value_cache


def _fused_rope_store(query, key, value, cos, sin, positions, write_start):
    key_cache = torch.zeros(
        1, key.shape[1], 64, key.shape[-1], device=query.device, dtype=query.dtype
    )
    value_cache = torch.zeros_like(key_cache)
    q_out = kerv_rope_kv_store(
        query, key, value, cos, sin, positions, key_cache, value_cache, write_start
    )
    return q_out, key_cache, value_cache


def _native_accept_commit(storage, indices, previous_length):
    source = storage[..., previous_length + indices, :].clone()
    storage[..., previous_length : previous_length + indices.numel(), :].copy_(source)
    return storage


def _native_cache_store(destination, source, write_start):
    destination[..., write_start : write_start + source.shape[-2], :].copy_(source)
    return destination


def _case(name, native, fused, device, args):
    if device.type == "cuda":
        native_ms = _time_cuda(native, args.warmup, args.iterations, args.repeats)
        fused_ms = _time_cuda(fused, args.warmup, args.iterations, args.repeats)
    else:
        native_ms = _time_cpu(native, args.warmup, args.iterations, args.repeats)
        fused_ms = _time_cpu(fused, args.warmup, args.iterations, args.repeats)
    native_result = native()
    fused_result = fused()
    if isinstance(native_result, tuple):
        checks = []
        for left, right in zip(native_result, fused_result):
            if left.dtype in (torch.int32, torch.int64, torch.long, torch.bool):
                checks.append(torch.equal(left, right))
            else:
                checks.append(torch.allclose(left, right, rtol=2e-2, atol=2e-2))
        exact = all(checks)
    elif native_result.dtype in (torch.int32, torch.int64, torch.long, torch.bool):
        exact = torch.equal(native_result, fused_result)
    else:
        exact = bool(torch.allclose(native_result, fused_result, rtol=2e-2, atol=2e-2))
    return {
        "operator": name,
        "native_ms": native_ms,
        "fused_ms": fused_ms,
        "speedup": native_ms / fused_ms if fused_ms else None,
        "correct": exact,
    }


if __name__ == "__main__":
    main()

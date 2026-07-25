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

"""GDN (Gated DeltaNet) layer helpers shared across directions and model types."""

import torch


def is_gdn_layer(idx, freq):
    """Return True if layer idx uses GDN rather than standard attention."""
    return (idx % freq) != (freq - 1)


def merge_gdn_in_proj(qkv, z, b, a, cfg):
    """Merge HF's separate qkv, z, b, a into Megatron's fused in_proj.

    Megatron expects TP-rank-grouped layout for ColumnParallel TP sharding:
        [tp0_q, tp0_k, tp0_v, tp0_z, tp0_b, tp0_a,
         tp1_q, tp1_k, tp1_v, tp1_z, tp1_b, tp1_a, ...]

    Each TP rank gets a contiguous block: [rank_q, rank_k, rank_v, rank_z, rank_b, rank_a]
    which matches the forward split: qkv, gate, beta, alpha.
    """
    hidden = cfg.hidden_size
    qk_head_dim = cfg.linear_key_head_dim
    v_head_dim = cfg.linear_value_head_dim
    num_qk_heads = cfg.linear_num_key_heads
    num_v_heads = cfg.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    v_per_group = num_v_heads // num_qk_heads
    tp_size = cfg.tp

    # Split flat QKV into Q, K, V
    q_flat, k_flat, v_flat = torch.split(qkv, [qk_dim, qk_dim, v_dim], dim=0)

    # Reshape to (num_qk_heads, per_head_dim, hidden)
    q_g = q_flat.reshape(num_qk_heads, qk_head_dim, hidden)
    k_g = k_flat.reshape(num_qk_heads, qk_head_dim, hidden)
    v_g = v_flat.reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    z_g = z.reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    b_g = b.reshape(num_qk_heads, v_per_group, hidden)
    a_g = a.reshape(num_qk_heads, v_per_group, hidden)

    # Reorder to TP-rank-grouped layout
    q, k, v, z, b, a = [w.reshape(tp_size, -1, hidden) for w in [q_g, k_g, v_g, z_g, b_g, a_g]]
    in_proj = torch.cat([q, k, v, z, b, a], dim=1).reshape(-1, hidden)
    return in_proj


def split_gdn_in_proj(in_proj_weight, cfg):
    """Split Megatron fused in_proj back into HF's qkv, z, b, a."""
    hidden = cfg.hidden_size
    num_qk_heads = cfg.linear_num_key_heads
    num_v_heads = cfg.linear_num_value_heads
    qk_head_dim = cfg.linear_key_head_dim
    v_head_dim = cfg.linear_value_head_dim
    v_per_group = num_v_heads // num_qk_heads
    tp_size = cfg.tp
    heads_per_tp = num_qk_heads // tp_size

    per_rank_dim = in_proj_weight.shape[0] // tp_size
    rank_weights = in_proj_weight.reshape(tp_size, per_rank_dim, hidden)

    q_parts, k_parts, v_parts, z_parts, b_parts, a_parts = [], [], [], [], [], []
    for rank in range(tp_size):
        rw = rank_weights[rank]
        q_end = heads_per_tp * qk_head_dim
        k_end = q_end + heads_per_tp * qk_head_dim
        v_end = k_end + heads_per_tp * v_per_group * v_head_dim
        z_end = v_end + heads_per_tp * v_per_group * v_head_dim
        b_end = z_end + heads_per_tp * v_per_group
        a_end = b_end + heads_per_tp * v_per_group
        q_parts.append(rw[:q_end])
        k_parts.append(rw[q_end:k_end])
        v_parts.append(rw[k_end:v_end])
        z_parts.append(rw[v_end:z_end])
        b_parts.append(rw[z_end:b_end])
        a_parts.append(rw[b_end:a_end])

    q_g = torch.cat(q_parts, dim=0).reshape(num_qk_heads, qk_head_dim, hidden)
    k_g = torch.cat(k_parts, dim=0).reshape(num_qk_heads, qk_head_dim, hidden)
    v_g = torch.cat(v_parts, dim=0).reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    z_g = torch.cat(z_parts, dim=0).reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    b_g = torch.cat(b_parts, dim=0).reshape(num_qk_heads, v_per_group, hidden)
    a_g = torch.cat(a_parts, dim=0).reshape(num_qk_heads, v_per_group, hidden)

    qkv = torch.cat(
        [
            q_g.reshape(-1, hidden),
            k_g.reshape(-1, hidden),
            v_g.reshape(-1, hidden),
        ],
        dim=0,
    )
    z_flat = z_g.reshape(-1, hidden)
    b_flat = b_g.reshape(-1, hidden)
    a_flat = a_g.reshape(-1, hidden)

    return qkv, z_flat, b_flat, a_flat


def _shard_gdn_conv1d(full_weight, cfg):
    """Split a full GDN conv1d weight/bias into TP-aware shards.

    The HF/Megatron full conv1d is ordered [q_full, k_full, v_full] along dim 0.
    Each TP rank must receive [q_local, k_local, v_local] so that its channels
    match the local output of ``in_proj``.
    """
    tp = cfg.tp
    qk_dim = cfg.qk_dim
    v_dim = cfg.v_dim
    qk_local = qk_dim // tp
    v_local = v_dim // tp

    shards = []
    for r in range(tp):
        q = full_weight[r * qk_local : (r + 1) * qk_local]
        k = full_weight[qk_dim + r * qk_local : qk_dim + (r + 1) * qk_local]
        v = full_weight[2 * qk_dim + r * v_local : 2 * qk_dim + (r + 1) * v_local]
        shards.append(torch.cat([q, k, v], dim=0))
    return shards


def _merge_gdn_conv1d(shards, cfg):
    """Merge TP-aware GDN conv1d shards back to the full tensor."""
    tp = cfg.tp
    qk_dim = cfg.qk_dim
    qk_local = qk_dim // tp

    q_parts, k_parts, v_parts = [], [], []
    for r in range(tp):
        q = shards[r][:qk_local]
        k = shards[r][qk_local : 2 * qk_local]
        v = shards[r][2 * qk_local :]
        q_parts.append(q)
        k_parts.append(k)
        v_parts.append(v)

    q_full = torch.cat(q_parts, dim=0)
    k_full = torch.cat(k_parts, dim=0)
    v_full = torch.cat(v_parts, dim=0)
    return torch.cat([q_full, k_full, v_full], dim=0)

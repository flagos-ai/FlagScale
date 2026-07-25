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

"""FLOPS estimation helpers for performance monitoring."""

from __future__ import annotations


class FLOPSFormulas:
    """Collection of approximate transformer FLOPS formulas."""

    @staticmethod
    def attention_flops(batch_size, seq_length, hidden_size, num_attention_heads):
        head_dim = hidden_size // max(num_attention_heads, 1)
        qkv_flops = 3 * 2 * batch_size * seq_length * hidden_size * hidden_size
        score_flops = 2 * batch_size * num_attention_heads * seq_length * seq_length * head_dim
        value_flops = 2 * batch_size * num_attention_heads * seq_length * seq_length * head_dim
        out_flops = 2 * batch_size * seq_length * hidden_size * hidden_size
        return qkv_flops + score_flops + value_flops + out_flops

    @staticmethod
    def gqa_attention_flops(
        batch_size, seq_length, hidden_size, num_attention_heads, num_query_groups
    ):
        head_dim = hidden_size // max(num_attention_heads, 1)
        kv_hidden_size = head_dim * max(num_query_groups, 1)
        q_flops = 2 * batch_size * seq_length * hidden_size * hidden_size
        kv_flops = 2 * 2 * batch_size * seq_length * hidden_size * kv_hidden_size
        score_flops = 2 * batch_size * num_attention_heads * seq_length * seq_length * head_dim
        value_flops = 2 * batch_size * num_attention_heads * seq_length * seq_length * head_dim
        out_flops = 2 * batch_size * seq_length * hidden_size * hidden_size
        return q_flops + kv_flops + score_flops + value_flops + out_flops

    @staticmethod
    def ffn_flops(batch_size, seq_length, hidden_size, ffn_hidden_size, use_swiglu=False):
        if use_swiglu:
            gate_flops = 2 * batch_size * seq_length * hidden_size * ffn_hidden_size
            up_flops = 2 * batch_size * seq_length * hidden_size * ffn_hidden_size
            swiglu_flops = batch_size * seq_length * ffn_hidden_size
            down_flops = 2 * batch_size * seq_length * ffn_hidden_size * hidden_size
            return gate_flops + up_flops + swiglu_flops + down_flops

        up_flops = 2 * batch_size * seq_length * hidden_size * ffn_hidden_size
        down_flops = 2 * batch_size * seq_length * ffn_hidden_size * hidden_size
        return up_flops + down_flops

    @staticmethod
    def moe_flops(
        batch_size,
        seq_length,
        hidden_size,
        ffn_hidden_size,
        num_experts,
        top_k,
        use_swiglu=False,
    ):
        router_flops = 2 * batch_size * seq_length * hidden_size * num_experts
        active_tokens = batch_size * seq_length * top_k
        if use_swiglu:
            expert_flops = (
                3 * 2 * active_tokens * hidden_size * ffn_hidden_size
                + active_tokens * ffn_hidden_size
            )
        else:
            expert_flops = 4 * active_tokens * hidden_size * ffn_hidden_size
        return router_flops + expert_flops

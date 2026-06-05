# Copyright (c) 2025, BAAI. All rights reserved.
#
# Adopted from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention
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

from einops import rearrange
from torch import Tensor

from megatron.core.transformer.attention import (
    HAVE_FA3,
    BaseInferenceContext,
    PackedSeqParams,
    SelfAttention,
    deprecate_inference_params,
    is_fa_min_version,
)

from .rope import apply_rotary_pos_emb_absolute


class Qwen35SelfAttention(SelfAttention):
    """
    Qwen3.5 Self Attention.

    Overrides SelfAttention to use apply_rotary_pos_emb_absolute instead of
    the standard apply_rotary_pos_emb for mRoPE support.
    Also supports attention_output_gate for gated attention.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: (Tensor | tuple[Tensor, Tensor]) | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass with absolute RoPE for mRoPE support."""
        # Check if we need to skip RoPE
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA3 or is_fa_min_version("2.7.3"), (
                "flash attn version v2.7.3 and above is required for dynamic batching."
            )

        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # Get query, key, value tensors
        gate = None
        if self.config.attention_output_gate:
            query, key, value, gate = self.get_query_key_value_tensors(
                hidden_states, key_value_states, output_gate=True
            )
        else:
            query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        # Adjust key, value, and rotary_pos_emb for inference
        in_decode_mode = (
            inference_context is not None
            and inference_context.is_decode_only()
            and not self.training
        )

        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
                self.layer_number
            ]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            if gate is not None:
                context_layer = self._apply_output_gate(context_layer, gate)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        if (
            in_decode_mode
            and self.config.enable_cuda_graph
            and inference_context.is_static_batching()
        ):
            raise ValueError("CUDA graphs must use flash decode with static batching!")

        query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
            self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        # Apply absolute rotary positional embedding (mRoPE)

        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb_absolute(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query,
                        q_pos_emb,
                        self.config,
                        cu_seqlens_q,
                        self.model_comm_pgs.cp,
                    )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb_absolute(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                )

        # Core attention computation
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            else:
                q, k, v = query, key, value
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, kv_lengths_decode_only, max_seqlen_k = (
                    inference_context.cu_kv_lengths()
                )

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    kv_lengths_decode_only,
                    block_table,
                )
                core_attn_out = rearrange(core_attn_out, "s b h d -> s b (h d)")

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # Apply output gate for hybrid GDN+Attention
        if gate is not None:
            core_attn_out = self._apply_output_gate(core_attn_out, gate)

        # Output projection
        output, bias = self.linear_proj(core_attn_out)
        return output, bias

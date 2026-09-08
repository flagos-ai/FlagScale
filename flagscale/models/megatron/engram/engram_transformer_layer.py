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

from typing import Any, Optional

from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer import TransformerLayer
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

from megatron.core.transformer.engram import EngramModule 


class EngramTransformerLayer(TransformerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engram_layer_id = self.layer_number - 1  # global layer_number starts at 1 in MCore
        self.is_engram_layer = (
            self.config.use_engram
            and self.config.engram_layer_ids is not None
            and self.engram_layer_id in self.config.engram_layer_ids
        )
        if self.is_engram_layer:
            self.engram = EngramModule(
                config=self.config,
                layer_id=self.engram_layer_id,
            )
        else:
            self.engram = None
        self._deepseek_engram_hash_input_ids = None

    def forward(self, *args, **kwargs):
        kwargs.pop("dynamic_inference_decode_only", None)
        self._deepseek_engram_hash_input_ids = kwargs.pop(
            "engram_hash_input_ids", getattr(self, "_deepseek_engram_hash_input_ids", None)
        )

        try:
            return super().forward(*args, **kwargs)
        finally:
            self._deepseek_engram_hash_input_ids = None

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        mhc_recompute_manager = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        if self.is_engram_layer:
            nvtx_range_push(suffix="engram")
            hidden_states = (
                self.engram(hidden_states, self._deepseek_engram_hash_input_ids)
                + hidden_states
            )
            nvtx_range_pop(suffix="engram")

        return super()._forward_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
            input_ids=input_ids,
            inference_params=inference_params,
        )
    
    def pre_compute_embedding(self, engram_hash_input_ids):
        if not self.is_engram_layer or isinstance(self.engram, IdentityOp):
            return
        hash_input_ids = engram_hash_input_ids[self.engram_layer_id]
        self.engram.pre_compute_embedding(hash_input_ids)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ):
        return super().sharded_state_dict(prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata)

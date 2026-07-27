# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek transformer layer wrapper.

This file keeps the DeepSeek-specific engram and hyper-connection wiring, while
reusing the core attention and MLP flow from the local TransformerLayer.
"""

from dataclasses import dataclass
import logging
from typing import Any, Optional, Union

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayerSubmodules,
)

logger = logging.getLogger(__name__)


@dataclass
class DeepSeekTransformerLayerSubmodules(TransformerLayerSubmodules):
    engram: Union[ModuleSpec, type] = IdentityOp


class DeepSeekTransformerLayer(HyperConnectionTransformerLayer):
    """Single layer with DeepSeek-specific engram and hyper-connection hooks."""

    def __init__(self, config, submodules, *args, **kwargs):
        if not getattr(config, "enable_hyper_connections", False):
            raise RuntimeError(
                "DeepSeekTransformerLayer now requires config.enable_hyper_connections=True."
            )
        super().__init__(config=config, submodules=submodules, *args, **kwargs)
        if self.config.use_engram:
            # If not use_engram, the submodules.engram is None
            self.engram = build_module(
                submodules.engram,
                engram_cfg=self.config,
                layer_id=self.layer_number - 1,
            )
        else:
            self.engram = None
        self._deepseek_engram_hash_input_ids = None
        self._mhc_recompute_manager = None
        if (
            self.config.engram_layer_ids is not None
            and self.layer_number - 1 in self.config.engram_layer_ids
        ):
            self.is_engram_layer = True
        else:
            self.is_engram_layer = False

    def _get_submodules_under_cudagraphs(self):
        """Include the DeepSeek-specific submodules in cudagraph pre-forward hooks."""
        submodules = super()._get_submodules_under_cudagraphs()

        if isinstance(self.engram, IdentityOp):
            return submodules

        try:
            insert_at = submodules.index(self.input_layernorm)
        except ValueError:
            insert_at = 0
        submodules.insert(insert_at, self.engram)

        return submodules

    def forward(self, *args, **kwargs):
        """Stage DeepSeek-specific state, then reuse the parent forward path."""
        kwargs.pop("dynamic_inference_decode_only", None)
        self._deepseek_engram_hash_input_ids = kwargs.pop(
            "engram_hash_input_ids", getattr(self, "_deepseek_engram_hash_input_ids", None)
        )
        self._mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)

        try:
            return super().forward(*args, **kwargs)
        finally:
            self._deepseek_engram_hash_input_ids = None
            self._mhc_recompute_manager = None

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
        mhc_recompute_manager=None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """Apply DeepSeek engram before the parent attention path."""
        if self.engram is not None:
            nvtx_range_push(suffix="engram")
            hidden_states = self.engram(hidden_states, self._deepseek_engram_hash_input_ids)
            nvtx_range_pop(suffix="engram")

        return super()._forward_attention(
            hidden_states,
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
        if isinstance(self.engram, IdentityOp) or (
            self.layer_number not in self.config.engram_layer_ids
        ):
            return
        hash_input_ids = engram_hash_input_ids[self.layer_number - 1]
        self.engram.pre_compute_embedding(hash_input_ids)

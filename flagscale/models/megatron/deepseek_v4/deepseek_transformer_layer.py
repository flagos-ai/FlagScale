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
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import make_viewless_tensor, nvtx_range_pop, nvtx_range_push

from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

logger = logging.getLogger(__name__)


class CheckpointManager:
    pass


@dataclass
class DeepSeekTransformerLayerSubmodules(TransformerLayerSubmodules):
    engram: Union[ModuleSpec, type] = IdentityOp
    self_attention_hyper_connection: Union[ModuleSpec, type] = IdentityOp
    mlp_hyper_connection: Union[ModuleSpec, type] = IdentityOp
    cross_attention_connection: Union[ModuleSpec, type] = IdentityOp


class DeepSeekTransformerLayer(TransformerLayer):
    """Single layer with DeepSeek-specific engram and hyper-connection hooks."""

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config=config, submodules=submodules, *args, **kwargs)

        self.engram = build_module(
            submodules.engram,
            engram_cfg=self.config,
            layer_id=self.layer_number - 1,
        )
        if self.config.engram_layer_ids is not None and self.layer_number - 1 in self.config.engram_layer_ids:
            self.is_engram_layer = True
        else:
            self.is_engram_layer = False
        self.self_attention_hyper_connection = build_module(
            submodules.self_attention_hyper_connection,
            config=self.config,
            layer_number=self.layer_number,
        )
        self.mlp_hyper_connection = build_module(
            submodules.mlp_hyper_connection,
            config=self.config,
            layer_number=self.layer_number,
        )
        self.cross_attention_connection = build_module(
            submodules.cross_attention_connection,
            config=self.config,
            layer_number=self.layer_number,
        )

        self._deepseek_engram_hash_input_ids = None
        self._deepseek_self_attn_hyper_state = None
        self._deepseek_mlp_hyper_state = None
        self._deepseek_mhc_recompute_manager = None

        self._patch_self_attn_bda_forward()
        self._patch_mlp_bda_forward()

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Match the parent static inputs, but use n-stream hidden states."""
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        hidden_states = static_inputs["hidden_states"]
        num_streams = self.config.num_residual_streams
        static_inputs["hidden_states"] = torch.ones(
            (hidden_states.shape[0], hidden_states.shape[1], num_streams * self.config.hidden_size),
            dtype=hidden_states.dtype,
            requires_grad=hidden_states.requires_grad,
            device=hidden_states.device,
        )
        if (
            self.is_moe_layer
            and self.config.moe_n_hash_layers > 0
            and getattr(self.mlp.router, "is_hash_layer", False)
        ):
            static_inputs["input_ids"] = torch.zeros(
                (micro_batch_size, seq_length),
                dtype=torch.long,
                device=torch.cuda.current_device(),
            )
        return static_inputs

    def _get_submodules_under_cudagraphs(self):
        """Include the DeepSeek-specific submodules in cudagraph pre-forward hooks."""
        submodules = super()._get_submodules_under_cudagraphs()

        def insert_before(anchor_module, module):
            if isinstance(module, IdentityOp):
                return
            try:
                insert_at = submodules.index(anchor_module)
            except ValueError:
                insert_at = len(submodules)
            submodules.insert(insert_at, module)

        if not self.config.cuda_graph_scope or CudaGraphScope.attn in self.config.cuda_graph_scope:
            insert_before(self.input_layernorm, self.engram)
            insert_before(self.input_layernorm, self.self_attention_hyper_connection)
            if not isinstance(self.cross_attention_connection, IdentityOp):
                try:
                    insert_at = submodules.index(self.cross_attention) + 1
                except ValueError:
                    insert_at = len(submodules)
                submodules.insert(insert_at, self.cross_attention_connection)

        if (
            not self.config.cuda_graph_scope
            or CudaGraphScope.mlp in self.config.cuda_graph_scope
            or (
                self.is_moe_layer
                and (
                    CudaGraphScope.moe in self.config.cuda_graph_scope
                    or CudaGraphScope.moe_router in self.config.cuda_graph_scope
                )
            )
        ):
            insert_before(self.pre_mlp_layernorm, self.mlp_hyper_connection)

        return submodules

    def forward(self, *args, **kwargs):
        """Stage DeepSeek-specific state, then reuse the parent forward path."""
        kwargs.pop("dynamic_inference_decode_only", None)
        self._deepseek_engram_hash_input_ids = kwargs.pop("engram_hash_input_ids", None)
        self._deepseek_mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)

        try:
            return super().forward(*args, **kwargs)
        finally:
            self._deepseek_engram_hash_input_ids = None
            self._deepseek_self_attn_hyper_state = None
            self._deepseek_mlp_hyper_state = None
            self._deepseek_mhc_recompute_manager = None

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
        *,
        inference_params: Optional[Any] = None,
    ):
        """Apply DeepSeek engram and self-attention hyper-connection prep before the parent path."""
        if not isinstance(self.engram, IdentityOp):
            nvtx_range_push(suffix="engram")
            hidden_states = self.engram(hidden_states, self._deepseek_engram_hash_input_ids)
            nvtx_range_pop(suffix="engram")
        self._origin_attn_residual = hidden_states # [s, b, n, C]
        if not isinstance(self.self_attention_hyper_connection, IdentityOp):
            nvtx_range_push(suffix="self_attention_hyper_connection")
            hidden_states, self_attn_h_res, self_attn_hc_h_post = self._hyper_connection_forward(
                hidden_states,
                self.self_attention_hyper_connection,
                self._deepseek_mhc_recompute_manager,
            )
            nvtx_range_pop(suffix="self_attention_hyper_connection")
            self._deepseek_self_attn_hyper_state = (
                self_attn_h_res,
                self_attn_hc_h_post,
                self._deepseek_mhc_recompute_manager,
            )
        else:
            self._deepseek_self_attn_hyper_state = None

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

    def _forward_mlp(
        self,
        hidden_states: Tensor,
        inference_context: BaseInferenceContext | None = None,
        padding_mask: Tensor | None = None,
        input_ids: Optional[Tensor] = None,
    ) -> Tensor | list[Tensor | None]:
        """Apply DeepSeek hyper-connection prep before the parent MLP path."""
        mhc_recompute_manager = self._deepseek_mhc_recompute_manager
        is_last_in_recompute_block = bool(
            mhc_recompute_manager is not None
            and getattr(mhc_recompute_manager, "is_last_layer_in_recompute_block", False)
        )
        mhc_mlp_bda_manager = None if is_last_in_recompute_block else mhc_recompute_manager
        self._mlp_origin_residual = hidden_states # [s, b, n, C]
        if not isinstance(self.mlp_hyper_connection, IdentityOp):
            nvtx_range_push(suffix="mlp_hyper_connection")
            hidden_states, mlp_h_res, mlp_hc_h_post = self._hyper_connection_forward(
                hidden_states,
                self.mlp_hyper_connection,
                mhc_recompute_manager,
            )
            nvtx_range_pop(suffix="mlp_hyper_connection")
            self._deepseek_mlp_hyper_state = (
                mlp_h_res,
                mlp_hc_h_post,
                mhc_mlp_bda_manager,
            )
        else:
            self._deepseek_mlp_hyper_state = None

        return super()._forward_mlp(
            hidden_states,
            inference_context=inference_context,
            padding_mask=padding_mask,
            input_ids=input_ids,
        )

    def _forward_post_mlp(self, mlp_output_with_bias, residual):
        mhc_postprocess = self._deepseek_mlp_hyper_state

        if (
            mhc_postprocess is None
            or mhc_postprocess[0] is None
            or mhc_postprocess[1] is None
        ):
            if (not self.training) and self.config.inference_fuse_tp_communication:
                self._deepseek_mlp_hyper_state = None
            return super()._forward_post_mlp(mlp_output_with_bias, residual)

        if (not self.training) and self.config.inference_fuse_tp_communication:
            self._deepseek_mlp_hyper_state = None
            mlp_h_res, mlp_hc_h_post, mhc_mlp_bda_manager = mhc_postprocess
            return self._forward_post_mlp_with_fused_hyper_connection(
                mlp_output_with_bias,
                mlp_h_res,
                residual,
                mlp_hc_h_post,
                mhc_mlp_bda_manager,
            )

        return super()._forward_post_mlp(mlp_output_with_bias, residual)

    def _forward_post_mlp_with_fused_hyper_connection(
        self,
        mlp_output_with_bias,
        mlp_h_res,
        residual,
        mlp_hc_h_post,
        mhc_mlp_bda_recompute_manager: Optional[CheckpointManager] = None,
    ):
        """Run the DeepSeek mHC fusion path after the parent MLP computes its output."""
        if self.recompute_pre_mlp_layernorm or (
            mhc_mlp_bda_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
        ):
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        nvtx_range_push(suffix="mlp_fused_h_res_h_post_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_hyper_connection.fused_h_res_h_post_bda(
                mlp_h_res,
                residual,
                mlp_hc_h_post,
                mlp_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
                mhc_mlp_bda_recompute_manager,
            )
        nvtx_range_pop(suffix="mlp_fused_h_res_h_post_bda")

        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        if self.offload_mlp_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output

    def _patch_mlp_bda_forward(self):
        original_forward = self.mlp_bda

        def patched_forward(*bda_args, **bda_kwargs):
            if bda_args:
                training = bda_args[0]
                bias_dropout_fusion = bda_args[1] if len(bda_args) > 1 else bda_kwargs.get(
                    "bias_dropout_fusion"
                )
            else:
                training = bda_kwargs.get("training")
                bias_dropout_fusion = bda_kwargs.get("bias_dropout_fusion")

            base_bda = original_forward(training, bias_dropout_fusion)

            def apply(main_output_with_bias, residual, hidden_dropout):
                state = self._deepseek_mlp_hyper_state
                self._deepseek_mlp_hyper_state = None

                if state is None or isinstance(self.mlp_hyper_connection, IdentityOp):
                    return base_bda(main_output_with_bias, residual, hidden_dropout)

                mlp_h_res, mlp_hc_h_post, mhc_mlp_bda_manager = state
                if mlp_h_res is None or mlp_hc_h_post is None:
                    return base_bda(main_output_with_bias, residual, hidden_dropout)

                return self.mlp_hyper_connection.fused_h_res_h_post_bda(
                    mlp_h_res,
                    self._mlp_origin_residual,
                    mlp_hc_h_post,
                    main_output_with_bias,
                    hidden_dropout,
                    training,
                    bias_dropout_fusion,
                    mhc_mlp_bda_manager,
                )

            return apply

        self.mlp_bda = patched_forward

    def _patch_self_attn_bda_forward(self):
        original_forward = self.self_attn_bda

        def patched_forward(*bda_args, **bda_kwargs):
            if bda_args:
                training = bda_args[0]
                bias_dropout_fusion = bda_args[1] if len(bda_args) > 1 else bda_kwargs.get(
                    "bias_dropout_fusion"
                )
            else:
                training = bda_kwargs.get("training")
                bias_dropout_fusion = bda_kwargs.get("bias_dropout_fusion")

            base_bda = original_forward(training, bias_dropout_fusion)

            def apply(main_output_with_bias, residual, hidden_dropout):
                state = self._deepseek_self_attn_hyper_state
                self._deepseek_self_attn_hyper_state = None

                if (
                    state is None
                    or isinstance(self.self_attention_hyper_connection, IdentityOp)
                ):
                    return base_bda(main_output_with_bias, residual, hidden_dropout)

                h_res, h_post, mhc_recompute_manager = state
                if h_res is None or h_post is None:
                    return base_bda(main_output_with_bias, residual, hidden_dropout)

                return self.self_attention_hyper_connection.fused_h_res_h_post_bda(
                    h_res,
                    self._origin_attn_residual,
                    h_post,
                    main_output_with_bias,
                    hidden_dropout,
                    training,
                    bias_dropout_fusion,
                    mhc_recompute_manager,
                )

            return apply

        self.self_attn_bda = patched_forward

    def _hyper_connection_forward(
        self,
        hidden_states,
        hyper_connection_module,
        mhc_recompute_manager,
    ):
        """Run a DeepSeek hyper connection module and normalize the return shape."""
        if isinstance(hyper_connection_module, IdentityOp):
            return hidden_states, None, None
        assert hyper_connection_module is not None, (
            "Hyper connection module cannot be None when not IdentityOp."
        )
        output = hyper_connection_module(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager
        )
        transformed_hidden_states, h_res, hc_h_post = output
        return transformed_hidden_states, h_res, hc_h_post

    def pre_compute_embedding(self, engram_hash_input_ids):
        if not isinstance(self.engram, IdentityOp):
            hash_input_ids = engram_hash_input_ids[self.layer_number - 1]
            self.engram.pre_compute_embedding(hash_input_ids)

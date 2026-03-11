# ruff: noqa: RUF013
# ruff: noqa: E711

## built-in
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams

# megatron-core
from megatron.core.transformer import TransformerLayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    make_viewless_tensor,
)

# engram
from .engram import Engram
from .engram_config import EngramConfig


class EngramTransformerLayer(TransformerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.config, EngramConfig), "config must be a EngramConfig"
        self.engram_hash_layer_id = (
            self.layer_number - 1
        )  # global layer_number starts at 1 in MCore
        self.engram = Engram(
            engram_cfg=self.config,
            layer_id=self.engram_hash_layer_id,
        )

    def forward(
        self,
        input_ids: Tensor,
        hash_input_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        inference_context: Any | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: Tensor | None = None,
        *,
        inference_params: Any | None = None,
    ):
        if self.engram_hash_layer_id in self.config.engram_layer_ids:
            hidden_states = (
                self.engram(hidden_states=hidden_states, hash_input_ids=hash_input_ids)
                + hidden_states
            )

        return super().forward(
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
            inference_params=inference_params,
        )

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ):
        raise NotImplementedError("Sharded state dict is not supported for EngramTransformerLayer")


class EngramTransformerBlock(TransformerBlock):
    def forward(
        self,
        input_ids: Tensor,
        engram_hash_input_ids: Any,
        hidden_states: Tensor | WrappedTensor,
        attention_mask: Tensor | None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: Tensor | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        dynamic_inference_decode_only: bool | None = None,
    ):
        ########## FlagScale Begin ##########
        # for refined recompute
        self.current_microbatch = -1
        if (
            len(self.layers) > 0
        ):  # some pp-stage has no layers in pipeline_model_parallel_layout,such as embedding stage
            if hasattr(self.layers[0], "current_microbatch"):
                self.current_microbatch = self.layers[0].current_microbatch
        ########## FlagScale End ##########

        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
            )
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()
        ########## FlagScale Begin ##########
        if self.config.recompute_method_per_stage_micro_batch != None:
            if self.config.virtual_pipeline_model_parallel_size != None:
                if (
                    self.config.recompute_method_per_stage_micro_batch[
                        parallel_state.get_virtual_pipeline_model_parallel_rank()
                        * self.config.pipeline_model_parallel_size
                        + parallel_state.get_pipeline_model_parallel_rank()
                    ][self.current_microbatch]
                    == 0
                ):
                    self.config.recompute_method = "uniform"
                elif (
                    self.config.recompute_method_per_stage_micro_batch[
                        parallel_state.get_virtual_pipeline_model_parallel_rank()
                        * self.config.pipeline_model_parallel_size
                        + parallel_state.get_pipeline_model_parallel_rank()
                    ][self.current_microbatch]
                    == 1
                ):
                    self.config.recompute_method = "block"
                else:
                    raise ValueError(
                        "the item of recompute_method_per_stage_micro_batch must be '0' or '1' "
                    )
            else:
                if (
                    self.config.recompute_method_per_stage_micro_batch[
                        parallel_state.get_pipeline_model_parallel_rank()
                    ][self.current_microbatch]
                    == 0
                ):
                    self.config.recompute_method = "uniform"
                elif (
                    self.config.recompute_method_per_stage_micro_batch[
                        parallel_state.get_pipeline_model_parallel_rank()
                    ][self.current_microbatch]
                    == 1
                ):
                    self.config.recompute_method = "block"
                else:
                    raise ValueError(
                        "the item of recompute_method_per_stage_micro_batch must be '0' or '1' "
                    )
            ########## FlagScale End ##########
        if self.config.recompute_num_layers_per_stage_micro_batch != None:
            if self.config.virtual_pipeline_model_parallel_size != None:
                self.config.recompute_num_layers = (
                    self.config.recompute_num_layers_per_stage_micro_batch[
                        parallel_state.get_virtual_pipeline_model_parallel_rank()
                        * self.config.pipeline_model_parallel_size
                        + parallel_state.get_pipeline_model_parallel_rank()
                    ][self.current_microbatch]
                )
            else:
                self.config.recompute_num_layers = (
                    self.config.recompute_num_layers_per_stage_micro_batch[
                        parallel_state.get_pipeline_model_parallel_rank()
                    ][self.current_microbatch]
                )
            if self.config.recompute_num_layers == 0:
                self.config.recompute_method = None
                self.config.recompute_granularity = None

        if (
            self.config.recompute_granularity_per_stage_micro_batch != None
            and self.config.recompute_granularity_per_stage_micro_batch[
                parallel_state.get_pipeline_model_parallel_rank()
            ][self.current_microbatch]
            == 0
        ):
            self.config.recompute_granularity = None
            self.config.recompute_method = None

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with self.offload_context, inner_quantization_context:
                        # Build kwargs based on layer type
                        layer_kwargs = {}

                        # Only pass input_ids to EngramTransformerLayer
                        if isinstance(layer, EngramTransformerLayer):
                            layer_kwargs["input_ids"] = input_ids
                            engram_hash_layer_id = layer.layer_number - 1
                            hash_input_ids = engram_hash_input_ids[engram_hash_layer_id]
                            layer_kwargs["hash_input_ids"] = hash_input_ids

                        # Add common parameters
                        layer_kwargs.update(
                            {
                                "hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "context": context,
                                "context_mask": context_mask,
                                "rotary_pos_emb": rotary_pos_emb,
                                "rotary_pos_cos": rotary_pos_cos,
                                "rotary_pos_sin": rotary_pos_sin,
                                "rotary_pos_cos_sin": rotary_pos_cos_sin,
                                "attention_bias": attention_bias,
                                "inference_context": inference_context,
                                "packed_seq_params": packed_seq_params,
                                "sequence_len_offset": sequence_len_offset,
                            }
                        )

                        hidden_states, context = layer(**layer_kwargs)

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict = None
    ):
        raise NotImplementedError("Sharded state dict is not supported for EngramTransformerBlock")

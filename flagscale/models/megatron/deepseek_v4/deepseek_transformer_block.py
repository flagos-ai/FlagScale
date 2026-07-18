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

## built-in
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union, cast

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import CheckpointManager
else:
    CheckpointManager = None

# megatron-core
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    apply_module,
    get_transformer_layer_offset,
)
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_pg_rank,
    make_viewless_tensor,
)
from .hyper_connection import HyperConnectionModule


class DeepSeekTransformerBlock(TransformerBlock):
    ## TODO: Finish at next version. These codes are copied from Megatron-LM dev branch, megatron/core/transformer/transformer_block.py
    def _build_mhc_recompute_layer_plan(
        self, use_mhc_recompute: bool
    ) -> Tuple[List[Optional[CheckpointManager]], List[bool]]:
        """Pre-build per-layer MHC recompute managers and block-end markers."""
        num_layers = len(self.layers)
        layer_managers: List[Optional[CheckpointManager]] = [None] * num_layers
        is_recompute_block_end: List[bool] = [False] * num_layers

        if not use_mhc_recompute or num_layers == 0:
            return layer_managers, is_recompute_block_end

        mhc_recompute_layer_num = self.config.mhc_recompute_layer_num
        mhc_manager = CheckpointManager()

        for l_no in range(num_layers):
            is_last_in_transformer_block = l_no == num_layers - 1
            is_last_in_recompute_block = is_last_in_transformer_block
            if mhc_recompute_layer_num is not None:
                is_last_in_recompute_block = is_last_in_transformer_block or (
                    (l_no + 1) % mhc_recompute_layer_num == 0
                )

            layer_managers[l_no] = mhc_manager
            is_recompute_block_end[l_no] = is_last_in_recompute_block

            if is_last_in_recompute_block and not is_last_in_transformer_block:
                mhc_manager = CheckpointManager()

        return layer_managers, is_recompute_block_end

    @staticmethod
    def _finalize_mhc_recompute_layer(mhc_manager, hidden_states, is_last_in_recompute_block):
        """
        This function is only implemented in megatron dev branch, not any release version.
        And if not using mhc recompute, this is no needed. So we just keep the function signature and leave it empty for now.
        """
        pass

    # @staticmethod
    # def _finalize_mhc_recompute_layer(
    #     mhc_manager: Optional[CheckpointManager],
    #     hidden_states: Tensor,
    #     is_last_in_recompute_block: bool,
    # ) -> None:
    #     """Finalize MHC recompute state for the current layer when block ends."""
    #     if mhc_manager is not None and is_last_in_recompute_block:
    #         mhc_manager.discard_all_outputs_and_register_unified_recompute(hidden_states)

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
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
        extract_layer_indices: Optional[Set[int]] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
        input_ids: Optional[Tensor] = None,  # Used for hash-based MoE Router
        engram_hash_input_ids: Optional[
            Tensor
        ] = None,  # This is the input ids used for hashing in engram, which can be different from the input_ids used for attention mask and positional embedding. Only used when engram is enabled and at least one of the layers in this block has engram hashing.
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            extract_layer_indices (Set[int], optional): A set of global
                layer indices (0-based across all pipeline stages) from
                which to extract intermediate hidden states. If
                non-empty, the forward pass will collect hidden_states
                after each specified layer.
            dynamic_inference_decode_only: Optional[bool]: If true, indicates that the current
                inference context is for decode-only. This args is only used to uniquely
                identify decode and non-decode cuda graph runners in the cuda graph manager.

        Returns:
            Union[Tensor, Tuple[Tensor, List[Tensor]]]:
                - If extract_layer_indices is None or empty: Returns the output hidden states tensor
                  of shape [s, b, h].
                - If extract_layer_indices is non-empty: Returns a tuple
                  of (hidden_states, intermediate_hidden_states) where
                  intermediate_hidden_states is a list of tensors
                  corresponding to hidden states after each layer in
                  extract_layer_indices.
        """

        ########## FlagScale Begin ##########
        # for refined recompute
        self.current_microbatch = -1
        if len(self.layers) > 0:
            if hasattr(self.layers[0], "current_microbatch"):
                self.current_microbatch = self.layers[0].current_microbatch
        saved_recompute_granularity = self.config.recompute_granularity
        ########## FlagScale End ##########

        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager

        # Initialize feature collection (consistent with FastGen's Wan implementation)
        if extract_layer_indices is None:
            extract_layer_indices = set()
        intermediate_hidden_states: List[Tensor] = []

        # Calculate the global layer offset for this pipeline stage
        # This is needed to convert local layer indices to global indices for feature extraction
        pp_group = self.pg_collection.pp if hasattr(self.pg_collection, "pp") else None
        layer_offset = get_transformer_layer_offset(
            self.config, self.vp_stage, get_pg_rank(pp_group)
        )

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

        # Expand hidden states for hyper connections at the start of the block
        # Only expand at the first PP stage; subsequent stages receive n-stream from previous stage
        if self.config.enable_hyper_connections and self.pre_process:
            hidden_states = HyperConnectionModule.input_expand(
                hidden_states, self.config.num_residual_streams
            )  # [s, b, C] -> [s, b, n*C]

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

        # Determine if MHC recompute should be used
        # Only enable when: training mode AND hyper connections AND 'mhc' in recompute_modules
        use_mhc_recompute = (
            self.training
            and self.config.enable_hyper_connections
            and self.config.recompute_granularity == "selective"
            and "mhc" in self.config.recompute_modules
        )
        mhc_layer_managers, mhc_is_last_in_recompute_block = self._build_mhc_recompute_layer_plan(
            use_mhc_recompute
        )

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == "full" and self.training:
                checkpointed_result = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                    padding_mask=padding_mask,
                    extract_layer_indices=extract_layer_indices,
                    layer_offset=layer_offset,
                    input_ids=input_ids,
                )
                # Handle return value from _checkpointed_forward
                if len(extract_layer_indices) > 0:
                    # (hidden_states, intermediate_hidden_states) tuple
                    hidden_states, intermediate_hidden_states = checkpointed_result
                else:
                    # No intermediate_hidden_states requested: just hidden_states
                    hidden_states = checkpointed_result
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

                    mhc_manager = mhc_layer_managers[l_no]
                    if mhc_manager is not None:
                        mhc_manager.is_last_layer_in_recompute_block = (
                            mhc_is_last_in_recompute_block[l_no]
                        )

                    with self.offload_context, inner_quantization_context:
                        #### FlagScale Begin ####
                        # Pre-compute embeddings for the next DeepSeekTransformerLayer if engram exists, to overlap with current layer's computation
                        if l_no < len(self.layers) - 1:
                            next_layer = self.layers[l_no + 1]
                            if getattr(next_layer, "is_engram_layer", False):
                                next_layer.pre_compute_embedding(engram_hash_input_ids)
                        #### FlagScale End ####
                        hidden_states, context = layer(
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
                            mhc_recompute_manager=mhc_manager,
                            input_ids=input_ids,
                        )

                    self._finalize_mhc_recompute_layer(
                        mhc_manager=mhc_manager,
                        hidden_states=hidden_states,
                        is_last_in_recompute_block=mhc_is_last_in_recompute_block[l_no],
                    )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

                    # Extract intermediate embeddings using global layer index
                    if (l_no + layer_offset) in extract_layer_indices:
                        intermediate_hidden_states.append(hidden_states)

        # Only contract if the final layer norm is in this stage
        if self.config.enable_hyper_connections and self.has_final_layernorm_in_this_stage():
            hidden_states = HyperConnectionModule.output_contract(
                hidden_states, self.config.num_residual_streams
            )  # [s, b, n*C] -> [s, b, C]

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = apply_module(self.final_layernorm)(cast(Tensor, hidden_states))
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
        ########## FlagScale Begin ##########
        self.config.recompute_granularity = saved_recompute_granularity
        ########## FlagScale End ##########

        if len(extract_layer_indices) > 0:
            return hidden_states, intermediate_hidden_states

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ):
        # Engram let the layers be non-homogeneous, so we need to set the flag in metadata to let the sharded state dict logic know.
        # This is usefule when all layer are same, the TransformerBlock will be homogeneous, it generate sharded_state_dict will same keys for all layer and need all layers have the same structure.
        # The layer has engram module does not fit this assumption.
        # If the flag is set to True, the sharded_state_dict will use layer_number to generate different keys for different layer, which is same to models has dense layer leading and moe layer following.
        # Actually, engram really causes the layers to be non-homogeneous.
        if metadata is None:
            metadata = {}
        metadata["non_homogeneous_layers"] = True
        return super().sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )

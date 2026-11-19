# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor, nn


from megatron.core.transformer.transformer_block import TransformerBlock, _GradScale
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_pg_rank,
    make_viewless_tensor,
)
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region)

try:
    import transformer_engine.pytorch as te  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False

get_cpu_offload_context = None
te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    LayerNormImpl = TENorm

elif HAVE_APEX:
    LayerNormImpl = FusedLayerNorm

else:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LayerNormImpl = WrappedTorchNorm


class LanguageTransformerBlock(TransformerBlock):

    def __init__(
        self,
        config,
        spec,
        post_layer_norm=True,
        pre_process=True,
        post_process=True,
        pg_collection=None,
        vp_stage=None,
        dualpipev_stage=None,
    ):
        super().__init__(
            config, spec, post_layer_norm, pre_process, post_process, pg_collection, vp_stage, dualpipev_stage
        )

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            """Create a custom forward function for execution plan steps [start, end).

            When loop is enabled, start/end refer to indices in self._execution_plan.
            Each step maps to (local_layer_idx, loop_iteration).
            """
            def custom_forward(
                hidden_states, attention_mask, context, context_mask, rotary_pos_emb
            ):
                for exec_idx in range(start, end):
                    # Scale gradients at the prelude→loop boundary
                    if (
                        exec_idx == self._loop_entry_exec_idx
                        and self.config.loop_prelude_grad_scale is not None
                    ):
                        hidden_states = _GradScale.apply(
                            hidden_states, self.config.loop_prelude_grad_scale
                        )

                    local_idx, loop_iter = self._execution_plan[exec_idx]
                    layer = self._get_layer(local_idx)

                    loop_residual_scale = (
                        self._loop_residual_scales[loop_iter]
                        if self._loop_enabled and loop_iter >= 0
                        and self._loop_residual_scales is not None
                        else None
                    )

                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        # TODO: check if fp4 is supported in this case
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                            loop_residual_scale=loop_residual_scale,
                            loop_iter=loop_iter if self._loop_enabled else None,
                        )

                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # TODO: check if fp4 is supported in this case
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        num_exec_steps = len(self._execution_plan)

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of execution steps and checkpoint
            # the input activation of each divided chunk.
            exec_idx = 0
            while exec_idx < num_exec_steps:
                chunk_end = min(
                    exec_idx + self.config.recompute_num_layers, num_exec_steps
                )
                hidden_states, context = checkpoint_handler(
                    custom(exec_idx, chunk_end)
                )

                # Deepstack visual embedding injection (only on first loop visit)
                if visual_pos_masks is not None and deepstack_visual_embeds is not None:
                    assert len(self.layers) >= len(deepstack_visual_embeds), (
                        f"First pipeline stage should have at least "
                        f"{len(self.layers)} layers for deepstack."
                    )
                    for idx in range(exec_idx, chunk_end):
                        local_idx, loop_iter = self._execution_plan[idx]
                        if loop_iter == 0 and local_idx < len(deepstack_visual_embeds):
                            if idx == chunk_end - 1:
                                hidden_states = self._deepstack_process(
                                    hidden_states,
                                    visual_pos_masks,
                                    deepstack_visual_embeds[local_idx],
                                )

                exec_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # execution steps and skip the rest.
            recompute_skip_num_layers = 0
            for exec_idx in range(num_exec_steps):
                # Skip recomputation when input grad computation is not needed.
                # TODO: check if fp4 is supported in this case
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    exec_idx >= recompute_skip_num_layers
                    and exec_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(exec_idx, exec_idx + 1))
                else:
                    hidden_states, context = custom(exec_idx, exec_idx + 1)(
                        hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states


    """Transformer class."""
    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
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
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

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

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == "full" and self.training:
                if self._loop_enabled:
                    print(
                        f"[Loop] loop execution (checkpointed), "
                        f"loop_type='{self.config.loop_type}', "
                        f"exec_steps={len(self._execution_plan)}, "
                        f"scales={self._loop_residual_scales}",
                        flush=True,
                    )
                else:
                    print("[Loop] without loop execution (checkpointed)", flush=True)
                if visual_pos_masks is not None or deepstack_visual_embeds is not None:
                    assert (
                        self.config.recompute_method == "uniform"
                        and self.config.recompute_num_layers == 1
                    ), (
                        f"Only uniform recompute with recompute_num_layers=1 is supported "
                        f"for full recompute when deepstack is enabled."
                    )
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )
            else:
                if self._loop_enabled:
                    print(
                        f"[Loop] loop execution (non-checkpointed), "
                        f"loop_type='{self.config.loop_type}', "
                        f"exec_steps={len(self._execution_plan)}, "
                        f"scales={self._loop_residual_scales}",
                        flush=True,
                    )
                else:
                    print("[Loop] without loop execution (non-checkpointed)", flush=True)
                for exec_step, (local_idx, loop_iter) in enumerate(self._execution_plan):
                    # Scale gradients at the prelude→loop boundary
                    if (
                        exec_step == self._loop_entry_exec_idx
                        and self.config.loop_prelude_grad_scale is not None
                    ):
                        hidden_states = _GradScale.apply(
                            hidden_states, self.config.loop_prelude_grad_scale
                        )

                    layer = self.layers[local_idx]

                    # Determine loop residual scale for this execution step
                    loop_residual_scale = (
                        self._loop_residual_scales[loop_iter]
                        if self._loop_enabled and loop_iter >= 0
                        and self._loop_residual_scales is not None
                        else None
                    )

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
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                            padding_mask=padding_mask,
                            loop_residual_scale=loop_residual_scale,
                            loop_iter=loop_iter if self._loop_enabled else None,
                        )

                    # Deepstack visual embedding addition (only on first loop visit)
                    if visual_pos_masks is not None and deepstack_visual_embeds is not None:
                        assert len(self.layers) >= len(deepstack_visual_embeds), (
                            f"First pipeline stage should have at least "
                            f"{len(self.layers)} layers for deepstack."
                        )
                        if loop_iter == 0 and local_idx < len(deepstack_visual_embeds):
                            hidden_states = self._deepstack_process(
                                hidden_states,
                                visual_pos_masks,
                                deepstack_visual_embeds[local_idx],
                            )

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

        # --- Collect GDN decay diagnostic references (lazy, no device sync) ---
        if getattr(self.config, 'gdn_spike_diag_threshold', None) is not None:
            self._gdn_decay_diag = {}
            for local_idx, layer in enumerate(self.layers):
                attn = getattr(layer, 'self_attention', None)
                if attn is None:
                    attn = getattr(layer, 'inner', None)
                    if attn is not None:
                        attn = getattr(attn, 'self_attention', None)
                g_ref = getattr(attn, '_g_diag_ref', None) if attn is not None else None
                if g_ref is not None:
                    global_layer = layer.layer_number - 1  # 0-based, align with param names
                    self._gdn_decay_diag[global_layer] = g_ref
                    attn._g_diag_ref = None  # clear for next step

        return hidden_states


    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        # Gather hidden states from all sequence parallel ranks, becasuse visual_embeds is not sharded.
        if self.config.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        if self.config.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        return hidden_states

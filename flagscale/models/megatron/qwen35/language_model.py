# Copyright (c) 2025, BAAI. All rights reserved.
#
# Adopted from flagscale.models.megatron.qwen3_vl.language_model
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

from typing import Literal

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    mtp_on_this_rank,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params

from .language_transformer_block import LanguageTransformerBlock
from .rope import Qwen35LanguageRotaryEmbedding
from flagscale.models.megatron.qwen2_5_vl.language_module import QwenVLLanguageModelEmbedding


class Qwen35LanguageModule(GPTModel):
    """Qwen3.5 Language Module.

    Args:
        config: Transformer config
        transformer_layer_spec: Module spec for transformer layers
        vocab_size: Vocabulary size
        max_sequence_length: Maximum sequence length
        position_embedding_type: Position embedding type ('mrope')
        rotary_percent: Percent of rotary dimension
        rotary_base: Base for rotary position embeddings
        pre_process: Include embedding layer
        post_process: Include output layer
        fp16_lm_cross_entropy: Use fp16 for cross entropy loss
        parallel_output: Keep output split across TP ranks
        share_embeddings_and_output_weights: Share embeddings and output weights
        scatter_embedding_sequence_parallel: Scatter embeddings across SP ranks
        seq_len_interpolation_factor: RoPE interpolation factor
        mtp_block_spec: Multi-token prediction block spec
        pg_collection: Process group collection
        vp_stage: Virtual pipeline stage
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "mrope", "none"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: float | None = None,
        mtp_block_spec: ModuleSpec | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ) -> None:
        super(GPTModel, self).__init__(config=config, pg_collection=pg_collection)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vp_stage = vp_stage

        if hasattr(self.config, "position_embedding_type"):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        self.model_type = ModelType.encoder_or_decoder

        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if hasattr(self.config, "rotary_base"):
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling
        self.mtp_block_spec = mtp_block_spec
        self.mtp_process = mtp_block_spec is not None and mtp_on_this_rank(
            self.config, ignore_virtual=False, vp_stage=vp_stage
        )

        if self.pre_process or self.mtp_process:
            self.embedding = QwenVLLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == "mrope" and not self.config.multi_latent_attention:
            cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
            self.rotary_pos_emb = Qwen35LanguageRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                cp_group=cp_group,
            )
            self.mrope_section = self.config.mrope_section
            assert self.mrope_section is not None, (
                "mrope requires mrope_section setting, but we got None from TransformerConfig"
            )

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = LanguageTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )

        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(
                config=self.config,
                spec=self.mtp_block_spec,
                vp_stage=vp_stage,
                pg_collection=self.pg_collection,
            )

        # Output
        if self.post_process:
            if self.config.defer_embedding_wgrad_compute:
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process or self.mtp_process:
            self.setup_embeddings_and_output_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f"{type(self).__name__}_init_ckpt"
            )
        for name, module in self.named_modules():
            if hasattr(module, "finish_init"):
                quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
                module.finish_init(quant_config)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output=None,
        # args for deepstack
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        *,
        inference_params=None,
        loss_mask=None,
    ):
        # Forward logic adapted from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model.Qwen3VLGPTModel
        inference_context = deprecate_inference_params(inference_context, inference_params)

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        # Run decoder
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

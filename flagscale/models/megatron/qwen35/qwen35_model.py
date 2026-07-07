# Copyright (c) 2025, BAAI. All rights reserved.
#
# Adopted from flagscale.models.megatron.qwen3_vl.model
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

"""
Qwen3.5 Model.

Architecture:
- Hybrid GDN + Attention language model (Qwen35LanguageModule)
- Reuses Qwen3-VL vision encoder (identical architecture)
"""

import logging

import torch

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from .language_model import Qwen35LanguageModule
from .rope import get_rope_index
from .transformer_config import Qwen35TransformerConfig

# Re-use vision model from qwen3_vl (identical vision encoder for qwen3.5)
from flagscale.models.megatron.qwen3_vl.vision_model import Qwen3VisionModel


class Qwen35Model(MegatronModule):
    """Qwen3.5 multi-modal model.

    Args:
        language_transformer_config (Qwen35TransformerConfig): Transformer config for language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length.
        vision_transformer_config (TransformerConfig): Transformer config for vision model.
        vision_transformer_layer_spec (ModuleSpec): Specifies module to use for vision layers.
        vision_projection_config (TransformerConfig): Config for vision projection.
        vision_projection_layer_spec (ModuleSpec): Specifies module to use for vision projection.
        vision_projection_type (str): Type of vision projection. Default 'mlp'.
        parallel_output (bool): Keep output split across TP ranks. Default True.
        language_position_embedding_type (str): Position embedding type. Default 'mrope'.
        language_rotary_percent (float): Rotary percent. Default 0.25.
        pre_process (bool): Include embedding layer. Default True.
        post_process (bool): Include output layer. Default True.
        add_encoder (bool): Construct encoder module. Default True.
        add_decoder (bool): Construct decoder module. Default True.
        language_rotary_base (int): Rotary base. Default 10000000.
        fp16_lm_cross_entropy (bool): Use fp16 for cross entropy. Default False.
        language_share_embeddings_and_output_weights (bool): Share embeddings and output weights.
        vp_stage (int): Virtual pipeline stage.
    """

    def __init__(
        self,
        language_transformer_config: Qwen35TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        parallel_output: bool = True,
        language_position_embedding_type: str = "mrope",
        language_rotary_percent: float = 0.25,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000000,
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool = False,
        vp_stage: int = None,
        mtp_block_spec=None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        logging.getLogger(__name__).warning(
            "Qwen3.5 model is under development and may be missing features."
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        self.square_merge_size = (
            vision_projection_config.ffn_hidden_size // vision_transformer_config.hidden_size
        )

        self.share_embeddings_and_output_weights = False

        if self.pre_process:
            self.vision_model = Qwen3VisionModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                vision_projection_config,
                vision_projection_layer_spec,
                projection_type=vision_projection_type,
                pre_process=True,
                post_process=True,
            )

        self.language_model = Qwen35LanguageModule(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=language_position_embedding_type,
            rotary_percent=language_rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_rotary_base,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
            rope_scaling=False,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

    def shared_embedding_or_output_weight(self):
        """Surface the language model's word embeddings for gradient all-reduce."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules."""
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_model.projection is not None:
            modules.append(self.vision_model.projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vision_data: torch.Tensor = None,
        vision_grid_thw: torch.Tensor = None,
        video_start_index: int = -1,
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor | None = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of Qwen3.5 model."""
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            raise NotImplementedError()

        if self.pre_process:
            vision_embeds = None
            deepstack_feature_lists = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                vision_embeds, deepstack_feature_lists = self.vision_model(
                    vision_data=vision_data,
                    grid_thw=vision_grid_thw,
                )
            else:
                vision_embeds = None
                deepstack_feature_lists = None

            if inference_params is not None:
                raise NotImplementedError()

            if use_inference_kv_cache:
                language_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,
                )
                combined_embeddings = language_embeddings
            elif vision_embeds is not None:
                if image_input_mask is not None:
                    image_input_mask = image_input_mask.T
                if video_input_mask is not None:
                    video_input_mask = video_input_mask.T

                if video_start_index == 0:
                    image_embeds = None
                    video_embeds = vision_embeds
                    visual_pos_masks = video_input_mask
                elif video_start_index == vision_embeds.shape[0]:
                    image_embeds = vision_embeds
                    video_embeds = None
                    visual_pos_masks = image_input_mask
                elif 0 < video_start_index < vision_embeds.shape[0]:
                    image_embeds = vision_embeds[:video_start_index]
                    video_embeds = vision_embeds[video_start_index:]
                    visual_pos_masks = torch.logical_or(image_input_mask, video_input_mask)
                else:
                    raise ValueError(
                        f"Expect video token start index in range [0, {vision_embeds.shape[0]}], "
                        f"but got {video_start_index}"
                    )

                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,
                    image_input_mask=image_input_mask,
                    video_input_mask=video_input_mask,
                    image_embeds=image_embeds,
                    video_embeds=video_embeds,
                )
            else:
                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,
                )
                visual_pos_masks = None
                deepstack_feature_lists = None
        else:
            combined_embeddings = None
            visual_pos_masks = None
            deepstack_feature_lists = None

        output = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_feature_lists,
            **(extra_block_kwargs or {}),
        )
        return output

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mRoPE position indices for Qwen3.5."""
        return get_rope_index(
            spatial_merge_size=self.config.spatial_merge_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

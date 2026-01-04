# Copyright (c) 2025, BAAI. All rights reserved.
#
# Adopted from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/megatron_patch/model/qwen2_vl/gpt_model.py. Below is the original copyright:
#   Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Literal, Optional
from torch import Tensor

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding


class QwenVLLanguageModelEmbedding(LanguageModelEmbedding):
    """Language model embeddings. Used for Qwen2.5-VL, inserting the image and video hidden states.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head. Defaults to 0.
        scatter_to_sequence_parallel (bool): Set to False to disable scatter of embedding
            across sequence parallel region. Defaults to True.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        num_tokentypes: int = 0,
        scatter_to_sequence_parallel: bool = False, # chage default to False
    ):
        assert scatter_to_sequence_parallel == False, "QwenVLLanguageModelEmbedding does not support scatter_to_sequence_parallel"
        super().__init__(config, vocab_size, max_sequence_length, position_embedding_type, num_tokentypes, scatter_to_sequence_parallel)


    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        tokentype_ids: int = None,
        image_input_mask: Tensor = None,
        video_input_mask: Tensor = None,
        image_embeds: Tensor = None,
        video_embeds: Tensor = None
    ) -> Tensor:
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        word_embeddings = self.word_embeddings(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        if not self.reduce_scatter_embeddings:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose(0, 1).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:
            if not self.reduce_scatter_embeddings:
                embeddings = embeddings.clone()
                if image_embeds is not None:
                    embeddings[image_input_mask] = image_embeds.to(embeddings.device, embeddings.dtype)
                if video_embeds is not None:
                    embeddings[video_input_mask] = video_embeds.to(embeddings.device, embeddings.dtype)
                embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = embeddings.clone()
            if image_embeds is not None:
                embeddings[image_input_mask] = image_embeds.to(embeddings.device, embeddings.dtype)
            if video_embeds is not None:
                embeddings[video_input_mask] = video_embeds.to(embeddings.device, embeddings.dtype)
            embeddings = self.embedding_dropout(embeddings)

        return embeddings


class QwenVLLanguageModel(GPTModel):
    """GPT Transformer language model, replace language embedding using QwenVLLanguageModelEmbedding.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
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
            'learned_absolute', 'rope', 'mrope', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
    ) -> None:
        super().__init__(config=config, transformer_layer_spec=transformer_layer_spec,
                         vocab_size=vocab_size, max_sequence_length=max_sequence_length,
                         pre_process=pre_process, post_process=post_process,
                         fp16_lm_cross_entropy=fp16_lm_cross_entropy,
                         parallel_output=parallel_output,
                         share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                         position_embedding_type=position_embedding_type,
                         rotary_percent=rotary_percent,
                         rotary_base=rotary_base,
                         rope_scaling=rope_scaling,
                         rope_scaling_factor=rope_scaling_factor,
                         scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
                         seq_len_interpolation_factor=seq_len_interpolation_factor,
                         mtp_block_spec=mtp_block_spec)
        if self.pre_process:
            self.embedding = QwenVLLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

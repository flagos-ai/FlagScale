# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""BAGEL multimodal model for unified image understanding and generation.

BAGEL uses a Mixture-of-Transformers (MoT) architecture where understanding
(text + ViT) and generation (VAE latent) tokens have separate projections
but share the core attention computation.
"""
import re
import math
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn

from megatron.core import tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mot import MoTPackedSeqParams, create_packed_block_mask
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import log_single_rank

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TEFusedMLP, TENorm
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
else:
    TEFusedMLP, TENorm, TESpecProvider = None, None, None

try:
    import apex  # type: ignore[import-untyped]  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False

from flagscale.models.megatron.bagel.models import AutoEncoderParams, AutoEncoder


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Generate 2D sine-cosine positional embeddings (from DiT)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    half = embed_dim // 2
    embs = []
    for g in [grid[0], grid[1]]:
        omega = np.arange(half // 2, dtype=np.float64)
        omega /= half / 2.0
        omega = 1.0 / 10000 ** omega
        pos = g.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        embs.append(np.concatenate([np.sin(out), np.cos(out)], axis=1))
    return np.concatenate(embs, axis=1)


class TimestepEmbedder(MegatronModule):
    """Embeds scalar timesteps into vector representations (from DiT).

    Uses sinusoidal frequency encoding followed by a 2-layer MLP.
    """

    def __init__(self, config: TransformerConfig, frequency_embedding_size: int = 256):
        super().__init__(config=config)
        hidden_size = config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(t.dtype))


class PositionEmbedding(MegatronModule):
    """Frozen 2D sin-cos position embeddings for VAE latents and ViT tokens."""

    def __init__(self, config: TransformerConfig, max_num_patch_per_side: int):
        super().__init__(config=config)
        hidden_size = config.hidden_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side ** 2, hidden_size),
            requires_grad=False,
        )
        pos_embed = _get_2d_sincos_pos_embed(hidden_size, max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.pos_embed[position_ids]


class BagelModel(MegatronModule):
    """BAGEL multimodal model for unified image understanding and generation.

    Follows the LLaVA model pattern with added generation capabilities through
    Mixture-of-Transformers (MoT) layers and VAE latent processing.

    Args:
        language_transformer_config: Config for the language model backbone.
        language_transformer_layer_spec: Layer spec for decoder layers (MoT spec).
        language_vocab_size: Vocabulary size for the language model.
        language_max_sequence_length: Maximum sequence length.
        vision_model_spec: Complete ModuleSpec for the SigLIP vision encoder.
        connector_spec: Complete ModuleSpec for the vision-to-language connector.
        visual_gen: Enable visual generation path.
        visual_und: Enable visual understanding path.
        latent_patch_size: Patch size for patchifying VAE latents.
        max_latent_size: Maximum latent grid size (one side).
        vit_max_num_patch_per_side: Max ViT patches per side for position embedding.
        latent_channel: Number of VAE latent channels.
        vae_downsample: VAE spatial downsampling factor.
        timestep_shift: Flow matching timestep shift parameter.
        pre_process: Include embedding layer (pipeline parallel).
        post_process: Include output layer (pipeline parallel).
        parallel_output: Keep outputs split across TP ranks.
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_model_spec: ModuleSpec,
        connector_spec: ModuleSpec,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = 'learned_absolute',
        language_rotary_percent: float = 1.0,
        language_rotary_base: int = 10000,
        language_rope_scaling: bool = False,
        language_rope_scaling_factor: float = 8.0,
        visual_gen: bool = True,
        visual_und: bool = True,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        vit_max_num_patch_per_side: int = 70,
        latent_channel: int = 16,
        vae_downsample: int = 16,
        timestep_shift: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        dualpipev_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        if has_config_logger_enabled(language_transformer_config):
            log_config_to_disk(language_transformer_config, locals(), prefix=type(self).__name__)

        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            "Bagel is work in progress. Features are missing and methods can change.",
        )

        assert pre_process is True and post_process is True
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        # Optional BAGEL paths are always defined. The legacy visual flags only
        # control construction; runtime capability checks use the modules.
        self.vision_model = None
        self.connector = None
        self.vit_pos_embed = None
        self.vae_model = None
        self.vae2llm = None
        self.llm2vae = None
        self.time_embedder = None
        self.latent_pos_embed = None
        self.final_layernorm_gen = None

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        # --- Language Model (GPTModel with MoT layers) ---
        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=language_position_embedding_type,
            rotary_percent=language_rotary_percent,
            pre_process=self.pre_process,
            post_process=False,  # We handle post-processing ourselves
            rotary_base=language_rotary_base,
            rope_scaling=language_rope_scaling,
            rope_scaling_factor=language_rope_scaling_factor,
            scatter_embedding_sequence_parallel=False,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )
        assert not self.language_model.config.multi_latent_attention
        assert self.language_model.position_embedding_type == "rope"

        # --- Vision Encoder (SigLIP ViT) ---
        if self.visual_und:
            self.vision_model = build_module(
                vision_model_spec,
                pg_collection=self.pg_collection,
                vp_stage=vp_stage,
            )
            self.vision_model.convert_conv2d_to_linear()

            # Vision projection: ViT hidden -> LLM hidden
            self.connector = build_module(
                connector_spec,
                tp_group=self.pg_collection.tp,
            )

            # 2D sincos position embedding for ViT tokens
            self.vit_pos_embed = PositionEmbedding(
                config=language_transformer_config,
                max_num_patch_per_side=vit_max_num_patch_per_side,
            )

        # --- Visual Generation components ---
        if self.visual_gen:
            vae_params = AutoEncoderParams(
                resolution=256,
                in_channels=3,
                downsample=8,
                ch=128,
                out_ch=3,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                z_channels=16,
                scale_factor=0.3611,
                shift_factor=0.1159,
            )

            # Loading the autoencoder
            self.vae_model = AutoEncoder(vae_params)
            # The decoder is not used by the training forward. Keep the VAE in
            # eval mode and exclude decoder parameters from optimization to
            # avoid permanent unused parameters.
            self.vae_model.eval()
            self.vae_model.decoder.requires_grad_(False)

            self.latent_patch_size = latent_patch_size
            self.max_latent_size = max_latent_size
            self.latent_channel = latent_channel
            self.vae_downsample = vae_downsample
            self.timestep_shift = timestep_shift
            self.patch_latent_dim = latent_patch_size ** 2 * latent_channel

            # VAE latent <-> LLM space projections
            self.vae2llm = nn.Linear(self.patch_latent_dim, language_transformer_config.hidden_size)
            self.llm2vae = nn.Linear(language_transformer_config.hidden_size, self.patch_latent_dim)

            # Timestep and position embeddings for generation
            self.time_embedder = TimestepEmbedder(
                config=language_transformer_config,
            )
            self.latent_pos_embed = PositionEmbedding(
                config=language_transformer_config,
                max_num_patch_per_side=max_latent_size,
            )

            # Initialize llm2vae to zero (model starts with no generation output)
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

        # --- LM head for text prediction ---
        layer_norm_impl = TENorm if HAVE_TE else LNImpl
        self.final_layernorm = layer_norm_impl(
            config=language_transformer_config,
            hidden_size=language_transformer_config.hidden_size,
            eps=language_transformer_config.layernorm_epsilon,
        )
        if self.vae_model is not None:
            self.final_layernorm_gen = layer_norm_impl(
                config=language_transformer_config,
                hidden_size=language_transformer_config.hidden_size,
                eps=language_transformer_config.layernorm_epsilon,
            )

        self.output_layer = tensor_parallel.ColumnParallelLinear(
            language_transformer_config.hidden_size,
            language_vocab_size,
            config=language_transformer_config,
            init_method=language_transformer_config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not parallel_output,
            skip_weight_param_allocation=self.pre_process
            and self.share_embeddings_and_output_weights,
            embedding_activation_buffer=None,
            grad_output_buffer=None,
            tp_group=self.pg_collection.tp,
        )
        self.output_layer.weight.is_embedding_or_output_parameter = True

    def train(self, mode: bool = True):
        """Set training mode while keeping the pretrained VAE in eval mode."""
        super().train(mode)
        if self.vae_model is not None:
            self.vae_model.eval()
        return self

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for BagelModel'

        self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_vae_model: bool = False,
        freeze_text_embed: bool = False,
        freeze_connect: bool = False,
        freeze_und: bool = False,
        freeze_gen: bool = False,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_connect (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model:
            self.language_model.eval()
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            self.vision_model.eval()
            modules.append(self.vision_model)
        if freeze_vae_model and self.vae_model is not None:
            self.vae_model.eval()
            modules.append(self.vae_model)
        if freeze_text_embed and self.language_model:
            modules.append(self.language_model.embedding.word_embeddings)
            modules.append(self.output_layer)
        if freeze_connect:
            if self.connector is not None:
                modules.append(self.connector)
            if self.vae_model is not None:
                if self.time_embedder is not None:
                    modules.append(self.time_embedder)
                if self.vae2llm is not None:
                    modules.append(self.vae2llm)
                if self.llm2vae is not None:
                    modules.append(self.llm2vae)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        freeze_pattern = []
        if freeze_und:
            freeze_pattern.extend([
                r"language_model.*self_attention\.linear_qkv\.",
                r"language_model.*self_attention\.linear_proj\.",
                r"language_model.*self_attention\.q_layernorm\.",
                r"language_model.*self_attention\.k_layernorm\.",
                r"language_model.*\.mlp\.",
                r"language_model.*\.input_layernorm\.",
                r"language_model.*\.pre_mlp_layernorm\.",
            ])

        if freeze_gen:
            freeze_pattern.extend([
                r"language_model.*self_attention\.linear_qkv_gen\.",
                r"language_model.*self_attention\.linear_proj_gen\.",
                r"language_model.*self_attention\.q_layernorm_gen\.",
                r"language_model.*self_attention\.k_layernorm_gen\.",
                r"language_model.*\.mlp_gen\.",
                r"language_model.*\.input_layernorm_gen\.",
                r"language_model.*\.pre_mlp_layernorm_gen\.",
            ])

        print(f"{freeze_und=}, {freeze_gen=}, {freeze_pattern=}")
        if freeze_pattern:
            freeze_pattern = re.compile("|".join(freeze_pattern))
            for name, param in self.named_parameters():
                if freeze_pattern.search(name):
                    param.requires_grad = False
                    print(f"Freeze: {name}")

    def _embed_vision_tokens(
        self,
        pixel_values: Tensor,
        position_ids: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Tensor:
        """Encode vision tokens and map them into the language hidden space."""
        assert self.vision_model is not None
        assert self.connector is not None
        assert self.vit_pos_embed is not None

        vision_dtype = next(self.vision_model.parameters()).dtype
        image_embeddings = self.vision_model(
            packed_pixel_values=pixel_values.to(vision_dtype),
            packed_flattened_position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        image_embeddings = self.connector(image_embeddings)

        position_embeddings = self.vit_pos_embed(position_ids)
        if position_embeddings.dim() == 2:
            position_embeddings = position_embeddings.unsqueeze(1)
        return image_embeddings + position_embeddings

    def _patchify_vae_latents(
        self,
        padded_latents: Tensor,
        patchified_shapes: List,
    ) -> Tensor:
        """Convert padded VAE feature maps into packed latent tokens."""
        patch_size = self.latent_patch_size
        packed_latent_parts = []
        for latent, (height, width) in zip(padded_latents, patchified_shapes):
            latent = latent[
                :, : height * patch_size, : width * patch_size
            ].reshape(
                self.latent_channel,
                height,
                patch_size,
                width,
                patch_size,
            )
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(
                -1, self.patch_latent_dim
            )
            packed_latent_parts.append(latent)
        return torch.cat(packed_latent_parts, dim=0)

    def _apply_flow_matching_noise(
        self,
        clean_latents: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the BAGEL flow-matching noise schedule to packed latents."""
        noise = torch.randn_like(clean_latents)
        processed_timesteps = torch.sigmoid(timesteps.to(clean_latents.dtype))
        processed_timesteps = (
            self.timestep_shift
            * processed_timesteps
            / (1 + (self.timestep_shift - 1) * processed_timesteps)
        )
        noisy_latents = (
            (1 - processed_timesteps[:, None]) * clean_latents
            + processed_timesteps[:, None] * noise
        )
        return noisy_latents, noise, processed_timesteps

    def _embed_generation_tokens(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Project noisy VAE latents and add timestep and position embeddings."""
        assert self.vae2llm is not None
        assert self.time_embedder is not None
        assert self.latent_pos_embed is not None
        return (
            self.vae2llm(noisy_latents)
            + self.time_embedder(timesteps)
            + self.latent_pos_embed(position_ids)
        )

    def _build_mot_routing(
        self,
        packed_text_indexes: Tensor,
        packed_vit_token_indexes: Optional[Tensor],
        packed_vae_token_indexes: Optional[Tensor],
        sample_lens: Optional[List[int]],
        split_lens: Optional[List[int]],
        attn_modes: Optional[List[str]],
        device: torch.device,
    ) -> tuple[MoTPackedSeqParams, Tensor, Tensor]:
        """Build BAGEL branch routing and its optional packed attention mask."""
        packed_und_token_indexes = packed_text_indexes
        if packed_vit_token_indexes is not None:
            packed_und_token_indexes = torch.cat(
                [packed_text_indexes, packed_vit_token_indexes],
                dim=0,
            )

        if packed_vae_token_indexes is None:
            packed_vae_token_indexes = packed_und_token_indexes.new_empty(0)

        block_mask = None
        if (
            sample_lens is not None
            and split_lens is not None
            and attn_modes is not None
        ):
            block_mask = create_packed_block_mask(
                sample_lens=sample_lens,
                split_lens=split_lens,
                attn_modes=attn_modes,
                device=device,
            )

        packed_seq_params = MoTPackedSeqParams(
            branch_token_indexes={
                "und": packed_und_token_indexes,
                "gen": packed_vae_token_indexes,
            },
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            block_mask=block_mask,
        )
        return packed_seq_params, packed_und_token_indexes, packed_vae_token_indexes

    def _apply_task_layernorms(
        self,
        hidden_states: Tensor,
        und_token_indexes: Tensor,
        gen_token_indexes: Tensor,
        has_generation_path: bool,
    ) -> Tensor:
        """Normalize decoder outputs with the task-specific final norms."""
        if hidden_states.dim() != 3 or hidden_states.size(1) != 1:
            raise ValueError(
                "Expected decoder output with shape [num_tokens, 1, hidden_size], "
                f"but got {tuple(hidden_states.shape)}"
            )

        hidden_states = hidden_states.squeeze(1)
        normalized_hidden_states = torch.zeros_like(hidden_states)
        normalized_hidden_states[und_token_indexes] = apply_module(
            self.final_layernorm
        )(hidden_states[und_token_indexes])

        if has_generation_path:
            assert self.final_layernorm_gen is not None
            normalized_hidden_states[gen_token_indexes] = apply_module(
                self.final_layernorm_gen
            )(hidden_states[gen_token_indexes])

        return normalized_hidden_states

    @staticmethod
    def _has_trainable_params(module: Optional[nn.Module]) -> bool:
        """Return whether a module has any parameter participating in training."""
        return module is not None and any(
            parameter.requires_grad for parameter in module.parameters()
        )

    def _compute_ce_loss(
        self,
        hidden_states: Tensor,
        labels: Optional[Tensor],
        loss_indexes: Optional[Tensor],
    ) -> tuple[Optional[Tensor], Tensor]:
        """Compute the text loss and its empty-task graph contribution."""
        dummy_loss = hidden_states.new_zeros(())
        has_ce_tokens = loss_indexes is not None and loss_indexes.numel() > 0
        if has_ce_tokens and labels is not None and labels.numel() > 0:
            logits, _ = self.output_layer(hidden_states[loss_indexes])
            ce_loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)
            return ce_loss, dummy_loss

        if self._has_trainable_params(self.output_layer):
            dummy_logits, _ = self.output_layer(hidden_states[:1])
            dummy_loss = dummy_logits.sum() * 0.0
        return None, dummy_loss

    def _compute_mse_loss(
        self,
        hidden_states: Tensor,
        loss_indexes: Optional[Tensor],
        clean_latents: Optional[Tensor],
        noise: Optional[Tensor],
        timesteps: Optional[Tensor],
        has_generation_path: bool,
    ) -> tuple[Optional[Tensor], Tensor]:
        """Compute the latent loss and its empty-task graph contribution."""
        dummy_loss = hidden_states.new_zeros(())
        has_mse_tokens = loss_indexes is not None and loss_indexes.numel() > 0
        if has_generation_path and has_mse_tokens and clean_latents is not None:
            assert self.llm2vae is not None
            assert noise is not None
            assert timesteps is not None
            predictions = self.llm2vae(hidden_states[loss_indexes])
            target = noise - clean_latents
            has_mse = timesteps > 0
            mse_loss = (predictions - target[has_mse]) ** 2
            return mse_loss, dummy_loss

        if has_generation_path and self._has_trainable_params(self.llm2vae):
            assert self.llm2vae is not None
            dummy_predictions = self.llm2vae(hidden_states[:1])
            dummy_loss = dummy_predictions.sum() * 0.0
        return None, dummy_loss

    def forward(
        self,
        # Core packed text fields (always present)
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # Sequence packing metadata
        sequence_length: Optional[int] = None,
        sample_lens: Optional[List[int]] = None,
        split_lens: Optional[List[int]] = None,
        attn_modes: Optional[List[str]] = None,
        # CE loss fields
        packed_label_ids: Optional[Tensor] = None,
        ce_loss_indexes: Optional[Tensor] = None,
        # ViT understanding inputs
        packed_vit_tokens: Optional[Tensor] = None,
        packed_vit_token_indexes: Optional[Tensor] = None,
        packed_vit_position_ids: Optional[Tensor] = None,
        vit_token_seqlens: Optional[Tensor] = None,
        # VAE generation inputs
        padded_images: Optional[Tensor] = None,
        # padded_latent: Optional[Tensor] = None,
        # packed_latent_clean: Optional[Tensor] = None,
        # packed_latent_noise: Optional[Tensor] = None,
        # packed_timesteps_effective: Optional[Tensor] = None,
        patchified_vae_latent_shapes: Optional[List] = None,
        packed_latent_position_ids: Optional[Tensor] = None,
        packed_vae_token_indexes: Optional[Tensor] = None,
        packed_timesteps: Optional[Tensor] = None,
        mse_loss_indexes: Optional[Tensor] = None,
        # Standard args
        inference_context: BaseInferenceContext = None,
        packed_seq_params: Optional[MoTPackedSeqParams] = None,
        extra_block_kwargs: Optional[dict] = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Dict[str, Optional[Tensor]]:
        """Forward pass for BAGEL model.

        Returns:
            Dictionary with 'ce' and 'mse' loss tensors (or None).
        """

        # --- Step 1: Get text embeddings from language model embedding ---
        # Use language model's embedding to get text token embeddings
        # GPTModel.embedding gives us [seq, batch, hidden]
        if packed_text_ids.dim() == 1:
            packed_text_ids = packed_text_ids.unsqueeze(0)
        elif packed_text_ids.dim() == 2:
            if packed_text_ids.size(0) != 1:
                raise ValueError(
                    "Bagel packed input currently requires batch size 1, "
                    f"but got {packed_text_ids.size(0)}"
                )
        else:
            raise ValueError(
                "packed_text_ids must have shape [seq] or [1, seq], "
                f"but got {tuple(packed_text_ids.shape)}"
            )

        # print(f"{packed_text_ids.shape=}, {torch.sum(packed_text_ids, dtype=torch.float32)=}, {packed_text_ids=}")
        text_embeddings = self.language_model.embedding(
            input_ids=packed_text_ids, position_ids=packed_position_ids.unsqueeze(0),
        )
        # print(f"{packed_text_ids.shape=}, {text_embeddings.shape=}")

        # Create the combined sequence tensor
        # For packed sequences: [total_seq, 1, hidden]
        combined_embeddings = text_embeddings.new_zeros(size=(sequence_length, self.language_model.config.hidden_size))
        # print(f"{packed_text_indexes.shape=}, {combined_embeddings.shape=}")
        combined_embeddings[packed_text_indexes] = text_embeddings.squeeze(1)
        # print(f"{text_embeddings.shape=}, {torch.sum(text_embeddings, dtype=torch.float32)=}, {text_embeddings=}")

        # --- Step 2: Process ViT tokens (understanding path) ---
        has_vision_path = (
            self.vision_model is not None
            and self.connector is not None
            and self.vit_pos_embed is not None
        )
        has_vit_tokens = (
            has_vision_path
            and packed_vit_tokens is not None
            and packed_vit_token_indexes is not None
            and packed_vit_token_indexes.numel() > 0
            and packed_vit_position_ids is not None
            and vit_token_seqlens is not None
        )
        if has_vit_tokens:
            # Get ViT embeddings
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            # [num_patches, 1, vit_hidden]
            # print(f"{packed_vit_tokens.shape=}, {packed_vit_tokens.shape=}")
            image_embeddings = self._embed_vision_tokens(
                pixel_values=packed_vit_tokens,
                position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            # print(f"after vit {image_embeddings.shape=}, {torch.sum(image_embeddings, dtype=torch.float32)=}, {image_embeddings=}")
            # print(f"{image_embeddings.shape=}")

            if image_embeddings.dtype != combined_embeddings.dtype:
                image_embeddings = image_embeddings.to(combined_embeddings.dtype)
            combined_embeddings[packed_vit_token_indexes] = image_embeddings.squeeze(1)
            # print(f"{image_embeddings.shape=}, {torch.sum(image_embeddings, dtype=torch.float32)=}, {image_embeddings=}")
        elif has_vision_path:
            # Keep trainable vision modules in the autograd graph when this
            # microbatch contains no vision-understanding task.
            vision_dtype = next(self.vision_model.parameters()).dtype
            dummy_pixels = torch.zeros(
                1,
                self.vision_model.patch_dim * self.vision_model.patch_dim * 3,
                device=combined_embeddings.device,
                dtype=vision_dtype,
            )
            dummy_position_ids = torch.zeros(
                1, device=combined_embeddings.device, dtype=torch.long
            )
            dummy_cu_seqlens = torch.tensor(
                [0, 1], device=combined_embeddings.device, dtype=torch.int32
            )
            dummy_image_embeddings = self._embed_vision_tokens(
                pixel_values=dummy_pixels,
                position_ids=dummy_position_ids,
                cu_seqlens=dummy_cu_seqlens,
                max_seqlen=1,
            )
            combined_embeddings[0] = combined_embeddings[0] * 1.0 + dummy_image_embeddings[0] * 0.0

        # --- Step 3: Process VAE latents (generation path) ---
        packed_latent_clean = None
        packed_latent_noise = None
        packed_timesteps_effective = None
        noise = None
        processed_timesteps = None
        has_generation_path = (
            self.vae_model is not None
            and self.vae2llm is not None
            and self.llm2vae is not None
            and self.time_embedder is not None
            and self.latent_pos_embed is not None
            and self.final_layernorm_gen is not None
        )
        has_gen_tokens = (
            has_generation_path
            and (
                packed_latent_clean is not None
                or padded_latent is not None
                or padded_images is not None
            )
            and packed_vae_token_indexes is not None
            and packed_vae_token_indexes.numel() > 0
            and patchified_vae_latent_shapes is not None
            and packed_latent_position_ids is not None
            and packed_timesteps is not None
        )
        if has_gen_tokens:
            if packed_latent_clean is None:
                if padded_latent is None:
                    vae_dtype = next(self.vae_model.parameters()).dtype
                    padded_latent = self.vae_model.encode(padded_images.to(vae_dtype))
                packed_latent_clean = self._patchify_vae_latents(
                    padded_latent,
                    patchified_vae_latent_shapes,
                )
            if packed_latent_noise is not None:
                noise = packed_latent_noise.to(packed_latent_clean.dtype)
                if packed_timesteps_effective is None:
                    processed_timesteps = torch.sigmoid(
                        packed_timesteps.to(packed_latent_clean.dtype)
                    )
                    processed_timesteps = (
                        self.timestep_shift
                        * processed_timesteps
                        / (1 + (self.timestep_shift - 1) * processed_timesteps)
                    )
                else:
                    processed_timesteps = packed_timesteps_effective.to(
                        packed_latent_clean.dtype
                    )
                packed_latent = (
                    (1 - processed_timesteps[:, None]) * packed_latent_clean
                    + processed_timesteps[:, None] * noise
                )
            else:
                packed_latent, noise, processed_timesteps = (
                    self._apply_flow_matching_noise(
                        packed_latent_clean,
                        packed_timesteps,
                    )
                )
            packed_latent = self._embed_generation_tokens(
                packed_latent,
                processed_timesteps,
                packed_latent_position_ids,
            )
            combined_embeddings[packed_vae_token_indexes] = packed_latent.to(
                combined_embeddings.dtype
            )
            # print(f"{packed_latent.shape=}, {torch.sum(packed_latent, dtype=torch.float32)=}, {packed_latent=}")
        elif has_generation_path:
            # Match the real generation path with one dummy image so the VAE
            # encoder and all generation-input modules stay in the graph.
            vae_dtype = next(self.vae_model.parameters()).dtype
            dummy_image_size = self.vae_downsample * self.latent_patch_size
            dummy_image = torch.zeros(
                1,
                3,
                dummy_image_size,
                dummy_image_size,
                device=combined_embeddings.device,
                dtype=vae_dtype,
            )
            dummy_padded_latent = self.vae_model.encode(dummy_image)
            dummy_latent_clean = self._patchify_vae_latents(
                dummy_padded_latent,
                [(1, 1)],
            )
            dummy_timesteps = torch.zeros(
                1,
                device=combined_embeddings.device,
                dtype=dummy_latent_clean.dtype,
            )
            dummy_latent, _, dummy_timesteps = self._apply_flow_matching_noise(
                dummy_latent_clean,
                dummy_timesteps,
            )
            dummy_position_ids = torch.zeros(
                1,
                device=combined_embeddings.device,
                dtype=torch.long,
            )
            dummy_gen_embedding = self._embed_generation_tokens(
                dummy_latent,
                dummy_timesteps,
                dummy_position_ids,
            )
            combined_embeddings[0] = combined_embeddings[0] * 1.0 + dummy_gen_embedding[0] * 0.0

        decoder_input = combined_embeddings.unsqueeze(1)

        max_pos = int(packed_position_ids.max().item()) + 1
        rotary_pos_emb_full = self.language_model.rotary_pos_emb(
            max_pos,
            packed_seq=False,
            cp_group=None,
        )  # [max_pos, 1, 1, rot_dim]
        rotary_pos_emb = rotary_pos_emb_full[packed_position_ids]  # [total_tokens, 1, 1, rot_dim]

        # --- Step 4: Run through language model decoder ---
        # Build MoTPackedSeqParams for MoT layers
        (
            packed_seq_params,
            packed_und_token_indexes,
            packed_vae_token_indexes,
        ) = self._build_mot_routing(
            packed_text_indexes=packed_text_indexes,
            packed_vit_token_indexes=packed_vit_token_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            device=combined_embeddings.device,
        )
        # print(f"{packed_und_token_indexes.shape=}")

        # Run through GPTModel with decoder_input (skips embedding)
        hidden_states = self.language_model.decoder(
            hidden_states=decoder_input,
            attention_mask=None,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            rotary_pos_cos_sin=None,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=None,
            **(extra_block_kwargs or {}),
        )

        # --- Step 5: Compute losses ---
        last_hidden_states = self._apply_task_layernorms(
            hidden_states=hidden_states,
            und_token_indexes=packed_und_token_indexes,
            gen_token_indexes=packed_vae_token_indexes,
            has_generation_path=has_generation_path,
        )

        ce_loss, ce_dummy_loss = self._compute_ce_loss(
            hidden_states=last_hidden_states,
            labels=packed_label_ids,
            loss_indexes=ce_loss_indexes,
        )
        mse_loss, mse_dummy_loss = self._compute_mse_loss(
            hidden_states=last_hidden_states,
            loss_indexes=mse_loss_indexes,
            clean_latents=packed_latent_clean,
            noise=noise,
            timesteps=processed_timesteps,
            has_generation_path=has_generation_path,
        )
        dummy_loss = last_hidden_states.new_zeros((), requires_grad=True)
        dummy_loss = dummy_loss + ce_dummy_loss + mse_dummy_loss

        return ce_loss, mse_loss, dummy_loss

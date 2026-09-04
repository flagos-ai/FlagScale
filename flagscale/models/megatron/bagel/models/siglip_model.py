# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SigLIP Vision Model with NaViT-style dynamic resolution support.

Reference: /share/project/zhaoyingli/codes/Bagel/modeling/bagel/siglip_navit.py

This model accepts packed, variable-length image patches from different
resolutions and processes them with variable-length attention via PackedSeqParams.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TENorm

    NORM_IMPL = TENorm
except ImportError:
    NORM_IMPL = torch.nn.LayerNorm


class SiglipVisionModel(VisionModule):
    """SigLIP ViT with NaViT-style dynamic resolution.

    Accepts pre-patchified pixel values packed into a single sequence,
    enabling variable-length attention across images of different resolutions.

    Args:
        transformer_config (TransformerConfig): Transformer config for ViT layers.
        transformer_layer_spec (ModuleSpec): Layer spec for transformer layers.
        patch_dim (int): Patch size (pixels per side).
        max_num_patches_per_side (int): Max patches per spatial dimension,
            determines position embedding table size (max_num_patches_per_side^2).
        ln_post_impl: Layer norm implementation for post-norm.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_dim: int = 14,
        image_size: int = 336,
        ln_post_impl=NORM_IMPL,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=transformer_config)

        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.image_size = image_size
        self.pg_collection = pg_collection
        self.vp_stage = vp_stage
        self.use_rope = bool(getattr(transformer_config, "rope", False))

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
            padding="valid",
        )

        max_num_patches_per_side = self.image_size // self.patch_dim
        if self.use_rope:
            raise NotImplementedError(
                "SiglipVisionModel rope=True requires Bagel's split H/W 2D RoPE "
                "inside vision self-attention; Megatron's standard ViT attention "
                "does not implement that layout yet."
            )
        else:
            # Match Bagel siglip_navit.py with config.rope=False.
            num_positions = max_num_patches_per_side * max_num_patches_per_side
            self.position_embeddings = nn.Embedding(
                num_positions,
                self.visual_hidden_size,
                dtype=transformer_config.params_dtype,
            )

        # Post layer norm (SigLIP style: no pre-LN, has post-LN)
        self.ln_post = build_module(
            ln_post_impl,
            config=transformer_config,
            hidden_size=self.visual_hidden_size,
            eps=transformer_config.layernorm_epsilon,
        )

        self.model_type = ModelType.encoder_or_decoder

        # Transformer encoder
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model."""
        self.decoder.set_input_tensor(input_tensor)

    def convert_conv2d_to_linear(self) -> None:
        # 3 is num_channels
        linear_patch_embedding = nn.Linear(
            self.patch_dim * self.patch_dim * 3, self.visual_hidden_size, bias=True,
        )
        W = self.patch_embedding.weight.permute(0, 2, 3, 1).reshape(
            self.visual_hidden_size, 3 * self.patch_dim ** 2
        )
        linear_patch_embedding.weight.data.copy_(W)
        linear_patch_embedding.bias.data.copy_(self.patch_embedding.bias.data)
        del self.patch_embedding
        self.patch_embedding = linear_patch_embedding

    def forward(
        self,
        packed_pixel_values: Tensor,
        packed_flattened_position_ids: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Tensor:
        """Forward pass with NaViT-style packed inputs.

        Args:
            packed_pixel_values: [total_patches, patch_dim^2 * 3]
                Pre-patchified pixel values from all images, concatenated.
            packed_flattened_position_ids: [total_patches]
                2D position ids flattened to 1D (row * max_num_patches_per_side + col).
            cu_seqlens: [num_images + 1], int32
                Cumulative sequence lengths for variable-length attention.
            max_seqlen: int
                Maximum number of patches in any single image.

        Returns:
            Tensor of shape [total_patches, 1, hidden_size].
        """
        # Patch embedding
        x = self.patch_embedding(packed_pixel_values)  # [total_patches, hidden_size]

        # Bagel uses learned absolute position embeddings when config.rope=False.
        if not self.use_rope:
            x = x + self.position_embeddings(packed_flattened_position_ids)

        # Reshape for TransformerBlock: [seq_len, batch=1, hidden_size]
        x = x.unsqueeze(1).contiguous()

        # print(f"{cu_seqlens=}, {max_seqlen=}")
        # Build PackedSeqParams for variable-length attention
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            qkv_format='thd',
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

        # print(f"{x.shape=}")
        # Transformer forward
        # print(f"before encoder: {x.shape=}, {torch.sum(x, dtype=torch.float32)=}, {x=}, {packed_seq_params=}")
        x = self.decoder(
            hidden_states=x,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )

        # Post layer norm
        x = self.ln_post(x)

        return x

# Copyright (c) 2025, BAAI. All rights reserved.
# Sample type definitions for Bagel-Energon integration.

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BagelSample:
    """Intermediate sample representation after encode_sample, before packing.

    This is what goes into the PackingDataset buffer.
    """

    image_tensor_list: list[torch.Tensor]  # ViT/VAE preprocessed images
    text_ids_list: list[list[int]]  # Tokenized text segments
    sequence_plan: list[dict[str, Any]]  # Packing instructions per segment
    num_tokens: int  # Total token count for this sample
    is_mandatory: bool = False  # Whether this sample must appear in every pack
    subflavor: str = ""  # Task type: "t2i" / "vlm"
    # Energon metadata
    __key__: str = ""
    __restore_key__: tuple = ()


@dataclass
class BagelPackedBatch:
    """Output of pack_selected_samples — a fully packed sequence ready for the model."""

    sequence_length: int
    sample_lens: list[int]
    packed_text_ids: torch.Tensor
    packed_text_indexes: torch.Tensor
    packed_position_ids: torch.Tensor
    # FlexAttention fields
    split_lens: list[int] = field(default_factory=list)
    attn_modes: list[str] = field(default_factory=list)
    # VAE image generation (optional)
    padded_images: torch.Tensor | None = None
    patchified_vae_latent_shapes: list | None = None
    packed_latent_position_ids: torch.Tensor | None = None
    packed_vae_token_indexes: torch.Tensor | None = None
    # ViT image understanding (optional)
    packed_vit_tokens: torch.Tensor | None = None
    packed_vit_position_ids: torch.Tensor | None = None
    packed_vit_token_indexes: torch.Tensor | None = None
    vit_token_seqlens: torch.Tensor | None = None
    # Diffusion timesteps (optional)
    packed_timesteps: torch.Tensor | None = None
    mse_loss_indexes: torch.Tensor | None = None
    # CE loss (optional)
    packed_label_ids: torch.Tensor | None = None
    ce_loss_indexes: torch.Tensor | None = None
    ce_loss_weights: torch.Tensor | None = None

    def to_dict(self):
        """Convert to dict for get_batch compatibility."""
        data = {
            "sequence_length": self.sequence_length,
            "sample_lens": self.sample_lens,
            "packed_text_ids": self.packed_text_ids,
            "packed_text_indexes": self.packed_text_indexes,
            "packed_position_ids": self.packed_position_ids,
            "split_lens": self.split_lens,
            "attn_modes": self.attn_modes,
        }
        if self.padded_images is not None:
            data["padded_images"] = self.padded_images
            data["patchified_vae_latent_shapes"] = self.patchified_vae_latent_shapes
            data["packed_latent_position_ids"] = self.packed_latent_position_ids
            data["packed_vae_token_indexes"] = self.packed_vae_token_indexes
        if self.packed_vit_tokens is not None:
            data["packed_vit_tokens"] = self.packed_vit_tokens
            data["packed_vit_position_ids"] = self.packed_vit_position_ids
            data["packed_vit_token_indexes"] = self.packed_vit_token_indexes
            data["vit_token_seqlens"] = self.vit_token_seqlens
        if self.packed_timesteps is not None:
            data["packed_timesteps"] = self.packed_timesteps
            data["mse_loss_indexes"] = self.mse_loss_indexes
        if self.packed_label_ids is not None:
            data["packed_label_ids"] = self.packed_label_ids
            data["ce_loss_indexes"] = self.ce_loss_indexes
            data["ce_loss_weights"] = self.ce_loss_weights
        return data

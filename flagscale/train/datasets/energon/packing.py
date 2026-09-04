# Copyright (c) 2026, BAAI. All rights reserved.

import random
from typing import Any

import numpy as np
import torch

from .data_utils import (
    len2weight,
    patchify,
)
from .sample_types import BagelPackedBatch, BagelSample


def make_sequence_status():
    """Initialize an empty sequence status dict for accumulating packed tokens."""
    return dict(
        curr=0,
        sample_lens=list(),
        packed_position_ids=list(),
        nested_attention_masks=list(),
        split_lens=list(),
        attn_modes=list(),
        packed_text_ids=list(),
        packed_text_indexes=list(),
        packed_label_ids=list(),
        ce_loss_indexes=list(),
        ce_loss_weights=list(),
        vae_image_tensors=list(),
        packed_latent_position_ids=list(),
        vae_latent_shapes=list(),
        packed_vae_token_indexes=list(),
        packed_timesteps=list(),
        mse_loss_indexes=list(),
        packed_vit_tokens=list(),
        vit_token_seqlens=list(),
        packed_vit_position_ids=list(),
        packed_vit_token_indexes=list(),
    )


def pack_sequence(
    sample: BagelSample,
    sequence_status: dict[str, Any],
    special_tokens: dict[str, int],
    data_config,
    get_flattened_position_ids,
) -> dict[str, Any]:
    """Pack a single sample into the sequence status.

    This is the core packing logic from Bagel's PackedDataset.pack_sequence,
    supporting text, vit_image, and vae_image modalities.
    """
    image_tensor_list = list(sample.image_tensor_list)
    text_ids_list = list(sample.text_ids_list)
    sequence_plan = sample.sequence_plan

    bos_token_id = special_tokens["bos_token_id"]
    eos_token_id = special_tokens["eos_token_id"]
    start_of_image_id = special_tokens["start_of_image_id"]
    end_of_image_id = special_tokens["end_of_image_id"]

    split_lens, attn_modes = list(), list()
    curr = sequence_status["curr"]
    curr_rope_id = 0
    sample_lens = 0

    for item in sequence_plan:
        split_start = item.get("split_start", True)
        if split_start:
            curr_split_len = 0

        if item["type"] == "text":
            text_ids = text_ids_list.pop(0)
            if item["enable_cfg"] == 1 and random.random() < data_config.text_cond_dropout_prob:
                continue

            shifted_text_ids = [bos_token_id, *text_ids]
            sequence_status["packed_text_ids"].extend(shifted_text_ids)
            sequence_status["packed_text_indexes"].extend(range(curr, curr + len(shifted_text_ids)))
            if item["loss"] == 1:
                sequence_status["ce_loss_indexes"].extend(range(curr, curr + len(shifted_text_ids)))
                sequence_status["ce_loss_weights"].extend(
                    [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                )
                sequence_status["packed_label_ids"].extend([*text_ids, eos_token_id])
            curr += len(shifted_text_ids)
            curr_split_len += len(shifted_text_ids)

            # <|im_end|> token
            sequence_status["packed_text_ids"].append(eos_token_id)
            sequence_status["packed_text_indexes"].append(curr)
            if item["special_token_loss"] == 1:
                sequence_status["ce_loss_indexes"].append(curr)
                sequence_status["ce_loss_weights"].append(1.0)
                sequence_status["packed_label_ids"].append(item["special_token_label"])
            curr += 1
            curr_split_len += 1

            attn_modes.append("causal")
            sequence_status["packed_position_ids"].extend(
                range(curr_rope_id, curr_rope_id + curr_split_len)
            )
            curr_rope_id += curr_split_len

        elif item["type"] == "vit_image":
            image_tensor = image_tensor_list.pop(0)
            if item["enable_cfg"] == 1 and random.random() < data_config.vit_cond_dropout_prob:
                curr_rope_id += 1
                continue

            # <|startofimage|>
            sequence_status["packed_text_ids"].append(start_of_image_id)
            sequence_status["packed_text_indexes"].append(curr)
            curr += 1
            curr_split_len += 1

            # Patchify image
            vit_tokens = patchify(image_tensor, data_config.vit_patch_size)
            num_img_tokens = vit_tokens.shape[0]
            sequence_status["packed_vit_token_indexes"].extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            curr_split_len += num_img_tokens

            sequence_status["packed_vit_tokens"].append(vit_tokens)
            sequence_status["vit_token_seqlens"].append(num_img_tokens)
            sequence_status["packed_vit_position_ids"].append(
                get_flattened_position_ids(
                    image_tensor.size(1),
                    image_tensor.size(2),
                    data_config.vit_patch_size,
                    max_num_patches_per_side=data_config.max_num_patch_per_side,
                )
            )

            # <|endofimage|>
            sequence_status["packed_text_ids"].append(end_of_image_id)
            sequence_status["packed_text_indexes"].append(curr)
            if item["special_token_loss"] == 1:
                sequence_status["ce_loss_indexes"].append(curr)
                sequence_status["ce_loss_weights"].append(1.0)
                sequence_status["packed_label_ids"].append(item["special_token_label"])
            curr += 1
            curr_split_len += 1

            attn_modes.append("full")
            sequence_status["packed_position_ids"].extend([curr_rope_id] * (num_img_tokens + 2))
            curr_rope_id += 1

        elif item["type"] == "vae_image":
            image_tensor = image_tensor_list.pop(0)
            if item["enable_cfg"] == 1 and random.random() < data_config.vae_cond_dropout_prob:
                curr_rope_id += 1
                continue

            # <|startofimage|>
            sequence_status["packed_text_ids"].append(start_of_image_id)
            sequence_status["packed_text_indexes"].append(curr)
            curr += 1
            curr_split_len += 1

            # Compute latent shape
            _, h, w = image_tensor.shape
            latent_h = min(h // data_config.vae_image_downsample, data_config.max_latent_size)
            latent_w = min(w // data_config.vae_image_downsample, data_config.max_latent_size)
            num_img_tokens = latent_h * latent_w

            sequence_status["packed_vae_token_indexes"].extend(range(curr, curr + num_img_tokens))
            if item["loss"] == 1:
                sequence_status["mse_loss_indexes"].extend(range(curr, curr + num_img_tokens))
                if split_start:
                    timestep = np.random.randn()
            else:
                timestep = float("-inf")

            sequence_status["packed_timesteps"].extend([timestep] * num_img_tokens)
            curr += num_img_tokens
            curr_split_len += num_img_tokens

            # <|endofimage|>
            sequence_status["packed_text_ids"].append(end_of_image_id)
            sequence_status["packed_text_indexes"].append(curr)
            if item["special_token_loss"] == 1:
                sequence_status["ce_loss_indexes"].append(curr)
                sequence_status["ce_loss_weights"].append(1.0)
                sequence_status["packed_label_ids"].append(item["special_token_label"])
            curr += 1
            curr_split_len += 1

            if split_start:
                if item["loss"] == 1 and "frame_delta" not in item.keys():
                    attn_modes.append("noise")
                else:
                    attn_modes.append("full")
            sequence_status["packed_position_ids"].extend([curr_rope_id] * (num_img_tokens + 2))
            if "frame_delta" in item.keys():
                curr_rope_id += item["frame_delta"]
            elif item["loss"] == 0:
                curr_rope_id += 1

            # Store image tensor and latent shape
            sequence_status["vae_image_tensors"].append(image_tensor)
            sequence_status["vae_latent_shapes"].append((latent_h, latent_w))
            sequence_status["packed_latent_position_ids"].append(
                get_flattened_position_ids(
                    latent_h * data_config.vae_image_downsample,
                    latent_w * data_config.vae_image_downsample,
                    data_config.vae_image_downsample,
                    max_num_patches_per_side=data_config.max_latent_size,
                )
            )

        if item.get("split_end", True):
            split_lens.append(curr_split_len)
            sample_lens += curr_split_len

    sequence_status["curr"] = curr
    sequence_status["sample_lens"].append(sample_lens)
    sequence_status["split_lens"].extend(split_lens)
    sequence_status["attn_modes"].extend(attn_modes)

    return sequence_status


def to_tensor(sequence_status: dict[str, Any], max_num_tokens: int) -> BagelPackedBatch:
    """Convert accumulated sequence_status into a BagelPackedBatch."""
    sequence_length = sum(sequence_status["sample_lens"])
    pad_len = max_num_tokens - sequence_length

    batch = BagelPackedBatch(
        sequence_length=sequence_length,
        sample_lens=sequence_status["sample_lens"] + [pad_len],
        packed_text_ids=torch.tensor(sequence_status["packed_text_ids"]),
        packed_text_indexes=torch.tensor(sequence_status["packed_text_indexes"]),
        packed_position_ids=torch.tensor(sequence_status["packed_position_ids"]),
        split_lens=sequence_status["split_lens"] + [pad_len],
        attn_modes=sequence_status["attn_modes"] + ["causal"],
    )

    # VAE images
    if len(sequence_status["vae_image_tensors"]) > 0:
        image_tensors = sequence_status["vae_image_tensors"]
        image_sizes = [item.shape for item in image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
        for i, image_tensor in enumerate(image_tensors):
            padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2]] = image_tensor

        batch.padded_images = padded_images
        batch.patchified_vae_latent_shapes = sequence_status["vae_latent_shapes"]
        batch.packed_latent_position_ids = torch.cat(
            sequence_status["packed_latent_position_ids"], dim=0
        )
        batch.packed_vae_token_indexes = torch.tensor(sequence_status["packed_vae_token_indexes"])

    # ViT tokens
    if len(sequence_status["packed_vit_tokens"]) > 0:
        batch.packed_vit_tokens = torch.cat(sequence_status["packed_vit_tokens"], dim=0)
        batch.packed_vit_position_ids = torch.cat(sequence_status["packed_vit_position_ids"], dim=0)
        batch.packed_vit_token_indexes = torch.tensor(sequence_status["packed_vit_token_indexes"])
        batch.vit_token_seqlens = torch.tensor(sequence_status["vit_token_seqlens"])

    # Diffusion timesteps
    if len(sequence_status["packed_timesteps"]) > 0:
        batch.packed_timesteps = torch.tensor(sequence_status["packed_timesteps"])
        batch.mse_loss_indexes = torch.tensor(sequence_status["mse_loss_indexes"])

    # CE loss
    if len(sequence_status["packed_label_ids"]) > 0:
        batch.packed_label_ids = torch.tensor(sequence_status["packed_label_ids"])
        batch.ce_loss_indexes = torch.tensor(sequence_status["ce_loss_indexes"])
        batch.ce_loss_weights = torch.tensor(sequence_status["ce_loss_weights"])

    return batch

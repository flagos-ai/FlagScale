# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import ast
import re
import math
import logging
from collections import namedtuple
from functools import partial
from typing import List, Optional

import torch

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt import GPTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args, get_tokenizer
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

from .clip_vit_model import CLIPViTModel, get_num_image_embeddings

IMAGE_TOKEN_INDEX = -200  # ID for images in the input sequence.
IGNORE_INDEX = -100  # ID for labels that should be ignored.


class LLaVAOneVisionModel(MegatronModule):
    """LLaVA OneVision multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Language model spec.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Vision model spec.
        drop_vision_class_token (bool): Drop vision class token(s) before the language model.
        vision_projection_config (TransformerConfig): Vision projection config.
        vision_projection_layer_spec (ModuleSpec): Vision projection spec.
        vision_projection_type (str): Type of the vision projection. Default: 2-layer MLP.
        allow_missing_vision_projection_checkpoint (bool): Allow vision projection weights to be
            missing when loading a checkpoint. Default False.
        parallel_output (bool): Keep outputs split across tensor parallel ranks.
            This is typically True for training and False for inference.
        language_position_embedding_type (str): Language model position embedding type.
        language_rotary_percent (float): RoPE percent. Defaults to 1.0.
        pre_process (bool): Include embedding layer in the decoder (used with pipeline parallel).
        post_process (bool): Include output layer in the decoder (used with pipeline parallel).
        add_encoder (bool): Construct the encoder (used with pipeline parallel).
            When we use pipelining, the encoder will live on only the first stage
        add_decoder (bool): Construct the decoder (used with pipeline parallel).
            When we use pipelining, the decoder will live on every stage after the first one.
        img_h (int): Input image height.
        img_w (int): Input image width.
        patch_dim (int): The size of each image patch side.
        language_rotary_base (int): RoPE base.
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        allow_missing_vision_projection_checkpoint: bool = False,
        parallel_output: bool = True,
        language_position_embedding_type: str = "learned_absolute",
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        img_h: int = 336,
        img_w: int = 336,
        patch_dim: int = 14,
        language_rotary_base: int = 10000,
        add_class_token: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)

        if has_config_logger_enabled(language_transformer_config):
            log_config_to_disk(
                language_transformer_config, locals(), prefix=type(self).__name__
            )

        logging.getLogger(__name__).warning(
            "LLaVA OneVision model is under active development. "
            "It may be missing features and its methods may change."
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None
        self.args = get_args()

        args = self.args
        # Init image_newline
        if "unpad" in args.mm_patch_merge_type:
            embed_std = 1 / torch.sqrt(
                torch.tensor(args.hidden_size, dtype=torch.bfloat16)
            )
            self.image_newline = torch.nn.Parameter(
                torch.randn(args.hidden_size, dtype=torch.bfloat16) * embed_std
            )

        # Add share_embeddings_and_output_weights to the language model.
        self.share_embeddings_and_output_weights = (
            not args.untie_embeddings_and_output_weights
        )

        if self.add_decoder:
            self.language_model = GPTModel(
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
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            )
            self._language_max_sequence_length = language_max_sequence_length
            self._language_is_pipeline_parallel = (
                language_transformer_config.pipeline_model_parallel_size > 1
            )

        class_token_len = 1
        if self.add_encoder:
            self.vision_model = CLIPViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                img_h=img_h,
                img_w=img_w,
                class_token_len=class_token_len,
                patch_dim=patch_dim,
                add_class_token=add_class_token,
                ln_post_impl=FusedLayerNorm,
            )
            self._drop_vision_class_token = drop_vision_class_token
            # Map (intermediate) vision model outputs to the language model input dimension.
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                vision_projection_type,
                vision_transformer_config.hidden_size,  # input size to the projection.
            )
            # Ignore missing weights for the vision projection during checkpoint loading.
            # This should be disabled by default but can be enabled if your checkpoint contains
            # pretrained vision and language models but not the projection from vision model
            # outputs to language model inputs.
            if allow_missing_vision_projection_checkpoint:
                vision_projection_param_names = [
                    f"vision_projection.{name}"
                    for name in self.vision_projection.state_dict().keys()
                ]
                vision_extra_state_param_names = []
                for name in self.vision_model.state_dict().keys():
                    if "_extra_state" in name:
                        vision_extra_state_param_names.append(f"vision_model.{name}")
                self.vision_projection.register_load_state_dict_post_hook(
                    partial(
                        _load_state_dict_hook_ignore_param_names,
                        vision_projection_param_names,
                    )
                )
                self.vision_model.register_load_state_dict_post_hook(
                    partial(
                        _load_state_dict_hook_ignore_param_names,
                        vision_extra_state_param_names,
                    )
                )
                llava_param_names = ["image_newline"]
                self.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, llava_param_names)
                )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for llava"

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def encode_images(self, images):
        # ViT model
        encoded_image_features = self.vision_model(images)
        # MLP model, [b, num_img_patches, hidden_size] -> [num_img_patches, b, hidden_size]
        encoded_image_features = encoded_image_features.permute(1, 0, 2).contiguous()
        encoded_image_features = self.vision_projection(encoded_image_features)
        # [num_img_patches, b, hidden_size] -> [b, num_img_patches, hidden_size]
        encoded_image_features = encoded_image_features.permute(1, 0, 2).contiguous()

        return encoded_image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels,
        images,
        modalities=["image"],
        image_sizes=None,
        past_key_values=None,
    ):
        """This function is modified from LLaVA-NeXT."""
        args = self.args
        vision_tower = self.vision_model

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # [BugFix]: comment out the embed_tokens
            # input_ids = self.embed_tokens(input_ids)
            loss_mask = torch.where(
                labels == IGNORE_INDEX, torch.tensor(0), torch.tensor(1)
            )
            return input_ids, position_ids, attention_mask, labels, loss_mask

        if isinstance(modalities, str):
            modalities = [modalities]

        # Comment out the code because we're starting to support it
        """
        text_modality = False
        image_or_video_modality = False
        for modality in modalities:
            if modality == "text":
                text_modality = True
            elif modality in ["image", "video"]:
                image_or_video_modality = True

        if text_modality and image_or_video_modality:
            raise ValueError(
                "Text and image/video modalities cannot be mixed in the same batch."
            )
        """

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))
                    # Comment out the code because we're starting to support it
                    # raise ValueError(
                    #     "Video not supported yet. In the future, we will support video."
                    # )

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]

            encoded_image_features = self.encode_images(concat_images)

            # Get every sample image features
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                    # Comment out the code because we're starting to support it
                    # raise ValueError(
                    #     "Video not supported yet. In the future, we will support video."
                    # )
                else:
                    image_features.append(image_feat)

            mm_patch_merge_type = args.mm_patch_merge_type
            # mm_patch_merge_type all values are supported

            image_aspect_ratio = args.image_aspect_ratio
            if image_aspect_ratio != "square":
                assert (
                    "anyres" in image_aspect_ratio
                ), f"Unexpected image_aspect_ratio: {image_aspect_ratio}"

            mm_newline_position = args.mm_newline_position
            assert mm_newline_position in [
                "one_token",
                "grid",
                "frame",
                "no_token",
            ], f"Unexpected mm_newline_position: {mm_newline_position}"

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_idx in video_idx_in_batch:  # video operations
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            assert not self.args.add_faster_video
                            # No all_faster_video_features variable because encode_multimodals is not called.
                            # So we didn't add any code about add_faster_video
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))

                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if "unpad" in mm_patch_merge_type:
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.image_newline[None].to(
                                            image_feature.device
                                        ),
                                    ),
                                    dim=0,
                                )
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(
                                f"Unexpected mm_newline_position: {mm_newline_position}"
                            )
                    elif image_feature.shape[0] > 1:
                        # Raw image features
                        base_image_feature = image_feature[0]
                        # Patch iamge features
                        image_feature = image_feature[1:]
                        assert (
                            args.img_h == args.img_w
                        ), "Only support square image size."
                        height = width = args.img_h // args.patch_dim
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(
                                r"anyres_max_(\d+)", image_aspect_ratio
                            )
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(
                                    matched_anyres_max_num_patches.group(1)
                                )

                        if (
                            image_aspect_ratio == "anyres"
                            or "anyres_max" in image_aspect_ratio
                        ):
                            vision_tower_image_size = args.img_h
                            assert (
                                args.image_grid_pinpoints is not None
                            ), "image_grid_pinpoints must be provided."
                            num_patch_width, num_patch_height = (
                                get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    args.image_grid_pinpoints,
                                    vision_tower_image_size,
                                )
                            )
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif (
                            "unpad" in mm_patch_merge_type
                            and "anyres_max" in image_aspect_ratio
                            and matched_anyres_max_num_patches
                        ):
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = torch.nn.functional.interpolate(
                                    image_feature,
                                    [int(h // times), int(w // times)],
                                    mode="bilinear",
                                )[0]
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat(
                                (base_image_feature, image_feature), dim=0
                            )
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature, self.image_newline[None]), dim=0
                            )
                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {args.mm_patch_merge_type}"
                )
        else:
            image_features = self.encode_images(images)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # Inserting images embedding
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # Text Modality
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = args.max_position_embeddings

        new_input_embeds = [
            x[:tokenizer_model_max_length]
            for x, modality in zip(new_input_embeds, modalities)
        ]
        new_labels = [
            x[:tokenizer_model_max_length]
            for x, modality in zip(new_labels, modalities)
        ]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        tokenizer = get_tokenizer()
        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if tokenizer.padding_side == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
            loss_mask = None
        else:
            new_labels = new_labels_padded
            loss_mask = torch.where(
                new_labels == IGNORE_INDEX, torch.tensor(0), torch.tensor(1)
            )

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if args.use_pos_skipping and args.training:
            position_ids = (
                torch.arange(new_input_embeds.size(1), device=new_input_embeds.device)
                .unsqueeze(0)
                .to(new_input_embeds.device)
            )
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, args.pos_skipping_range)
            right_add = random.randint(left_add, args.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return new_input_embeds, position_ids, attention_mask, new_labels, loss_mask

    # Currently only suitable for training
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
        image_sizes: Optional[List[torch.Tensor]] = None,
        modalities: Optional[List[str]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
    ) -> torch.Tensor:

        input_embeds, position_ids, attention_mask, labels, loss_mask = (
            self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                labels,
                images,
                modalities,
                image_sizes,
            )
        )
        # Attention mask should be None for the language model forward.
        attention_mask = None
        # Convert to [s, b, h]
        input_embeds = input_embeds.transpose(1, 0).contiguous()
        # Language model forward
        # labels should be None for the language model forward and return the logits
        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=input_embeds,
            labels=None,
        )
        return output, labels, loss_mask

    def embed_tokens(self, input_ids):
        language_embeddings = self.language_model.embedding(
            input_ids, position_ids=None
        )  # [text_seq_len, b, h_language]
        language_embeddings = language_embeddings.transpose(
            1, 0
        ).contiguous()  # [b, text_seq_len, h_language]
        return language_embeddings

    def get_2dPool(self, image_feature, stride=2):
        args = self.args
        height = width = args.img_h // args.patch_dim
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        if args.mm_spatial_pool_mode == "average":
            image_feature = torch.nn.functional.avg_pool2d(image_feature, stride)
        elif args.mm_spatial_pool_mode == "max":
            image_feature = torch.nn.functional.max_pool2d(image_feature, stride)
        elif args.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = torch.nn.functional.interpolate(
                image_feature, size=scaled_shape, mode="bilinear"
            )

        else:
            raise ValueError(
                f"Unexpected mm_spatial_pool_mode: {args.mm_spatial_pool_mode}"
            )
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat(
            (
                image_feature,
                self.image_newline[:, None, None]
                .expand(*image_feature.shape[:-1], 1)
                .to(image_feature.device),
            ),
            dim=-1,
        )
        if self.args.add_faster_video:
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames, resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            return image_feature
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature = torch.cat(
            (
                image_feature,
                self.image_newline[:, None, None]
                .expand(*image_feature.shape[:-1], 1)
                .to(image_feature.device),
            ),
            dim=-1,
        )
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this if you want to load a checkpoint that contains vision and language
    model weights but not the vision projection weights.

    Args:
        param_names (list str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in LlavaModel"
            )
            incompatible_keys.missing_keys.remove(param_name)


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Same as LLaVA-NeXT.
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)

    return width // patch_size, height // patch_size


def select_best_resolution(original_size, possible_resolutions):
    """
    Same as LLaVA-NeXT.
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )

        # Calculate effective and wasted resolutions
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def unpad_image(tensor, original_size):
    """
    Same as LLaVA-NeXT.
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

# Copyright (c) 2026, BAAI. All rights reserved.
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

"""Colocated Qwen3.5 MIMO model provider."""

from typing import Any

import torch
import torch.distributed as dist

from megatron.core import InferenceParams
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from flagscale.models.megatron.qwen3_vl.vision_model import Qwen3VisionModel
from flagscale.models.megatron.qwen35.language_model import Qwen35LanguageModule
from flagscale.models.megatron.qwen35.rope import get_rope_index
from flagscale.models.megatron.qwen35.transformer_config import Qwen35TransformerConfig
from flagscale.models.mimo.colocated import (
    ModuleParallelismConfig,
    build_colocated_pg_collections,
    validate_mimo_config,
)
from flagscale.models.mimo.colocated.model import ColocatedMIMOModel
from flagscale.models.mimo.colocated.parallel_state_ctx import switch_parallel_state
from flagscale.models.mimo.colocated.utils import compute_microbatch_token_counts


def build_qwen35_colocated_parallelism(
    args,
) -> tuple[ModuleParallelismConfig, ModuleParallelismConfig]:
    """Derive the ``(vision, language)`` module parallelisms from the global args.

    Vision TP/PP default to the language sizes; vision DP is always derived
    from the world size, never set manually (same as Megatron DP).
    """
    world_size = dist.get_world_size()
    vision_tp = (
        getattr(args, "vision_tensor_model_parallel_size", None) or args.tensor_model_parallel_size
    )
    vision_pp = (
        getattr(args, "vision_pipeline_model_parallel_size", None)
        or args.pipeline_model_parallel_size
    )
    vision_dp = world_size // vision_tp // vision_pp
    vision_parallelism = ModuleParallelismConfig(
        tensor_model_parallel_size=vision_tp,
        pipeline_model_parallel_size=vision_pp,
        data_parallel_size=vision_dp,
    )
    language_parallelism = ModuleParallelismConfig(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        data_parallel_size=(
            world_size // args.tensor_model_parallel_size // args.pipeline_model_parallel_size
        ),
        expert_model_parallel_size=getattr(args, "expert_model_parallel_size", 1),
    )
    return vision_parallelism, language_parallelism


def _cat_imgs_videos(batch: dict[str, Any]):
    """Concat image and video tensors along the sample dimension."""
    imgs = batch.get("imgs")
    videos = batch.get("videos")
    if imgs is None:
        return videos
    if videos is None:
        return imgs
    return torch.cat([imgs, videos], dim=0)


def _cat_grids(batch: dict[str, Any]):
    """Concat image and video T/H/W grids along the sample dimension."""
    image_grids = batch.get("image_thw_grids")
    video_grids = batch.get("video_thw_grids")
    if image_grids is None:
        return video_grids
    if video_grids is None:
        return image_grids
    return torch.cat([image_grids, video_grids], dim=0)


class Qwen35ColocatedMIMOModel(ColocatedMIMOModel):
    """Qwen3.5 MIMO model with colocated vision and language modules.

    Generic colocated orchestration (scheduler lifecycle, intra-TP slicing,
    output exchange, delayed ViT backward) lives in ``ColocatedMIMOModel``;
    this class keeps only the Qwen3.5 glue: module construction, the vision
    hook implementations, and the embedding-injection forward.
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
        pg_collections: dict[str, object],
        vision_parallelism,
        language_parallelism,
        vit_batch_factor: int,
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
        mtp_block_spec=None,
        use_fp32_grad_cache: bool = False,
    ) -> None:
        super().__init__(
            config=language_transformer_config,
            pg_collections=pg_collections,
            vit_batch_factor=vit_batch_factor,
            use_fp32_grad_cache=use_fp32_grad_cache,
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        # Build vision module under the vision parallel context.
        if self.pre_process and self.add_encoder:
            with switch_parallel_state(self.vision_pg):
                self.vision_model = Qwen3VisionModel(
                    vision_transformer_config,
                    vision_transformer_layer_spec,
                    vision_projection_config,
                    vision_projection_layer_spec,
                    projection_type=vision_projection_type,
                    pre_process=True,
                    post_process=True,
                )

        # Build language module under the language parallel context.
        with switch_parallel_state(self.language_pg):
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
                pg_collection=self.language_pg,
            )

        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

    # ------------------------------------------------------------------
    # ColocatedMIMOModel hooks (Qwen3.5 data layout).
    # ------------------------------------------------------------------
    def _count_vision_tokens(self, batches: list[dict[str, Any]]) -> list[int] | None:
        per_batch_grids = [_cat_grids(b) for b in batches]
        if not any(t is not None for t in per_batch_grids):
            return None
        # Account for Qwen3-VL spatial merge when counting tokens.
        merge_unit = getattr(self.vision_model, "spatial_merge_unit", 1)
        return compute_microbatch_token_counts(per_batch_grids, merge_unit=merge_unit)

    def _drop_vision_data(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Keep the grid metadata: _count_vision_tokens needs it for the full
        # macro batch; only the pixel tensors are safe to drop here.
        batch["imgs"] = None
        batch["videos"] = None
        return batch

    def _extract_vision_inputs(self, my_batches: list[dict[str, Any]]):
        imgs_videos = [t for t in (_cat_imgs_videos(b) for b in my_batches) if t is not None]
        grids = [t for t in (_cat_grids(b) for b in my_batches) if t is not None]
        if not grids:
            return None
        return torch.cat(imgs_videos, dim=0), torch.cat(grids, dim=0)

    def _run_vision(self, vision_inputs):
        vision_data, vision_grid_thw = vision_inputs
        if vision_grid_thw.shape[0] == 0:
            return None, None
        return self.vision_model(vision_data=vision_data, grid_thw=vision_grid_thw)

    def _num_aux_features(self) -> int:
        return len(getattr(self.vision_model.config, "deepstack_visual_indexes", None) or [])

    def _vision_projection_module(self):
        if self.vision_model is None:
            return None
        return self.vision_model.projection

    # ------------------------------------------------------------------
    # Megatron interface.
    # ------------------------------------------------------------------
    def shared_embedding_or_output_weight(self):
        """Surface the language model's word embeddings for gradient all-reduce."""
        if self.add_decoder:
            with switch_parallel_state(self.language_pg):
                return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        with switch_parallel_state(self.language_pg):
            self.language_model.set_input_tensor(input_tensor)

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
        extra_block_kwargs: dict | None = None,
        vision_output: dict | None = None,
    ) -> torch.Tensor:
        """Forward function of Qwen3.5 MIMO model."""
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            raise NotImplementedError()

        if self.pre_process and self.add_encoder:
            if vision_output is None:
                # Model-internal advance: a caller prepared the macro batch but
                # left advancing to the model (not used by forward_step, which
                # needs the batch up front and therefore advances externally).
                _, _, vision_output = self.scheduler.advance()

            # Capture gradients on the served vision outputs for the delayed
            # ViT backward.
            vision_embeds, deepstack_feature_lists = self._register_vision_output_hooks(
                vision_output
            )

            if inference_params is not None:
                raise NotImplementedError()

            with switch_parallel_state(self.language_pg):
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

        with switch_parallel_state(self.language_pg):
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


def build_qwen35_colocated_mimo_model(
    args,
    *,
    language_transformer_config,
    language_transformer_layer_spec,
    vision_transformer_config,
    vision_transformer_layer_spec,
    vision_projection_config,
    vision_projection_layer_spec,
    mtp_block_spec=None,
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
):
    """Build the colocated Qwen3.5 MIMO model (validation + assembly)."""
    # Keep package imports independent of the full megatron.training stack.
    from megatron.training.utils import print_rank_0

    vision_parallelism, language_parallelism = build_qwen35_colocated_parallelism(args)
    pg_collections = build_colocated_pg_collections(
        vision_parallelism, language_parallelism, dist.get_world_size()
    )
    pg_summary = ", ".join(
        f"{name}(tp={dist.get_world_size(pgs.tp)}, "
        f"dp={dist.get_world_size(pgs.dp)}, "
        f"pp={dist.get_world_size(pgs.pp)})"
        for name, pgs in pg_collections.items()
    )
    print_rank_0(f"MIMO process group collections: {pg_summary}")

    # Single-point validation of model-agnostic MIMO config constraints.
    vit_batch_factor = validate_mimo_config(
        args, vision_parallelism, language_parallelism, get_num_microbatches()
    )
    print_rank_0(
        f"MIMO vit_batch_factor={vit_batch_factor} "
        f"(vision_dp={vision_parallelism.data_parallel_size}, "
        f"language_dp={language_parallelism.data_parallel_size}, "
        f"num_microbatches={get_num_microbatches()})"
    )

    model = Qwen35ColocatedMIMOModel(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_transformer_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        pg_collections=pg_collections,
        vision_parallelism=vision_parallelism,
        language_parallelism=language_parallelism,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        mtp_block_spec=mtp_block_spec,
        vit_batch_factor=vit_batch_factor,
        use_fp32_grad_cache=getattr(args, "mimo_fp32_grad_cache", False),
    )

    # Attach the language pg_collection to the wrapper for compatibility with
    # code that expects a top-level pg_collection attribute.
    model.pg_collection = pg_collections["language"]
    return model

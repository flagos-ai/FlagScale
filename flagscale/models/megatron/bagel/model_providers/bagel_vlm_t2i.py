# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model provider for the Bagel Vision-Language Model.

This provider assembles the Bagel multimodal model that consists of:
• Qwen2.5-7B language backbone with MoT (Mixture-of-Transformers) layers.
• SigLIP ViT vision encoder for image understanding.
• MLP projector that maps vision embeddings into the LLM hidden space.
• VAE-based generation pathway for image generation (flow matching).
"""
import dataclasses

from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from flagscale.models.megatron.bagel.configs.bagel_vlm_t2i import (
    get_bagel_language_layer_spec,
    get_bagel_projection_config,
    get_bagel_projection_layer_spec,
    get_qwen_language_model_config,
    get_vision_model_config,
)
from flagscale.models.megatron.bagel.models import SiglipVisionModel, BagelModel


def model_provider_bagel_vlm_t2i(
    pre_process: bool = True,
    post_process: bool = True,
    pg_collection=None,
):
    """Build the Bagel multimodal model.

    Assembles a BagelModel from:
    • Qwen2.5-7B language model with MoT layers (understanding + generation branches).
    • SigLIP ViT-SO400M/14 vision encoder (26 layers, 1152 hidden).
    • 2-layer MLP vision projection (1152 -> 3584).
    • VAE latent processing for image generation (flow matching).
    """

    # --- Get configs from Megatron args ---
    try:
        from megatron.training import get_args
        args = get_args()
    except (ModuleNotFoundError, AssertionError):
        args = None

    # --- Language model config (Qwen2.5-7B) ---
    language_config = get_qwen_language_model_config()

    # --- Vision encoder config (SigLIP ViT) ---
    vision_config = get_vision_model_config()

    # --- Vision projection config ---
    projection_config = get_bagel_projection_config(
        hidden_size=language_config.hidden_size,
    )

    # Sync precision flags from global args.
    if args is not None:
        for f in dataclasses.fields(TransformerConfig):
            if hasattr(args, f.name):
                setattr(language_config, f.name, getattr(args, f.name))
        for cfg in [vision_config, projection_config]:
            if getattr(args, "bf16", False):
                cfg.bf16 = True
            if getattr(args, "fp16", False):
                cfg.fp16 = True

        # Sync parallelism settings.
        if hasattr(args, "context_parallel_size"):
            language_config.context_parallel_size = args.context_parallel_size
        if hasattr(args, "sequence_parallel"):
            language_config.sequence_parallel = args.sequence_parallel
        if hasattr(args, "tensor_model_parallel_size"):
            for cfg in [language_config, vision_config, projection_config]:
                cfg.tensor_model_parallel_size = args.tensor_model_parallel_size
        if hasattr(args, "pipeline_model_parallel_size"):
            for cfg in [language_config, vision_config, projection_config]:
                cfg.pipeline_model_parallel_size = args.pipeline_model_parallel_size

    # --- Bagel-specific parameters from args ---
    # Visual modality flags.
    visual_und = getattr(args, "visual_und", True) if args else True
    visual_gen = getattr(args, "visual_gen", True) if args else True

    # Generation parameters.
    latent_patch_size = getattr(args, "latent_patch_size", 2) if args else 2
    max_latent_size = getattr(args, "max_latent_size", 64) if args else 64
    vit_max_num_patch_per_side = getattr(args, "vit_max_num_patch_per_side", 70) if args else 70
    latent_channel = getattr(args, "latent_channel", 16) if args else 16
    vae_downsample = getattr(args, "vae_downsample", 16) if args else 16
    timestep_shift = getattr(args, "timestep_shift", 1.0) if args else 1.0

    # Vision encoder params.
    patch_dim = getattr(args, "patch_dim", 14) if args else 14
    image_size = getattr(args, "image_size", 980) if args else 980
    assert vit_max_num_patch_per_side == image_size // patch_dim

    # Vocab and sequence length.
    vocab_size = getattr(args, "padded_vocab_size", 152064) if args else 152064
    max_sequence_length = language_config.seq_length

    # rope
    position_embedding_type = getattr(args, "position_embedding_type", "rope") if args else "rope"
    rotary_percent = getattr(args, "rotary_percent", 1.0) if args else 1.0
    rotary_base = getattr(args, "rotary_base", 1000000) if args else 1000000
    rope_scaling = getattr(args, "use_rope_scaling", False) if args else False

    # --- Layer specs ---
    qk_layernorm = getattr(args, "qk_layernorm", True) if args else True
    language_layer_spec = get_bagel_language_layer_spec(qk_layernorm=qk_layernorm)
    projection_layer_spec = get_bagel_projection_layer_spec()
    connector_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": projection_layer_spec.submodules,
            "projector_type": "mlp",
            "input_size": vision_config.hidden_size,
        },
    )

    # --- Vision encoder layer spec (standard GPT TE spec for SigLIP) ---
    from megatron.core.models.vision.vit_layer_specs import (
        get_vit_layer_with_transformer_engine_spec,
    )
    vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
    vision_model_spec = ModuleSpec(
        module=SiglipVisionModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": vision_layer_spec,
            "patch_dim": patch_dim,
            "image_size": image_size,
        },
    )

    # --- Build BagelModel ---
    model = BagelModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_layer_spec,
        language_vocab_size=vocab_size,
        language_max_sequence_length=max_sequence_length,
        language_position_embedding_type=position_embedding_type,
        language_rotary_percent=rotary_percent,
        language_rotary_base=rotary_base,
        language_rope_scaling=rope_scaling,
        vision_model_spec=vision_model_spec,
        connector_spec=connector_spec,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        visual_gen=visual_gen,
        visual_und=visual_und,
        latent_patch_size=latent_patch_size,
        max_latent_size=max_latent_size,
        vit_max_num_patch_per_side=vit_max_num_patch_per_side,
        latent_channel=latent_channel,
        vae_downsample=vae_downsample,
        timestep_shift=timestep_shift,
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=pg_collection,
    )

    print(f"{model=}")

    # --- Freeze modules as specified in args ---
    if args is not None:
        freeze_llm = getattr(args, "freeze_LLM", False)
        freeze_vit = getattr(args, "freeze_ViT", False)
        freeze_vae = getattr(args, "freeze_VAE", False)
        freeze_text_embed = getattr(args, "freeze_text_embed", False)
        freeze_connect = getattr(args, "freeze_connect", False)
        freeze_und = getattr(args, "freeze_und", False)
        freeze_gen = getattr(args, "freeze_gen", False)

        model.freeze(
            freeze_language_model=freeze_llm,
            freeze_vision_model=freeze_vit,
            freeze_vae_model=freeze_vae,
            freeze_text_embed=freeze_text_embed,
            freeze_connect=freeze_connect,
            freeze_und=freeze_und,
            freeze_gen=freeze_gen,
        )

    # --- Load pretrained checkpoints if specified ---
    if args is not None:
        load_path = getattr(args, "load", None)
        if load_path:
            print(f"[Bagel] Model checkpoint will be loaded from: {load_path}")

    return model

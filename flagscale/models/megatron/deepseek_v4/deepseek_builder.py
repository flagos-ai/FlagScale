# Copyright (c) 2026 FlagScale CORPORATION & AFFILIATES. All rights reserved.

"""Alternative GPT builder file.

This file is intentionally shaped like deepseek_builders.py so you can switch by
changing only the import path in training scripts.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_mtp_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)

from megatron.core.transformer.enums import LayerType
from megatron.training.utils import get_args
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
    _get_backend_spec_provider,
    get_dsv4_hybrid_module_spec_for_backend,
    _get_moe_module_spec,
    get_moe_layer_pattern,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.engram import EngramModule

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import nvidia_kitchen  # pylint: disable=unused-import

    from megatron.core.extensions.kitchen import KitchenSpecProvider

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False

from .deepseek_transformer_layer import DeepSeekTransformerLayer, DeepSeekTransformerLayerSubmodules
from .deepseek_model import DeepSeekModel


def get_deepseek_layer_spec(
    use_te: bool,
    config: TransformerConfig,
    build_engram: bool = False,
) -> ModuleSpec:
    """
    Build LayerSpec that inserts engram and mhc into TransformerLayer.
    Because not all layers have engram, we build the engram module as an optional submodule.
    """
    backend = _get_backend_spec_provider(config=config)
    hybrid_attn_spec = get_dsv4_hybrid_module_spec_for_backend(config=config, backend=backend)

    moe_layer_spec = _get_moe_module_spec(config=config, backend=backend)
    rms_norm = config.normalization == "RMSNorm"
    input_layernorm = (
        IdentityOp
        if hybrid_attn_spec.metainfo["fuse_input_layernorm"]
        else backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    )
    pre_mlp_layernorm = (
        IdentityOp
        if moe_layer_spec.metainfo["fuse_pre_mlp_layernorm"]
        else backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    )
    if build_engram:
        engram_module = EngramModule
    else:
        engram_module = None
    submodules = DeepSeekTransformerLayerSubmodules(
        input_layernorm=input_layernorm,
        self_attention=hybrid_attn_spec,
        self_attn_bda=get_bias_dropout_add,
        self_attention_hyper_connection=HyperConnectionModule,
        pre_mlp_layernorm=pre_mlp_layernorm,
        mlp=moe_layer_spec,
        mlp_bda=get_bias_dropout_add,
        mlp_hyper_connection=HyperConnectionModule,
        engram=ModuleSpec(module=engram_module),
    )

    return ModuleSpec(module=DeepSeekTransformerLayer, submodules=submodules)


def get_deepseek_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: int | None = None,
    dualpipev_stage: Optional[int] = None,
    use_moe: bool | None = False,
):
    """Build decoder block spec and attach STM/HC placeholders to each local layer."""

    """GPT block spec."""
    layer_norm_impl = TENorm
    moe_deepseek_engram_layer_spec = get_deepseek_layer_spec(
        use_te=use_transformer_engine,
        config=config,
        build_engram=True,
    )
    moe_deepseek_layer_spec = get_deepseek_layer_spec(
        use_te=use_transformer_engine,
        config=config,
        build_engram=False,
    )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if config.use_engram and layer_number in config.engram_layer_ids:
            is_engram_layer = True
        else:
            is_engram_layer = False
        layer_specs.append(
            moe_deepseek_engram_layer_spec if is_engram_layer else moe_deepseek_layer_spec
        )

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    ######### FlagScale Modify ########
    num_layers_to_build = get_num_layers_to_build(
        config,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
        dualpipev_stage=dualpipev_stage,
    )

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
            )
        ]
    else:
        ######### FlagScale Modify ########
        offset = get_transformer_layer_offset(
            config,
            vp_stage=vp_stage,
            pp_rank=pp_rank,
            dualpipev_stage=dualpipev_stage,
        )
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs, layer_norm=layer_norm_impl
    )

    return block_spec


def deepseek_builder(
    args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
):
    """Drop-in replacement builder compatible with model_provider(...)."""
    print_rank_0("building DeepSeek model (engram and mhc file) ...")

    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        raise NotImplementedError("Legacy GPT models do not support deepseek module insertion.")
    else:
        if args.spec is not None:
            raise NotImplementedError("Using custom spec is not supported with deepseek builder.")
        else:
            use_te = args.transformer_impl == "transformer_engine"

            if args.heterogeneous_layers_config_path is not None:
                assert not (config.transformer_impl == "inference_optimized")
                raise NotImplementedError(
                    "Using heterogeneous layers is not supported with deepseek builder."
                )
            transformer_layer_spec = get_deepseek_decoder_block_spec(
                config=config,
                use_transformer_engine=use_te,
                normalization=args.normalization,
                qk_l2_norm=args.qk_l2_norm,
                vp_stage=vp_stage,
                use_moe=True,
            )

        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            assert not (config.transformer_impl == "inference_optimized")
            transformer_layer_spec_for_mtp = get_deepseek_layer_spec(
                use_te, config, build_engram=False
            )
            mtp_block_spec = get_gpt_mtp_block_spec(
                config,
                transformer_layer_spec_for_mtp,
                use_transformer_engine=use_te,
                vp_stage=vp_stage,
            )

        model = DeepSeekModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
    print(f"Model = {model}")
    return model

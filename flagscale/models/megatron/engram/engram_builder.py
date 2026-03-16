# ruff: noqa: TC001
# ruff: noqa: F401
## built-in
import logging

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import (
    BackendSpecProvider,
)

## megatron-core
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
    get_mlp_module_spec_for_backend,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

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

# engram
from .engram_model import EngramModel
from .engram_transformer_layer import EngramTransformerLayer

logger = logging.getLogger(__name__)


def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.

    Args:
        use_te (bool): Whether to use Transformer Engine
        args: Training arguments
        config: Model configuration

    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    args = get_args()
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    elif config.transformer_impl == "inference_optimized":
        return get_gpt_layer_with_inference_spec(
            args.qk_layernorm,
            args.multi_latent_attention,
            qk_l2_norm=args.qk_l2_norm,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=config.use_kitchen,
        )


def get_engram_transformer_layer_spec(
    num_experts: int | None = None,
    moe_grouped_gemm: bool | None = False,
    qk_layernorm: bool | None = False,
    multi_latent_attention: bool | None = False,
    fp8: str | None = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: bool | None = False,
    qk_l2_norm: bool | None = False,
    use_te_op_fuser: bool | None = False,
    use_kitchen: bool = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    if use_kitchen:
        assert HAVE_KITCHEN
        backend: BackendSpecProvider = KitchenSpecProvider(fallback=TESpecProvider())
        if use_te_op_fuser:
            raise AssertionError("use_te_op_fuser not compatible with using kitchen in mlp.")
        if use_te_activation_func:
            raise AssertionError("use_te_activation_func not compatible with using kitchen.")
    else:
        backend = TESpecProvider()

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
        use_te_activation_func=use_te_activation_func,
    )

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        linear_q_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        linear_kv_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        return ModuleSpec(
            module=EngramTransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=backend.layer_norm(),
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_q_down_proj=backend.linear(),
                        linear_q_up_proj=linear_q_up_proj,
                        linear_kv_down_proj=backend.linear(),
                        linear_kv_up_proj=linear_kv_up_proj,
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:
        qk_norm = backend.layer_norm(for_qk=True)
        return ModuleSpec(
            module=EngramTransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=backend.column_parallel_layer_norm_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                    "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                    "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                    "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                    "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                    "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
                },
            ),
        )


def get_engram_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: str | None = None,
    qk_l2_norm: bool | None = False,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
    is_dualpipev_first_chunk: bool | None = False,
    use_moe: bool | None = False,
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    layer_norm_impl = TENorm
    ## original transformer layer spec
    dense_orig_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
    )
    moe_orig_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
    )
    # engram transformer layer spec
    dense_engram_layer_spec = get_engram_transformer_layer_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
    )
    moe_engram_layer_spec = get_engram_transformer_layer_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if use_moe:
        if isinstance(config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
            ]
        elif isinstance(config.moe_layer_freq, list):
            moe_layer_pattern = config.moe_layer_freq
            assert len(moe_layer_pattern) == config.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {config.num_layers}, "
                f"current moe layer pattern: {config.moe_layer_freq}"
            )
        else:
            raise ValueError(
                f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
            )
    else:
        moe_layer_pattern = [0] * config.num_layers

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        is_engram_layer = True if layer_number in config.engram_layer_ids else False
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_engram_layer_spec if is_engram_layer else moe_orig_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(
                dense_engram_layer_spec if is_engram_layer else dense_orig_layer_spec
            )
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    ######### FlagScale Modify ########
    num_layers_to_build = get_num_layers_to_build(
        config,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
        is_dualpipev_first_chunk=is_dualpipev_first_chunk,
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
            is_dualpipev_first_chunk=is_dualpipev_first_chunk,
        )
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs, layer_norm=layer_norm_impl
    )

    return block_spec


def engram_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0("building Engram model ...")

    config = core_transformer_config_from_args(args)

    assert not args.use_legacy_models, "Engram only supported in Mcore!"
    assert args.spec is None, "Engram only supported in Mcore!"
    use_te = args.transformer_impl == "transformer_engine"
    assert use_te, "Engram only supported in Transformer Engine!"
    use_moe = True if args.num_experts else False

    # Define the decoder block spec
    if args.use_engram:
        decoder_func = get_engram_decoder_block_spec
    else:
        decoder_func = get_gpt_decoder_block_spec

    decoder_kwargs = {
        "config": config,
        "use_transformer_engine": use_te,
        "normalization": args.normalization,
        "qk_l2_norm": args.qk_l2_norm,
        "vp_stage": vp_stage,
    }

    if args.use_engram:
        decoder_kwargs["use_moe"] = use_moe

    transformer_layer_spec = decoder_func(
        **decoder_kwargs,
    )

    # do not support engram for mtp now
    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        assert not (config.transformer_impl == "inference_optimized")
        transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
        mtp_block_spec = get_gpt_mtp_block_spec(
            config,
            transformer_layer_spec_for_mtp,
            use_transformer_engine=use_te,
            vp_stage=vp_stage,
        )

    if args.use_engram:
        model_class = EngramModel
    else:
        model_class = GPTModel

    print(f"init model, args.padded_vocab_size: {args.padded_vocab_size}")
    model = model_class(
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
    print(f"Engram model built successfully, {model=}")

    return model

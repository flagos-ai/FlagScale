# Copyright (c) 2025, BAAI. All rights reserved.
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


from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from flagscale.models.megatron.qwen35.attention import Qwen35SelfAttention


def _patch_standard_attention_specs(
    block_spec: TransformerBlockSubmodules,
    attention_cls,
) -> None:
    """Selectively replace self_attention module on standard attention layer specs.

    Copied from megatron.bridge.models.qwen_vl.qwen35_vl_provider._patch_standard_attention_specs

    In a hybrid block spec, each layer spec has a different self_attention submodule:
    - Standard attention layers have a ``SelfAttention``-like module.
    - GDN layers have a ``GatedDeltaNet``-like module.

    This function patches only the standard attention layers with *attention_cls*
    (e.g., Qwen35SelfAttention for mRoPE support), leaving GDN layers unchanged.

    Detection heuristic: GDN layer specs use ``GatedDeltaNet`` (or similar) as the
    self_attention module and do NOT have a ``linear_qkv`` submodule. Standard
    attention specs DO have ``linear_qkv``. We use this to distinguish them.
    """
    for layer_spec in block_spec.layer_specs:
        attn_spec = layer_spec.submodules.self_attention
        # Standard attention specs use SelfAttention (or a subclass) as the module
        # and have linear_qkv in their submodules. GDN specs use GatedDeltaNet.
        if attn_spec.module is SelfAttention or (
            isinstance(attn_spec.module, type) and issubclass(attn_spec.module, SelfAttention)
        ):
            attn_spec.module = attention_cls


def get_qwen35_language_model_spec(config, patch=True, vp_stage=None) -> TransformerBlockSubmodules:
    """Build hybrid GDN + Attention block spec for Qwen3.5 language model.

    Args:
        config: Qwen35TransformerConfig with:
            - experimental_attention_variant: "gated_delta_net"
            - linear_attention_freq: 4 (1 attention per 4 layers)
            - num_layers: 64 (for 27B dense)
        vp_stage: Virtual pipeline stage index. Must be forwarded when VPP is on,
            otherwise get_transformer_layer_offset() asserts (vp_stage must be
            provided if virtual_pipeline_model_parallel_size is set).

    Returns:
        TransformerBlockSubmodules with per-layer specs where:
        - GDN layers use GatedDeltaNet (from Megatron-LM-main)
        - Standard attention layers use Qwen35SelfAttention (mRoPE + output gate)
    """
    # Build hybrid block spec: produces TransformerBlockSubmodules with
    # per-layer specs (GDN layers get GatedDeltaNet, attention layers get
    # standard SelfAttention + standard MLP)
    block_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config,
        vp_stage=vp_stage,
    )

    # This flag only for mtp layer (patch = false).
    if patch:
        # Selectively patch only the standard (full) attention layer specs
        # with Qwen35SelfAttention for mRoPE support. GDN layers are left as-is.
        _patch_standard_attention_specs(block_spec, Qwen35SelfAttention)

    return block_spec


def get_mlp_module_spec(
    use_te: bool = True,
    num_experts: int | None = None,
    moe_grouped_gemm: bool = False,
    add_norm: bool = True,
):
    """Get MLP or MoE module spec for vision/language model."""
    assert use_te, "Only Transformer Engine backend is supported"
    if num_experts is None:
        if add_norm:
            return ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            )
        else:
            return ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            )
    else:
        assert moe_grouped_gemm, "Only TE Group GEMM is supported."
        return ModuleSpec(
            module=MoELayer,
            submodules=MoESubmodules(
                experts=ModuleSpec(
                    module=TEGroupedMLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TEColumnParallelGroupedLinear,
                        linear_fc2=TERowParallelGroupedLinear,
                    ),
                ),
            ),
        )


def get_qwen35_mtp_block_spec(args, config, vp_stage=None):
    mtp_block_spec = None
    if getattr(args, "mtp_num_layers", None) is not None:
        # MTP uses standard SelfAttention (not Qwen35SelfAttention or GDN).
        # Generate an unpatched block spec so MTP gets vanilla SelfAttention.
        # NOTE(wqq) Maybe we can make this code clear but it match Megatron-Bridge behavior.
        # NOTE(vpp): vp_stage must be forwarded here too. With VPP on, both the
        # unpatched decoder spec build and get_gpt_mtp_block_spec compute layer
        # offsets that assert on vp_stage. MTP layers are only built on the last
        # pipeline stage's last vp chunk (get_mtp_num_layers_to_build gates this),
        # so on other stages this returns None before touching layer_specs.
        unpatched_spec = get_qwen35_language_model_spec(config, patch=False, vp_stage=vp_stage)
        mtp_block_spec = get_gpt_mtp_block_spec(
            config,
            unpatched_spec,
            use_transformer_engine=args.use_te,
            vp_stage=vp_stage,
        )
    return mtp_block_spec

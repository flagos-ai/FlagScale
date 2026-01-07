import math
import re

from importlib.metadata import version
from typing import Any, Callable, List, Optional, Tuple, Type

import packaging
import torch

from torch import nn

from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    TECL = (TEColumnParallelLinear, TELayerNormColumnParallelLinear, TEColumnParallelGroupedLinear)
    TERL = (TERowParallelLinear, TERowParallelGroupedLinear)
    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def match_module(m, name, prefix, target_modules):
    """ """
    full_name = f"{prefix}.{name}" if prefix else name
    for pattern in target_modules:
        if name == pattern or wildcard_match(pattern, full_name):
            return (pattern, full_name)
    return None


def wildcard_match(pattern, key):
    """
    Return whether the pattern (target module to add LoRA) matches the key (model weight name).

    Example:
    --------
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.0.self_attention.linear_qkv")
        True
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.1.self_attention.linear_qkv")
        False
    """
    if key is None:
        return False
    regex_pattern = re.compile("^" + pattern.replace("*", "(.*)") + "$")
    match = regex_pattern.match(key)
    return match is not None


def get_adapter_attributes_from_linear(m: nn.Module):
    """
    Return input_is_parallel, in_features, out_feature attributes based on implementation of the base layer.
    """
    base_linear_is_parallel = True
    if HAVE_TE and any(isinstance(m, te_column_parallel) for te_column_parallel in TECL):
        input_is_parallel = False
        # m.in_features and m.out_features are divided by tp_size already,
        # but in_features and out_features passed to ParallelLinearAdapter are not.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        in_features = m.in_features
        out_features = m.out_features * tp_size
        if isinstance(m, TELayerNormColumnParallelLinear):
            # LoRA is applied after layernorm, so layernorm output must be returned
            m.return_layernorm_output = True
    elif HAVE_TE and any(isinstance(m, te_row_parallel) for te_row_parallel in TERL):
        input_is_parallel = True
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        in_features = m.in_features * tp_size
        out_features = m.out_features
    elif HAVE_TE and isinstance(m, TELinear):  # parallel_mode="duplicated"
        input_is_parallel = False
        in_features = m.in_features
        out_features = m.out_features
        base_linear_is_parallel = False
    elif isinstance(m, ColumnParallelLinear):
        input_is_parallel = False
        in_features = m.input_size
        out_features = m.output_size
    elif isinstance(m, RowParallelLinear):
        input_is_parallel = True
        in_features = m.input_size
        out_features = m.output_size
    else:
        raise NotImplementedError(f"Layer type is unrecognized for LoRA: {type(m)}")

    return (input_is_parallel, in_features, out_features, base_linear_is_parallel)


def is_expert_linear(fqn):
    """
    Return whether the current base module is an expert linear module.
    See ParallelLinearAdapter.is_expert for usage details.
    """
    return re.match(r'(?!.*shared_).*mlp\..*experts.*\.linear_fc[1-2]$', fqn) is not None


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def init_method_kaiming_uniform(val):
    """ """

    def init_(tensor):
        return nn.init.kaiming_uniform_(tensor, a=val)

    return init_


def init_method_const(val):
    """ """

    def init_(tensor):
        return nn.init.constant_(tensor, val)

    return init_


class ParallelLinearAdapter(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        model_parallel_config: ModelParallelConfig,
        gather_output: bool,
        input_is_parallel: bool,
        is_expert: bool,
        in_init_method: str = "xavier",
        out_init_method: str = "zero",
        dropout: float = 0.0,
        alpha: float | None = None,
        dropout_position: str = 'post',
    ):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        self.alpha = alpha if alpha is not None else self.dim
        self.dropout_position = dropout_position
        self.is_expert = is_expert
        self.input_is_parallel = input_is_parallel

        if input_is_parallel:
            self.linear_in = TERowParallelLinear(
                input_size=in_features,
                output_size=dim,
                config=model_parallel_config,
                init_method=self._get_init_fn(in_init_method),
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=is_expert,
            )
        else:
            self.linear_in = TELinear(
                input_size=in_features,
                output_size=dim,
                parallel_mode="duplicated",
                config=model_parallel_config,
                init_method=self._get_init_fn(in_init_method),
                bias=False,
                skip_bias_add=True,
                is_expert=is_expert,
                symmetric_ar_type=model_parallel_config.symmetric_ar_type,
                skip_weight_param_allocation=False,
            )

        if input_is_parallel:
            self.linear_out = TELinear(
                input_size=dim,
                output_size=out_features,
                parallel_mode="duplicated",
                config=model_parallel_config,
                init_method=self._get_init_fn(out_init_method),
                bias=False,
                skip_bias_add=True,
                is_expert=is_expert,
                symmetric_ar_type=model_parallel_config.symmetric_ar_type,
                skip_weight_param_allocation=False,
            )
        else:
            self.linear_out = TEColumnParallelLinear(
                input_size=dim,
                output_size=out_features,
                config=model_parallel_config,
                init_method=self._get_init_fn(out_init_method),
                gather_output=gather_output,
                bias=False,
                skip_bias_add=True,
                is_expert=is_expert,
                skip_weight_param_allocation=False,
            )

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def _get_init_fn(self, init_method: str):
        if init_method == 'xavier':
            init_fn = nn.init.xavier_normal_
        elif init_method == 'normal':
            init_fn = init_method_normal(0.2)
        elif init_method == 'kaiming':
            init_fn = init_method_kaiming_uniform(math.sqrt(5))
        elif init_method == "zero":
            init_fn = init_method_const(0.0)
        else:
            raise NotImplementedError("init_method should be zero, normal, kaiming or xavier")
        return init_fn

    def forward(self, x):
        """ """
        if self.dropout is not None and self.dropout_position == 'pre':
            x = self.dropout(x)

        x, _ = self.linear_in(x)
        x, _ = self.linear_out(x)

        if self.dropout is not None and self.dropout_position == 'post':
            x = self.dropout(x)

        x = x * (self.alpha / self.dim)

        return x

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ):
        """
        Sharded state dict for LoRA adapter. Special treatment is given to the linear_fc1 adapter
        since TP is sharded separately for the two logical matrices (gate and up)
        """
        sharded_state_dict = {}
        linear_in_sd = self.linear_in.sharded_state_dict(
            f"{prefix}linear_in.", sharded_offsets, metadata
        )
        linear_out_sd = self.linear_out.sharded_state_dict(
            f"{prefix}linear_out.", sharded_offsets, metadata
        )

        sharded_state_dict.update(linear_in_sd)
        sharded_state_dict.update(linear_out_sd)
        return sharded_state_dict

    def __repr__(self):
        return (
            f"{type(self).__name__}(linear_in: in_features={self.linear_in.in_features}, "
            f"out_features={self.linear_in.out_features}, bias={self.linear_in.use_bias}, TP={self.linear_in.tp_size}), "
            f"(linear_out: in_features={self.linear_out.in_features}, "
            f"out_features={self.linear_out.out_features}, bias={self.linear_out.use_bias}, TP={self.linear_out.tp_size})"
        )

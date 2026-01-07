import logging

from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from megatron.training.utils import unwrap_model

from megatron.training.peft.peft import PEFT, AdapterWrapper
from megatron.training.peft.utils import (
    ParallelLinearAdapter,
    get_adapter_attributes_from_linear,
    is_expert_linear,
    match_module,
)

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEGroupedLinear,
        TERowParallelGroupedLinear,
    )

    TEGROUP = (TEColumnParallelGroupedLinear, TERowParallelGroupedLinear, TEGroupedLinear)
    HAVE_GROUP = True
except ImportError:
    HAVE_GROUP = False


class LoRALinear(AdapterWrapper):
    """An adapter wrapper that adds the output of the adapter to the output of the wrapped module.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        adapter_output = self.adapter(layernorm_output.contiguous())
        adapter_output = adapter_output.reshape(linear_output.shape)
        return linear_output + adapter_output, bias


class LoRA(PEFT, peft_type='lora'):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target_modules = config.lora_target_modules
        self.dim = config.lora_dim
        self.alpha = config.lora_alpha
        self.dropout = config.lora_dropout
        self.dropout_position = config.lora_dropout_position
        self.in_init_method = config.lora_in_init_method
        self.out_init_method = config.lora_out_init_method

    def transform(self, m: nn.Module, name=None, prefix=None):
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        if (ans := match_module(m, name, prefix, self.target_modules)) is not None:
            (match, full_name) = ans

            (input_is_parallel, in_features, out_features, base_linear_is_parallel) = (
                get_adapter_attributes_from_linear(m)
            )

            adapter = ParallelLinearAdapter(
                in_features=in_features,
                out_features=out_features,
                dim=self.dim,
                model_parallel_config=getattr(m, "config", None),
                gather_output=False,
                input_is_parallel=input_is_parallel,
                is_expert=is_expert_linear(full_name),
                in_init_method=self.in_init_method,
                out_init_method=self.out_init_method,
                dropout=self.dropout,
                alpha=self.alpha,
                dropout_position=self.dropout_position,
            )

            lora_linear = LoRALinear(m, adapter)
            return lora_linear

        return m

    def freeze_model(self, model: nn.Module):
        """Freeze main model for lora"""
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True

    def load_state_dict_pre_hooks(self, model: nn.Module):
        def load_state_dict_hook_remap_main_model_params(
            module: torch.nn.Module,
            state_dict: Dict[str, Any],
            prefix: str,
            local_metadata: Optional[dict],
            strict: bool,
            missing_keys: List[str],
            unexpected_keys: List[str],
            errors: List[Any],
        ):
            if HAVE_GROUP and any(isinstance(module.to_wrap, te_group) for te_group in TEGROUP):
                old_keys = []
                new_keys = []
                for gemm_id in range(module.to_wrap.num_gemms):
                    old_keys.append(prefix + f"weight{gemm_id}")
                    new_keys.append(prefix + f"to_wrap.weight{gemm_id}")
                    old_keys.append(prefix + f"bias{gemm_id}")
                    new_keys.append(prefix + f"to_wrap.bias{gemm_id}")
            else:
                old_keys = [prefix + "weight", prefix + "bias"]
                new_keys = [prefix + "to_wrap.weight", prefix + "to_wrap.bias"]

            for old_key, new_key in zip(old_keys, new_keys):
                if old_key in state_dict.keys():
                    if new_key not in state_dict.keys():
                        state_dict[new_key] = state_dict.pop(old_key)
                    else:
                        state_dict.pop(old_key)

        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.register_load_state_dict_pre_hook(
                    load_state_dict_hook_remap_main_model_params
                )

    def load_state_dict_post_hooks(self, model: nn.Module):
        def load_state_dict_hook_ignore_param_names(
            param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
        ):
            for param_name in param_names:
                if param_name in incompatible_keys.missing_keys:
                    logging.getLogger(__name__).warning(
                        f"{param_name} being removed from incompatible_keys.missing_keys in this model"
                    )
                    incompatible_keys.missing_keys.remove(param_name)

        lora_param_names = []
        for name in model.state_dict().keys():
            if "adapter" in name:
                lora_param_names.append(name)
        model.register_load_state_dict_post_hook(
            partial(load_state_dict_hook_ignore_param_names, lora_param_names)
        )

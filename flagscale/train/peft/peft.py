from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn


class PEFT(ABC):
    """Abstract base class for Parameter-Efficient Fine-Tuning (PEFT) methods.

    This class defines the interface for PEFT methods, which are used to fine-tune
    large language models efficiently by modifying only a small subset of the model's
    parameters.
    """

    _REGISTRY: Dict[str, Type["PEFT"]] = {}

    def __init_subclass__(cls, *, peft_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        if peft_type in cls._REGISTRY:
            raise ValueError(f"peft_type={peft_type} registered already!")
        cls._REGISTRY[peft_type] = cls
        cls._peft_type = peft_type

    @classmethod
    def from_config(cls, config) -> "PEFT":
        peft_type = config.peft_type
        if peft_type not in cls._REGISTRY:
            raise KeyError(f"Unsupported peft_type: {peft_type}, registered: {list(cls._REGISTRY)}")
        sub_cls = cls._REGISTRY[peft_type]
        return sub_cls(config)

    @abstractmethod
    def transform(self, module, name=None, prefix=None):
        """Transform a single module according to the PEFT method.

        This method is called for each module in the model during the PEFT application process.
        It should be implemented by subclasses to define how individual modules are transformed
        for the specific PEFT technique.

        Args:
            module (nn.Module): The individual module to be transformed.
            name (Optional[str]): The name of the module within the model structure. Defaults to None.
            prefix (Optional[str]): A prefix to be added to the module name, typically used for
                                    nested modules. Defaults to None.

        Returns:
            nn.Module: The transformed module. This can be the original module with modifications,
                       a new module replacing the original, or the original module if no
                       transformation is needed for this specific module.

        Note:
            This method is automatically called for each module in the model when the PEFT
            instance is applied to the model using the __call__ method.
        """
        raise NotImplementedError("The transform method should be implemented by subclasses.")

    def apply_transform(self, model: nn.Module):
        for full_name, module in model.named_modules():
            prefix, name = full_name.rsplit('.', 1) if '.' in full_name else ('', full_name)
            replaced_module = self.transform(module, name, prefix)
            if replaced_module == module:
                continue
            model.set_submodule(full_name, replaced_module)

    def freeze_model(self, model: nn.Module):
        """Apply a default freeze method to the model."""
        pass

    def load_state_dict_pre_hooks(self, model: nn.Module):
        pass

    def load_state_dict_post_hooks(self, model: nn.Module):
        pass


class AdapterWrapper(nn.Module):
    """Abstract base class for wrapping modules with adapters in Parameter-Efficient Fine-Tuning (PEFT).

    This class wraps a module and its associated adapter, providing methods for
    managing the state dictionaries of both the main module and the adapter. It does not
    implement the forward method, which must be implemented by concrete subclasses.

    Attributes:
        to_wrap (nn.Module): The main module to be wrapped.
        adapter (nn.Module): The adapter module to be applied.

    """

    def __init__(self, to_wrap: nn.Module, adapter: nn.Module):
        super(AdapterWrapper, self).__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter

    def base_linear_forward(self, x, *args, **kwargs):
        """
        Run the forward method of the linear module `to_wrap`.
        Return a tuple of three elements: linear_output, bias, layernorm_output

        x -> [layernorm/identity] -> layernorm_output -> [linear] -> linear_output, bias

        layernorm_output is different from input x only when linear layer is LayerNormColumnParallelLinear.
        """
        linear_output = self.to_wrap(x, *args, **kwargs)
        assert isinstance(
            linear_output, tuple
        ), f"{self.to_wrap} should return a tuple but instead returns {linear_output}"
        """ Four cases for the wrapped module's return values
        1. nothing: (out, None)
        2. return_bias: (out, bias)
        2. return_layernorm_output: ((out, ln_out), None)
        3. both: (out, bias, ln_out)
        """
        bias = None
        layernorm_output = x
        if len(linear_output) == 2:
            linear_output, bias = linear_output
            if isinstance(linear_output, tuple) and len(linear_output) == 2:
                linear_output, layernorm_output = linear_output
        elif len(linear_output) == 3:
            linear_output, bias, layernorm_output = linear_output

        return linear_output, bias, layernorm_output

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Retrieve the state dictionary of the wrapped module and adapter.

        This method overrides the default state_dict behavior to include both
        the main module's state and the adapter's state under a special 'adapters' key.

        Args:
            destination (Optional[dict]): A dictionary to store the state. If None, a new
                                          dictionary is created. Defaults to None.
            prefix (str): A prefix added to parameter and buffer names. Defaults to ''.
            keep_vars (bool): If True, returns variables instead of tensor values.
                              Defaults to False.

        Returns:
            dict: The state dictionary containing both the main module and adapter states.
        """

        if destination is None:
            destination = {}

        # Get state dict of the main module
        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Store adapter state dict under the "adapter" prefix in the destination dict
        self.adapter.state_dict(
            destination=destination, prefix=f'{prefix}adapter.', keep_vars=keep_vars
        )
        return destination

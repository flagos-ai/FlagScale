import inspect

import torch
import torch.nn as nn

from flagscale.inference.runtime_context import current_ctx
from flagscale.transformations.hook import ModelHook, ModuleHookRegistry, unwrap_module
from flagscale.transformations.transformation import Transformation


# TODO(yupu): Check if this hook is general and reliable
class TimestepTrackerHook(ModelHook):
    """Hook that captures the `timestep` argument before module forward.

    This updates the process-local RuntimeContext with the current diffusion
    timestep value and a monotonically increasing step index whenever the
    timestep changes.
    """

    def __init__(self):
        """Initialize the hook."""
        super().__init__()
        self._cached_parameter_indices: dict[str, int] | None = None

    # Modified from:
    # https://github.com/huggingface/diffusers/blob/310fdaf5561d1b20240a2b66e978edb66175ad5c/src/diffusers/hooks/_helpers.py#L33
    def _get_parameter_from_args_kwargs(
        self, module: nn.Module, identifier: str, args=(), kwargs=None
    ):
        """Fetch a named parameter from args/kwargs given a module forward.

        Mirrors diffusers' helper to support both positional and keyword
        invocation styles.

        Args:
          module: The module whose forward signature is inspected.
          identifier: The parameter name to fetch (e.g., "timestep").
          args: Positional args passed to forward.
          kwargs: Keyword args passed to forward.

        Returns:
          The value of the requested parameter.
        """
        kwargs = kwargs or {}
        if identifier in kwargs:
            return kwargs[identifier]
        if self._cached_parameter_indices is not None:
            return args[self._cached_parameter_indices[identifier]]
        parameters = list(inspect.signature(module.__class__.forward).parameters.keys())
        parameters = parameters[1:]  # skip `self`
        self._cached_parameter_indices = {param: i for i, param in enumerate(parameters)}
        if identifier not in self._cached_parameter_indices:
            raise ValueError(
                f"Parameter '{identifier}' not found in function signature but was requested."
            )
        index = self._cached_parameter_indices[identifier]
        if index >= len(args):
            raise ValueError(f"Expected {index} arguments but got {len(args)}.")
        return args[index]

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        """Update RuntimeContext with the current timestep before forward."""
        ts: torch.Tensor | float | int | None = self._get_parameter_from_args_kwargs(
            unwrap_module(module), "timestep", args, kwargs
        )
        ctx = current_ctx()
        if ctx:
            ctx.update_timestep(ts)
        else:
            raise ValueError("No context found")

        return args, kwargs


class TimestepTrackerTransformation(Transformation):
    """Attach a `TimestepTrackerHook` to the provided model/module."""

    def __init__(self):
        """Initialize the transformation."""
        super().__init__()

    def apply(self, model: nn.Module) -> bool:
        """Register the tracking hook on the root module.

        Args:
          model: Root module whose forward (and its submodules') calls carry a
            `timestep` argument we want to observe.

        Returns:
          True if the hook is successfully attached.
        """
        reg = ModuleHookRegistry.get_or_create_registry(model)
        hook = TimestepTrackerHook()
        reg.register_hook(hook, "timestep_tracker")
        return True

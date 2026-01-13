from typing import Any

import torch
from torch import nn

from flagscale.runner.utils import logger
from flagscale.transformations.hook import ModelHook, ModuleHookRegistry
from flagscale.transformations.transformation import Transformation


def _shape_of(x: torch.Tensor, name: str = "N/A") -> str:
    return f"{name}: {getattr(x, 'shape', type(x).__name__)}"


class LogIOHook(ModelHook):
    """A simple hook that logs the input/output shapes of a module.
    Only used for debugging. Will be removed in the future.
    """

    def __init__(self, log_level: str = "info") -> None:
        """Initialize the hook.

        Args:
            log_level (str): The log level to use. It must be a valid logger level.
        """

        super().__init__()
        if not hasattr(logger, log_level) or not callable(getattr(logger, log_level)):
            raise ValueError(f"Invalid log level: {log_level}")
        self._logger_func = getattr(logger, log_level)

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple[Any], dict[str, Any]]:
        """Log the input shapes of the module."""

        shapes = tuple(_shape_of(a) for a in args)
        shapes += tuple(_shape_of(v, k) for k, v in kwargs.items())

        self._logger_func(f"[LogIOHook] {module.__class__.__name__} input shapes: {shapes}")
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Log the output shape of the module."""

        if isinstance(output, (list, tuple)):
            shapes = tuple(_shape_of(o) for o in output)
        else:
            shapes = _shape_of(output)

        self._logger_func(f"[LogIOHook] {module.__class__.__name__} output shape: {shapes}")
        return output


class LogIOTransformation(Transformation):
    """A transform that logs the input shapes of a module. Just to showcase the transform API."""

    def __init__(self, log_level: str = "info") -> None:
        """Initialize the transform.

        Args:
            log_level (str): The log level to use. It must be a valid logger level.
        """

        super().__init__()

        self._log_level = log_level

    def apply(self, model: nn.Module) -> bool:
        reg = ModuleHookRegistry.get_or_create_registry(model)
        hook = LogIOHook(log_level=self._log_level)
        reg.register_hook(hook, "log_io")
        return True

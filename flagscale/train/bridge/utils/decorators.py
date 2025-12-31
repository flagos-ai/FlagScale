# Copyright (c) 2025, BAAI. All rights reserved.
#
# Copied from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

import functools
import logging
import warnings

from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Define a TypeVar for generic return types
R = TypeVar("R")


def experimental_fn(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to mark a function as experimental and issue a warning upon its call."""
    warning_message = f"Function '{func.__name__}' is experimental. APIs in this module are subject to change without notice."

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        warnings.warn(warning_message, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper

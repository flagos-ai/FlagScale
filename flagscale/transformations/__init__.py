from typing import Dict, List, Type

from omegaconf import DictConfig

from .diffusion.taylorseer_transformation import TaylorSeerTransformation
from .diffusion.timestep_embedding_flip_sine_cosine_pass import TimestepEmbeddingFlipSineCosinePass
from .diffusion.timestep_tracker_transformation import TimestepTrackerTransformation
from .log_io_transformation import LogIOTransformation
from .state_scope_transformation import StateScopeTransformation
from .torch_compile_transformation import TorchCompileTransformation
from .transformation import Transformation
from flagscale.compilation.inductor_pass import InductorPass

# Registry of supported Transformation classes by their class names.
_TRANSFORMATION_REGISTRY: Dict[str, Type[Transformation]] = {
    "LogIOTransformation": LogIOTransformation,
    "StateScopeTransformation": StateScopeTransformation,
    "TimestepTrackerTransformation": TimestepTrackerTransformation,
    "TaylorSeerTransformation": TaylorSeerTransformation,
    "TorchCompileTransformation": TorchCompileTransformation,
}

_PASS_REGISTRY: Dict[str, Type[InductorPass]] = {
    "TimestepEmbeddingFlipSineCosinePass": TimestepEmbeddingFlipSineCosinePass
}

__all__ = ["create_transformations_from_config", "create_passes_from_config"]


def create_transformations_from_config(cfg: DictConfig) -> List[Transformation]:
    """Instantiate transformations from the configuration

    Args:
        cfg: The configuration

    Returns:
        A list of instantiated transformations
    """

    instances: List[Transformation] = []

    for name, kwargs in cfg.items():
        cls = _TRANSFORMATION_REGISTRY.get(name)
        if cls is None:
            raise KeyError(
                f"Unknown transformation class '{name}'. Available: {sorted(_TRANSFORMATION_REGISTRY.keys())}"
            )
        try:
            if kwargs is None:
                kwargs = {}
            inst = cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate transformation '{name}' with kwargs {kwargs}: {e}"
            ) from e
        instances.append(inst)

    return instances


def create_passes_from_config(cfg: DictConfig) -> List[InductorPass]:
    """Instantiate passes from the configuration

    Args:
        cfg: The configuration

    Returns:
        A list of instantiated passes
    """
    instances: List[InductorPass] = []

    for name, kwargs in cfg.items():
        cls = _PASS_REGISTRY.get(name)
        if cls is None:
            raise KeyError(
                f"Unknown pass class '{name}'. Available: {sorted(_PASS_REGISTRY.keys())}"
            )
        try:
            if kwargs is None:
                kwargs = {}
            inst = cls(**kwargs)
        except TypeError as e:
            raise TypeError(f"Failed to instantiate pass '{name}' with kwargs {kwargs}: {e}") from e
        instances.append(inst)

    return instances

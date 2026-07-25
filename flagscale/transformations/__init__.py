# Copyright 2026 FlagOS Contributors
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

from omegaconf import DictConfig

from .log_io_transformation import LogIOTransformation
from .state_scope_transformation import StateScopeTransformation
from .transformation import Transformation

# Registry of supported Transformation classes by their class names.
# Using lazy imports to avoid circular dependencies
_TRANSFORMATION_REGISTRY: dict[str, type[Transformation]] = {
    "LogIOTransformation": LogIOTransformation,
    "StateScopeTransformation": StateScopeTransformation,
}

__all__ = ["create_transformations_from_config"]


def _get_transformation_registry() -> dict[str, type[Transformation]]:
    """Get the transformation registry with lazy imports to avoid circular dependencies."""
    if "TimestepTrackerTransformation" not in _TRANSFORMATION_REGISTRY:
        from flagscale.inference.core.diffusion.timestep_tracker_transformation import (
            TimestepTrackerTransformation,
        )

        _TRANSFORMATION_REGISTRY["TimestepTrackerTransformation"] = TimestepTrackerTransformation

    if "TaylorSeerTransformation" not in _TRANSFORMATION_REGISTRY:
        from flagscale.inference.core.diffusion.taylorseer_transformation import (
            TaylorSeerTransformation,
        )

        _TRANSFORMATION_REGISTRY["TaylorSeerTransformation"] = TaylorSeerTransformation

    return _TRANSFORMATION_REGISTRY


def create_transformations_from_config(cfg: DictConfig) -> list[Transformation]:
    """Instantiate transformations from the configuration

    Args:
        cfg: The configuration

    Returns:
        A list of instantiated transformations
    """

    instances: list[Transformation] = []
    registry = _get_transformation_registry()

    for name, kwargs in cfg.items():
        cls = registry.get(name)
        if cls is None:
            raise KeyError(
                f"Unknown transformation class '{name}'. Available: {sorted(registry.keys())}"
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

# Adapted from https://github.com/huggingface/lerobot/blob/4303b3c9/src/lerobot/configs/policies.py
# and https://github.com/huggingface/lerobot/blob/4303b3c9/src/lerobot/policies/pretrained.py
#
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import dataclasses
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

if TYPE_CHECKING:
    import builtins

from flagscale.models.configs.types import FeatureType, PolicyFeature
from flagscale.models.utils.constants import ACTION, OBS_STATE, resolve_pretrained_dir

if TYPE_CHECKING:
    from flagscale.train.train_config import TrainConfig

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)

CONFIG_NAME = "config.json"


class _ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, FeatureType):
            return o.value
        if isinstance(o, PolicyFeature):
            return {"type": o.type.value, "shape": list(o.shape)}
        return super().default(o)


def _decode_features(raw: dict[str, Any] | None) -> dict[str, PolicyFeature] | None:
    if raw is None:
        return None
    return {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        for k, v in raw.items()
    }


@dataclass
class PreTrainedConfig(ABC):
    """Base configuration class for policy models.

    Each concrete policy (PI0, QwenGr00t, ...) subclasses this to declare its
    architecture-specific fields while inheriting a shared interface for
    features, serialization, and delta-timestamp resolution.

    Attributes:
        input_features: Maps observation keys to their ``PolicyFeature``
            (type + shape).  Set at training time from dataset metadata;
            persisted in ``config.json`` so inference can reconstruct the
            model without the dataset.
        output_features: Same structure, for action outputs.

    Subclasses must implement:
        validate_features  -- check that required features are present.
        observation_delta_indices / action_delta_indices -- frame offsets
            used by ``resolve_delta_timestamps`` in the training script.

    Serialization:
        ``_save_pretrained(dir)`` writes all fields to ``config.json``.
        ``from_pretrained(dir)`` reads them back.  Subclasses with nested
        dataclass fields should override ``_from_dict`` to reconstruct
        the nested objects.

    Registration:
        Concrete (non-abstract) subclasses are auto-registered by class
        name with the ``Config`` suffix stripped::

            @dataclass
            class QwenGr00tConfig(PreTrainedConfig): ...


            # registered as "QwenGr00t"

        ``PreTrainedConfig.from_pretrained(path)`` reads the ``"type"`` key
        from ``config.json`` and dispatches to the registered subclass.
    """

    _registry: ClassVar[dict[str, builtins.type[PreTrainedConfig]]] = {}

    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        type_name = cls.__name__.removesuffix("Config")
        if type_name in PreTrainedConfig._registry:
            existing = PreTrainedConfig._registry[type_name]
            if existing is not cls:
                raise ValueError(
                    f"type name '{type_name}' is already registered by {existing.__name__}"
                )
        cls._type_name = type_name
        PreTrainedConfig._registry[type_name] = cls

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        """Return the STATE feature keyed by ``OBS_STATE``, or None."""
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """Return all VISUAL input features."""
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        """Return the ACTION output feature keyed by ``ACTION``, or None."""
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    @abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @classmethod
    def from_train_config(cls, train_config: TrainConfig):
        """Build a config from the OmegaConf-based TrainConfig.

        When called on the base class, dispatches to the registered subclass
        based on ``train_config.model.model_name``.  Concrete subclasses must
        override this with the actual construction logic.

        Features (``input_features`` / ``output_features``) are NOT set here;
        they are populated later from dataset metadata (matching LeRobot's
        ``make_policy`` pattern).
        """
        if cls is PreTrainedConfig:
            model_name = getattr(train_config.model, "model_name", None)
            if model_name is None:
                raise ValueError(
                    "train_config.model.model_name is required for polymorphic dispatch"
                )
            normalized = model_name.replace("_", "").lower()
            for type_name, config_cls in PreTrainedConfig._registry.items():
                if type_name.lower() == normalized:
                    return config_cls.from_train_config(train_config)
            raise ValueError(
                f"No config registered for model_name '{model_name}'. "
                f"Known types: {list(PreTrainedConfig._registry.keys())}"
            )
        raise NotImplementedError(f"{cls.__name__} must implement from_train_config")

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write all fields to ``config.json`` inside *save_directory*."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        data = {"type": self._type_name}
        data.update(dataclasses.asdict(self))
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(data, f, indent=4, cls=_ConfigEncoder)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        **kwargs,
    ) -> T:
        """Load a config from a directory containing ``config.json``.

        When called on the base class (``PreTrainedConfig.from_pretrained``),
        the ``"type"`` key in the JSON is used to dispatch to the registered
        subclass.  When called on a concrete subclass, the ``"type"`` key is
        ignored and the caller's class is used directly.
        """
        path = resolve_pretrained_dir(Path(pretrained_name_or_path), CONFIG_NAME)
        config_file = path / CONFIG_NAME
        if not config_file.exists():
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {path.resolve()}")

        with open(config_file) as f:
            data = json.load(f)

        if cls is PreTrainedConfig:
            type_name = data.pop("type", None)
            if type_name is None:
                raise ValueError(
                    f"config.json at {path.resolve()} is missing the 'type' field "
                    f"required for polymorphic dispatch. "
                    f"Known types: {list(PreTrainedConfig._registry.keys())}"
                )
            config_cls = PreTrainedConfig._registry.get(type_name)
            if config_cls is None:
                # Fall back to case-insensitive match for lerobot's checkpoints that stored
                # the type in lowercase (e.g. "pi05" instead of "PI05").
                type_name_lower = type_name.lower()
                config_cls = next(
                    (
                        v
                        for k, v in PreTrainedConfig._registry.items()
                        if k.lower() == type_name_lower
                    ),
                    None,
                )
            if config_cls is None:
                raise ValueError(
                    f"Unknown config type '{type_name}'. "
                    f"Known types: {list(PreTrainedConfig._registry.keys())}"
                )
        else:
            data.pop("type", None)
            config_cls = cls

        data["input_features"] = _decode_features(data.get("input_features"))
        data["output_features"] = _decode_features(data.get("output_features"))

        # Strip HubMixin fields that lerobot's checkpoints may have persisted.
        for key in ("repo_id", "push_to_hub", "private", "tags", "license"):
            data.pop(key, None)

        return config_cls._from_dict(data)

    @classmethod
    def _from_dict(cls: builtins.type[T], data: dict[str, Any]) -> T:
        """Reconstruct a config from a plain dict.

        Subclasses with nested dataclass fields should override this to
        handle nested reconstruction (see ``QwenGr00tConfig._from_dict``).
        """
        return cls(**data)

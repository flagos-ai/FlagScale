from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from safetensors.torch import load_model, save_file
from torch import Tensor, nn

from flagscale.models.configs.types import FeatureType, PolicyFeature
from flagscale.models.utils.constants import SAFETENSORS_FILE, resolve_pretrained_dir
from flagscale.models.vla.pretrained_config import PreTrainedConfig

if TYPE_CHECKING:
    import builtins

logger = getLogger(__name__)


class TrainablePolicy(nn.Module, ABC):
    """Base class for all trainable VLA policies.

    Provides generic ``save_pretrained`` / ``from_pretrained`` that
    persist ``config.json + model.safetensors``.  Override
    ``_save_pretrained`` to add model-specific artifacts (e.g. VLM
    processor files) and ``from_pretrained`` to resolve model-specific
    config fixups before weight loading.

    Subclasses must implement:
        forward(batch) -> dict with at least {"loss": Tensor}
        predict_action(batch) -> dict with at least {"action": Tensor}

    Registration:
        Concrete (non-abstract) subclasses are auto-registered by class
        name, matching the type name used by ``PreTrainedConfig``::

            class QwenGr00t(TrainablePolicy):  # registered as "QwenGr00t"
                ...

        When ``TrainablePolicy.from_pretrained(path)`` is called on the
        base class, the config's ``_type_name`` is used to dispatch to
        the registered policy subclass.

    Optional overrides:
        image_features — derived from input_features by filtering FeatureType.VISUAL
    """

    _registry: ClassVar[dict[str, builtins.type[TrainablePolicy]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        # Pi0 and Pi0.5 has "Policy" suffix
        type_name = cls.__name__.removesuffix("Policy")
        if type_name in TrainablePolicy._registry:
            existing = TrainablePolicy._registry[type_name]
            if existing is not cls:
                raise ValueError(
                    f"policy type '{type_name}' is already registered by {existing.__name__}"
                )
        TrainablePolicy._registry[type_name] = cls

    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.config = config
        self._input_features: dict[str, PolicyFeature] = {}
        self._output_features: dict[str, PolicyFeature] = {}

    @classmethod
    def from_config(cls, config: PreTrainedConfig) -> TrainablePolicy:
        """Create a fresh policy from a config, dispatching by config type.

        Uses ``config._type_name`` (e.g. ``"QwenGr00t"``) to look up the
        registered policy subclass and calls its constructor.

        Args:
            config: A ``PreTrainedConfig`` subclass instance.
        """
        type_name = config._type_name
        policy_cls = cls._registry.get(type_name)
        if policy_cls is None:
            raise ValueError(
                f"No policy registered for config type '{type_name}'. "
                f"Known policies: {list(cls._registry.keys())}"
            )
        return policy_cls(config=config)

    def save_pretrained(self, save_directory, *, state_dict=None) -> None:
        """Save policy to *save_directory* (config.json + model.safetensors).

        Accepts an optional *state_dict*, which is required under FSDP2
        where ``model.state_dict()`` returns sharded DTensors.  The
        caller (e.g. ``save_checkpoint``) gathers the full state via
        ``get_model_state_dict()`` and passes it here.

        Args:
            save_directory: Target directory.
            state_dict: Pre-gathered full state dict.  When ``None``,
                falls back to ``self.state_dict()``.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self._save_pretrained(save_directory, state_dict=state_dict)

    def _save_pretrained(self, save_directory: Path, state_dict=None) -> None:
        """Write ``config.json`` and ``model.safetensors`` to *save_directory*.

        Subclasses may override to write additional files (e.g. VLM
        processor artifacts), but must still save config and weights.
        """
        self.config._save_pretrained(save_directory)
        if state_dict is not None:
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        else:
            state_dict = {k: v.clone().contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, str(Path(save_directory) / SAFETENSORS_FILE))

    @classmethod
    def from_pretrained(cls, pretrained_path, device="cpu", *, config=None):
        """Load a saved policy from *pretrained_path*.

        When called on the base class (``TrainablePolicy.from_pretrained``),
        reads ``config.json`` to determine the policy type and dispatches
        to the registered subclass.  When called on a concrete subclass,
        uses that class directly.

        Subclasses may override to resolve config fixups (e.g. relative
        VLM paths) before delegating back here via ``super()``.

        Args:
            pretrained_path: Directory containing saved checkpoint.
            device: Device to load weights onto.
            config: Optional pre-built config; skips reading config.json.
        """
        path = resolve_pretrained_dir(Path(pretrained_path), SAFETENSORS_FILE)
        if config is None:
            config = PreTrainedConfig.from_pretrained(path)

        if cls is TrainablePolicy:
            type_name = config._type_name
            policy_cls = TrainablePolicy._registry.get(type_name)
            if policy_cls is None:
                raise ValueError(
                    f"No policy registered for config type '{type_name}'. "
                    f"Known policies: {list(TrainablePolicy._registry.keys())}"
                )
            return policy_cls.from_pretrained(pretrained_path, device=device, config=config)

        model = cls(config=config)

        weights_path = path / SAFETENSORS_FILE
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        missing, unexpected = load_model(
            model,
            str(weights_path),
            device="cpu" if str(device) == "musa" else device,
            strict=False,
        )
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {len(missing)} keys")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)} keys")

        model.to(device)
        model.eval()
        return model

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]: ...

    @abstractmethod
    def predict_action(self, batch: dict[str, Tensor]) -> dict[str, Tensor]: ...

    def fsdp_units(self) -> list[nn.Module]:
        """Return the list of sub-modules to shard individually under FSDP2.

        Each returned module is wrapped with ``fully_shard`` before the
        top-level policy itself is wrapped.  Override in subclasses to
        provide model-specific granularity (e.g. transformer blocks).

        Defaults to an empty list (only the top-level policy is sharded).
        """
        return []

    @property
    def input_features(self) -> dict[str, PolicyFeature]:
        return self._input_features

    @input_features.setter
    def input_features(self, value: dict[str, PolicyFeature]):
        self._input_features = value

    @property
    def output_features(self) -> dict[str, PolicyFeature]:
        return self._output_features

    @output_features.setter
    def output_features(self, value: dict[str, PolicyFeature]):
        self._output_features = value

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self._input_features:
            return {}
        return {
            key: ft for key, ft in self._input_features.items() if ft.type is FeatureType.VISUAL
        }

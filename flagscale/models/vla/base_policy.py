from abc import ABC, abstractmethod
from pathlib import Path

from torch import Tensor, nn

from flagscale.models.configs.types import FeatureType, PolicyFeature


class TrainablePolicy(nn.Module, ABC):
    """Base class for all trainable VLA policies.

    Subclasses must implement:
        forward(batch) -> dict with at least {"loss": Tensor}
        predict_action(batch) -> dict with at least {"action": Tensor}

    Optional overrides:
        image_features — derived from input_features by filtering FeatureType.VISUAL
        checkpoint_config_overrides() — config patches for checkpoint portability
        save_pretrained_artifacts(save_dir) — persist non-weight artifacts (e.g., VLM config)
    """

    def __init__(self):
        super().__init__()
        self._input_features: dict[str, PolicyFeature] = {}
        self._output_features: dict[str, PolicyFeature] = {}

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]: ...

    @abstractmethod
    def predict_action(self, batch: dict[str, Tensor]) -> dict[str, Tensor]: ...

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

    def checkpoint_config_overrides(self) -> dict:
        """Return config patches applied to train_config.yaml at checkpoint save time.

        Override for models that need to adjust config for checkpoint loading
        (e.g., switching from pretrained VLM download to local config-only init).
        """
        return {}

    def save_pretrained_artifacts(self, save_dir: Path) -> None:
        """Save non-weight artifacts needed to reconstruct architecture from checkpoint.

        Override for models with heavy pretrained components (e.g., VLM config/processor).
        Called before save_checkpoint so artifacts exist when the config references them.
        """
        pass

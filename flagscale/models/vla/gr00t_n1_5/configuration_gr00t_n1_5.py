from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from flagscale.models.configs.types import FeatureType, NormalizationMode, PolicyFeature
from flagscale.models.utils.constants import ACTION, OBS_STATE
from flagscale.models.vla.pretrained_config import PreTrainedConfig

if TYPE_CHECKING:
    from flagscale.train.train_config import TrainConfig


@dataclass
class Gr00tN15Config(PreTrainedConfig):
    """Configuration for the GR00T N1.5 policy."""

    # Path to the base GR00T N1.5 pretrained model (local or HuggingFace hub ID)
    base_model_path: str = "nvidia/GR00T-N1.5-3B"

    # Fine-tuning control flags (passed to GR00TN15.from_pretrained)
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # Compute dtype for torch.autocast
    compute_dtype: str = "bfloat16"

    # Embodiment tag used by the GR00T model
    embodiment_tag: str = "new_embodiment"

    # Padding dimensions — shorter state/action sequences are zero-padded to these sizes
    max_state_dim: int = 64
    max_action_dim: int = 32

    # Number of future action steps predicted per forward pass
    chunk_size: int = 50

    # LoRA parameters (lora_rank=0 disables LoRA)
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_full_model: bool = False

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    def validate_features(self) -> None:
        if not self.input_features:
            raise ValueError("input_features must be set before calling validate_features")

        image_features = [k for k, v in self.input_features.items() if v.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError("Gr00tN15 requires at least one VISUAL feature in input_features.")

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
        else:
            state_dim = self.input_features[OBS_STATE].shape[0]
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}."
                )

        if not self.output_features:
            raise ValueError("output_features must be set before calling validate_features")
        action_dim = self.output_features[ACTION].shape[0]
        if action_dim > self.max_action_dim:
            raise ValueError(
                f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}."
            )

    @classmethod
    def from_train_config(cls, train_config: TrainConfig) -> Gr00tN15Config:
        model_cfg = train_config.model
        kwargs: dict[str, Any] = dict(
            base_model_path=model_cfg.checkpoint_dir,
            tune_llm=model_cfg.get("tune_llm", False),
            tune_visual=model_cfg.get("tune_visual", False),
            tune_projector=model_cfg.get("tune_projector", True),
            tune_diffusion_model=model_cfg.get("tune_diffusion_model", True),
            compute_dtype=model_cfg.get("compute_dtype", "bfloat16"),
            embodiment_tag=model_cfg.get("embodiment_tag", "new_embodiment"),
            max_state_dim=model_cfg.get("max_state_dim", 64),
            max_action_dim=model_cfg.get("max_action_dim", 32),
            chunk_size=model_cfg.get("chunk_size", 50),
            lora_rank=model_cfg.get("lora_rank", 0),
            lora_alpha=model_cfg.get("lora_alpha", 16),
            lora_dropout=model_cfg.get("lora_dropout", 0.1),
            lora_full_model=model_cfg.get("lora_full_model", False),
        )
        raw_norm = model_cfg.get("normalization_mapping", None)
        if raw_norm is not None:
            kwargs["normalization_mapping"] = {k: NormalizationMode(v) for k, v in raw_norm.items()}
        return cls(**kwargs)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Gr00tN15Config:
        if "normalization_mapping" in data and isinstance(data["normalization_mapping"], dict):
            data["normalization_mapping"] = {
                k: NormalizationMode(v) for k, v in data["normalization_mapping"].items()
            }
        return cls(**data)

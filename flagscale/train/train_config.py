"""
Training configuration models using Pydantic.
"""

from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator


class FreezeConfig(BaseModel):
    """Pattern-based module freezing configuration.

    Freezing logic:
    1. For each parameter, check if name matches any `freeze_patterns`
    2. If matched, check if name also matches any `keep_patterns`
    3. If matched by freeze but NOT by keep → freeze (requires_grad=False)

    `keep_patterns` overrides `freeze_patterns` - this allows freezing a module
    but keeping specific sub-components trainable.

    Patterns are regex patterns matched against full parameter names.
    """

    model_config = {"extra": "allow"}

    freeze_patterns: list[str] | None = None
    keep_patterns: list[str] | None = None


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration.

    Uses transformers scheduler types when `name` is set. See transformers.SchedulerType for options:
    linear, cosine, cosine_with_restarts, polynomial, constant,
    constant_with_warmup, inverse_sqrt, cosine_with_min_lr, etc.

    Example:
        scheduler:
          name: cosine
          warmup_steps: 1000
          scheduler_kwargs:
            min_lr: 1e-6

    For backward compatibility with pi0/pi0.5, the legacy fields (decay_steps, decay_lr) are kept.
    """

    name: str | None = None
    warmup_steps: int = 1000
    scheduler_kwargs: dict[str, Any] | None = None

    # Legacy fields for pi0/pi0.5 backward compatibility
    decay_steps: int = 30000
    decay_lr: float = 2.5e-6


class OptimizerConfig(BaseModel):
    """Optimizer configuration.

    Attributes:
        name: Optimizer class name. Currently supported: "AdamW".
        lr: Learning rate (default for all param groups).
        betas: Adam beta coefficients (beta1, beta2).
        eps: Adam epsilon for numerical stability.
        weight_decay: Weight decay (L2 penalty).
        param_groups: Per-module optimizer overrides. Maps module paths to optimizer kwargs.
            Example: {"encoder": {"lr": 1e-5}, "decoder": {"lr": 1e-3}}
        scheduler: LR scheduler config.

    Example config (YAML):
        optimizer:
          name: AdamW
          lr: 1e-4
          weight_decay: 0.01
          param_groups:
            vision_encoder:
              lr: 1e-5
            action_head:
              lr: 2e-4
          scheduler:
            name: cosine
            warmup_steps: 1000
    """

    name: str = "AdamW"
    lr: float | None = None
    betas: tuple[float, float] | None = None
    eps: float | None = None
    weight_decay: float | None = None
    param_groups: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Per-module optimizer settings. Maps module paths to optimizer kwargs.",
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @field_validator("betas", mode="before")
    @classmethod
    def normalize_betas(cls, v):
        """Convert list to tuple for betas if provided.

        Accepts both list and tuple inputs, but always stores as tuple.
        Also validates that betas has exactly two elements.
        """
        if v is None:
            return None
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError(f"betas must have exactly 2 elements, got {len(v)}")
            return tuple(v)
        if isinstance(v, tuple) and len(v) != 2:
            raise ValueError(f"betas must have exactly 2 elements, got {len(v)}")
        return v

    def get_optimizer_kwargs(self) -> dict[str, Any]:
        """Get non-None optimizer kwargs for passing to optimizer.

        Returns:
            Dict of optimizer kwargs, excluding None values.
        """
        return {
            k: v
            for k, v in {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }.items()
            if v is not None
        }


class CheckpointConfig(BaseModel):
    """Checkpoint saving configuration"""

    save_checkpoint: bool = True
    save_freq: int = 1000
    output_directory: str


class SystemConfig(BaseModel):
    """Training loop configuration"""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    batch_size: int = 1
    train_steps: int = 100000
    log_freq: int = 10
    grad_clip_norm: float = 1.0
    use_amp: bool = False
    shuffle: bool = False
    num_workers: int = 4

    checkpoint: CheckpointConfig
    raw: DictConfig | None = Field(default=None, exclude=True)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raw = self.__dict__.get("raw")
        if raw is not None and hasattr(raw, name):
            return getattr(raw, name)
        raise AttributeError(name)


class DataConfig(BaseModel):
    """Dataset configuration"""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    dataset_type: str = "lerobot"
    data_path: str = Field(..., description="Path to training dataset")
    tolerance_s: float = 0.0001
    use_imagenet_stats: bool = True
    rename_map: dict[str, str] | None = None
    use_quantiles: bool = False
    raw: DictConfig | None = Field(default=None, exclude=True)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raw = self.__dict__.get("raw")
        if raw is not None and hasattr(raw, name):
            return getattr(raw, name)
        raise AttributeError(name)


class ModelConfig(BaseModel):
    """Model configuration.

    This accepts any model-specific fields dynamically, allowing any other model config directly from YAML.

    Required fields:
    - model_name: Which model to use ('pi0' or 'pi0.5')
    - checkpoint_dir: Path to pretrained checkpoint (for loading weights)

    All other fields are passed through to the model's config class.
    """

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }

    # Required fields to identify which model and checkpoint to use
    model_name: str = Field(..., description="Model name: 'pi0' or 'pi0.5'")
    checkpoint_dir: str = Field(..., description="Path to pretrained model checkpoint")
    freeze: FreezeConfig | None = None
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    raw: DictConfig | None = Field(default=None, exclude=True)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raw = self.__dict__.get("raw")
        if raw is not None and hasattr(raw, name):
            return getattr(raw, name)
        raise AttributeError(name)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        valid_names = {"pi0", "pi0.5", "qwen_gr00t"}
        if v not in valid_names:
            raise ValueError(f"Invalid model_name: {v}. Must be one of {valid_names}")
        return v

    def get_model_config_dict(self) -> dict[str, Any]:
        """Get all model-specific config fields (excluding train-level fields)."""
        return self.model_dump(
            exclude={"model_name", "checkpoint_dir", "freeze", "optimizer"}
        )


class TrainConfig(BaseModel):
    """Top-level training configuration for native backend"""

    system: SystemConfig
    model: ModelConfig
    data: DataConfig

    @classmethod
    def from_hydra_config(cls, hydra_config) -> "TrainConfig":
        """Convert Hydra DictConfig to Pydantic TrainConfig"""
        train = hydra_config.train
        train_dict = OmegaConf.to_container(train, resolve=True)
        train_dict["system"] = SystemConfig(**train_dict["system"], raw=train.system)
        train_dict["data"] = DataConfig(**train_dict["data"], raw=train.data)
        train_dict["model"] = ModelConfig(**train_dict["model"], raw=train.model)
        return cls(**train_dict)

    def to_omegaconf(self) -> DictConfig:
        """Reconstruct the full OmegaConf config from stored raw DictConfigs."""
        return OmegaConf.create({
            "system": self.system.raw,
            "model": self.model.raw,
            "data": self.data.raw,
        })

    class Config:
        # Allow arbitrary types for complex objects
        arbitrary_types_allowed = True

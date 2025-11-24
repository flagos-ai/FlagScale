# Adopted from huggingface/lerobot (https://github.com/huggingface/lerobot/tree/main)

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

ACTION = "action"
OBS_STATE = "observation.state"


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


class DictLike(Protocol):
    def __getitem__(self, key: Any) -> Any: ...


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple

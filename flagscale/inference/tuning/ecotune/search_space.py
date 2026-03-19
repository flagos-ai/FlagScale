from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class SearchDimension:
    name: str
    low: float
    high: float
    kind: str = "float"

    def sample(self, rng: np.random.Generator) -> Any:
        value = rng.uniform(self.low, self.high)
        if self.kind == "int":
            return int(round(value))
        return float(value)

    def denormalize(self, value: float) -> Any:
        raw = self.low + float(value) * (self.high - self.low)
        if self.kind == "int":
            return int(round(raw))
        return float(raw)

    def normalize(self, value: Any) -> float:
        return (float(value) - self.low) / (self.high - self.low + 1e-12)


class SearchSpace:
    def __init__(self, dimensions: Sequence[SearchDimension]):
        if not dimensions:
            raise ValueError("SearchSpace requires at least one dimension")
        self._dimensions = list(dimensions)

    @property
    def dimensions(self) -> List[SearchDimension]:
        return list(self._dimensions)

    @property
    def n_dims(self) -> int:
        return len(self._dimensions)

    def sample(self, rng: np.random.Generator, n: int = 1) -> List[Dict[str, Any]]:
        configs: List[Dict[str, Any]] = []
        for _ in range(n):
            configs.append({dim.name: dim.sample(rng) for dim in self._dimensions})
        return configs

    def to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [dim.normalize(config[dim.name]) for dim in self._dimensions],
            dtype=np.float64,
        )

    def from_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        vec = np.asarray(vector, dtype=np.float64)
        return {
            dim.name: dim.denormalize(np.clip(vec[idx], 0.0, 1.0))
            for idx, dim in enumerate(self._dimensions)
        }

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GPConfig:
    config_length_scale: float = 0.25
    fidelity_length_scale: float = 0.2
    signal_variance: float = 1.0
    noise_variance: float = 1e-3


class MultiFidelityGPSurrogate:
    def __init__(self, n_dims: int, config: Optional[GPConfig] = None):
        self.n_dims = int(n_dims)
        self.cfg = config or GPConfig()
        self._config_x: List[np.ndarray] = []
        self._fidelity_x: List[float] = []
        self._y: List[float] = []
        self._is_fit = False

    def add_observation(self, config_vec: np.ndarray, fidelity: float, score: float) -> None:
        x = np.asarray(config_vec, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(
                f"config_vec must be a 1-D array of length {self.n_dims}, got shape {x.shape!r}"
            )
        if x.shape[0] != self.n_dims:
            raise ValueError(
                f"config_vec length {x.shape[0]} does not match expected n_dims={self.n_dims}"
            )
        self._config_x.append(x)
        self._fidelity_x.append(float(fidelity))
        self._y.append(float(score))
        self._is_fit = False

    def _kernel(self, x1: np.ndarray, r1: float, x2: np.ndarray, r2: float) -> float:
        dx = (x1 - x2) / (self.cfg.config_length_scale + 1e-8)
        dr = (r1 - r2) / (self.cfg.fidelity_length_scale + 1e-8)
        sq_dist = float(np.dot(dx, dx) + dr * dr)
        return float(self.cfg.signal_variance * np.exp(-0.5 * sq_dist))

    def _ensure_fit(self) -> None:
        if self._is_fit or not self._y:
            return
        n = len(self._y)
        self._K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                self._K[i, j] = self._kernel(
                    self._config_x[i],
                    self._fidelity_x[i],
                    self._config_x[j],
                    self._fidelity_x[j],
                )
        self._K += (self.cfg.noise_variance + 1e-8) * np.eye(n)
        self._L = np.linalg.cholesky(self._K)
        self._y_arr = np.asarray(self._y, dtype=np.float64)
        self._is_fit = True

    def predict(self, config_vec: np.ndarray, fidelity: float) -> Tuple[float, float]:
        self._ensure_fit()
        if not self._y:
            return 0.0, self.cfg.signal_variance

        x = np.asarray(config_vec, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(
                f"config_vec must be a 1-D array of length {self.n_dims}, got shape {x.shape!r}"
            )
        if x.shape[0] != self.n_dims:
            raise ValueError(
                f"config_vec length {x.shape[0]} does not match expected n_dims={self.n_dims}"
            )
        n = len(self._y)
        k_star = np.zeros(n, dtype=np.float64)
        for i in range(n):
            k_star[i] = self._kernel(x, fidelity, self._config_x[i], self._fidelity_x[i])

        tmp = np.linalg.solve(self._L, self._y_arr)
        alpha = np.linalg.solve(self._L.T, tmp)
        mu = float(k_star @ alpha)
        tmp2 = np.linalg.solve(self._L, k_star)
        v = np.linalg.solve(self._L.T, tmp2)
        var = float(self.cfg.signal_variance - k_star @ v)
        return mu, max(var, 1e-12)

    def best_score(self, min_fidelity: Optional[float] = None) -> float:
        if not self._y:
            return 0.0
        if min_fidelity is None:
            return float(np.max(np.asarray(self._y, dtype=np.float64)))

        filtered = [
            score for score, fidelity in zip(self._y, self._fidelity_x) if fidelity >= min_fidelity
        ]
        if not filtered:
            return float(np.max(np.asarray(self._y, dtype=np.float64)))
        return float(np.max(np.asarray(filtered, dtype=np.float64)))

    @property
    def num_observations(self) -> int:
        return len(self._y)

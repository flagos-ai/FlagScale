from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm

from .surrogate import MultiFidelityGPSurrogate


class TokenAwareExpectedImprovement:
    def __init__(
        self,
        surrogate: MultiFidelityGPSurrogate,
        token_cost_fn: Callable[[dict, float], float],
        incumbent_fidelity: float,
        xi: float = 0.01,
    ):
        self.surrogate = surrogate
        self.token_cost_fn = token_cost_fn
        self.incumbent_fidelity = float(incumbent_fidelity)
        self.xi = float(xi)

    def _expected_improvement(self, mu: float, var: float, incumbent: float) -> float:
        sigma = float(np.sqrt(max(var, 1e-12)))
        improvement = mu - incumbent - self.xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return max(float(ei), 0.0)

    def score(self, config: dict, config_vec: np.ndarray, fidelity: float) -> float:
        incumbent = self.surrogate.best_score(min_fidelity=self.incumbent_fidelity)
        mu, var = self.surrogate.predict(config_vec, fidelity)
        ei = self._expected_improvement(mu, var, incumbent)
        token_cost = max(float(self.token_cost_fn(config, fidelity)), 1e-9)
        return ei / token_cost

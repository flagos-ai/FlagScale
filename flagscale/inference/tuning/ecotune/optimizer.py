from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .acquisition import TokenAwareExpectedImprovement
from .search_space import SearchSpace
from .surrogate import MultiFidelityGPSurrogate


@dataclass
class Suggestion:
    config: Dict[str, object]
    fidelity: float
    acquisition_score: float
    promoted: bool


@dataclass
class EvaluationResult:
    score: float
    token_cost: float
    metadata: Optional[dict] = None


@dataclass
class Observation:
    config: Dict[str, object]
    fidelity: float
    score: float
    token_cost: float
    cumulative_tokens: float
    promoted: bool


class EcoTuneOptimizer:
    def __init__(
        self,
        search_space: SearchSpace,
        token_cost_fn: Callable[[Dict[str, object], float], float],
        total_budget: float,
        fidelity_levels: Optional[List[float]] = None,
        r_min: float = 0.05,
        r_max: float = 1.0,
        promotion_threshold: float = 0.05,
        initial_design_size: int = 6,
        candidates_per_step: int = 64,
        random_seed: int = 42,
        bootstrap_config: Optional[Dict[str, object]] = None,
        bootstrap_fidelity: Optional[float] = None,
    ):
        self.search_space = search_space
        self.token_cost_fn = token_cost_fn
        self.total_budget = float(total_budget)
        self.remaining_budget = float(total_budget)
        self.promotion_threshold = float(promotion_threshold)
        self.initial_design_size = int(initial_design_size)
        self.candidates_per_step = int(candidates_per_step)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        if fidelity_levels is None:
            self.fidelity_levels = list(np.linspace(self.r_min, self.r_max, 10))
        else:
            normalized_levels: List[float] = []
            for fidelity in fidelity_levels:
                try:
                    value = float(fidelity)
                except (TypeError, ValueError):
                    continue
                if self.r_min <= value <= self.r_max:
                    normalized_levels.append(value)

            if not normalized_levels:
                self.fidelity_levels = list(np.linspace(self.r_min, self.r_max, 10))
            else:
                unique_sorted = sorted(set(normalized_levels))
                if not np.any(np.isclose(unique_sorted, self.r_max)):
                    unique_sorted.append(self.r_max)
                    unique_sorted = sorted(set(unique_sorted))
                self.fidelity_levels = unique_sorted
        self.rng = np.random.default_rng(random_seed)
        self.surrogate = MultiFidelityGPSurrogate(n_dims=search_space.n_dims)
        self.acquisition = TokenAwareExpectedImprovement(
            surrogate=self.surrogate,
            token_cost_fn=self.token_cost_fn,
            incumbent_fidelity=self.r_max - 1e-8,
        )
        self.bootstrap_config = dict(bootstrap_config) if bootstrap_config is not None else None
        self.bootstrap_fidelity = (
            float(bootstrap_fidelity) if bootstrap_fidelity is not None else None
        )
        self.history: List[Observation] = []
        self.best_score = -np.inf
        self.best_config: Dict[str, object] = {}
        self._pending: Optional[Suggestion] = None

    def _affordable_fidelities(self, config: Dict[str, object]) -> List[float]:
        return [
            fidelity
            for fidelity in self.fidelity_levels
            if self.token_cost_fn(config, fidelity) <= self.remaining_budget
        ]

    def _initial_suggestion(self) -> Suggestion:
        config = self.search_space.sample(self.rng, n=1)[0]
        affordable = self._affordable_fidelities(config)
        if not affordable:
            raise RuntimeError("No affordable fidelity remains within budget")
        fidelity = float(affordable[min(len(self.history), len(affordable) - 1)])
        return Suggestion(config=config, fidelity=fidelity, acquisition_score=0.0, promoted=False)

    def _model_based_suggestion(self) -> Suggestion:
        candidates = self.search_space.sample(self.rng, n=self.candidates_per_step)
        best: Optional[Suggestion] = None

        for config in candidates:
            config_vec = self.search_space.to_vector(config)
            affordable = self._affordable_fidelities(config)
            if not affordable:
                continue

            scores = [
                self.acquisition.score(config, config_vec, fidelity) for fidelity in affordable
            ]
            score_at_rmax = -np.inf
            promoted = False
            if np.any(np.isclose(affordable, self.r_max)):
                score_at_rmax = self.acquisition.score(config, config_vec, self.r_max)
                promoted = score_at_rmax > self.promotion_threshold

            if promoted:
                fidelity = self.r_max
                acquisition_score = score_at_rmax
            else:
                best_idx = int(np.argmax(scores))
                fidelity = float(affordable[best_idx])
                acquisition_score = float(scores[best_idx])

            suggestion = Suggestion(
                config=config,
                fidelity=fidelity,
                acquisition_score=float(acquisition_score),
                promoted=promoted,
            )
            if best is None or suggestion.acquisition_score > best.acquisition_score:
                best = suggestion

        if best is None:
            raise RuntimeError("Unable to generate an affordable suggestion")
        return best

    def ask(self) -> Suggestion:
        if self.remaining_budget <= 0:
            raise RuntimeError("Token budget exhausted")
        if self._pending is not None:
            return self._pending

        if self.bootstrap_config is not None and self.surrogate.num_observations == 0:
            affordable = self._affordable_fidelities(self.bootstrap_config)
            if not affordable:
                raise RuntimeError("Bootstrap config is not affordable within budget")
            if self.bootstrap_fidelity is None:
                fidelity = affordable[0]
            else:
                fidelity = min(affordable, key=lambda x: abs(x - self.bootstrap_fidelity))
            self._pending = Suggestion(
                config=dict(self.bootstrap_config),
                fidelity=float(fidelity),
                acquisition_score=0.0,
                promoted=False,
            )
            return self._pending

        if self.surrogate.num_observations < self.initial_design_size:
            self._pending = self._initial_suggestion()
        else:
            self._pending = self._model_based_suggestion()
        return self._pending

    def tell(self, result: EvaluationResult) -> None:
        if self._pending is None:
            raise RuntimeError("tell() called before ask()")
        if result.token_cost > self.remaining_budget + 1e-8:
            raise ValueError("Reported token cost exceeds remaining budget")
        self._record_observation(
            config=self._pending.config,
            fidelity=self._pending.fidelity,
            result=result,
            promoted=bool(self._pending.promoted),
        )
        self._pending = None

    def _record_observation(
        self,
        config: Dict[str, object],
        fidelity: float,
        result: EvaluationResult,
        promoted: bool,
    ) -> None:
        config_vec = self.search_space.to_vector(config)
        self.surrogate.add_observation(config_vec, fidelity, result.score)
        self.remaining_budget = max(0.0, self.remaining_budget - float(result.token_cost))
        cumulative = self.total_budget - self.remaining_budget
        if result.score > self.best_score:
            self.best_score = float(result.score)
            self.best_config = dict(config)
        self.history.append(
            Observation(
                config=dict(config),
                fidelity=float(fidelity),
                score=float(result.score),
                token_cost=float(result.token_cost),
                cumulative_tokens=float(cumulative),
                promoted=bool(promoted),
            )
        )

    def should_stop(self, max_steps: Optional[int] = None) -> bool:
        if self.remaining_budget <= 0:
            return True
        if max_steps is not None and len(self.history) >= max_steps:
            return True
        return False

    def summary(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "remaining_budget": self.remaining_budget,
            "num_trials": len(self.history),
            "best_score": self.best_score,
            "best_config": self.best_config,
        }

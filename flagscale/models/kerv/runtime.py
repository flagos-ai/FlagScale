# Copyright 2026 FlagOS Contributors
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

"""Model-independent KERV speculative decoding runtime.

This module contains the KERV control path that does not depend on OpenVLA's
model implementation: batched candidate expansion, verification-tree layout,
relaxed candidate acceptance, dynamic threshold adjustment, and Kalman-based
action completion.  A VLA only needs to provide two small callables:

* ``draft_fn(paths) -> logits`` proposes the next token for every path;
* ``verify_fn(tree) -> logits`` evaluates all paths in one verifier call.

Keeping the control path in FlagScale makes KERV directly importable and
testable while leaving model weights, processors, and OpenVLA internals in
their respective packages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


class DraftCallable(Protocol):
    """Return next-token logits for equal-length candidate paths."""

    def __call__(self, paths: torch.LongTensor) -> torch.Tensor: ...


class VerifyCallable(Protocol):
    """Return verifier logits aligned with ``tree.candidate_paths``."""

    def __call__(self, tree: CandidateTree) -> torch.Tensor: ...


@dataclass(frozen=True)
class KERVConfig:
    """Configuration for one model-independent KERV runtime.

    ``candidate_depth`` counts tokens after the root token.  ``max_paths`` is
    a global beam cap, not a per-parent cap.  ``token_offset`` supports a
    verifier that projects only the contiguous action-token vocabulary.
    """

    action_dim: int = 7
    candidate_depth: int = 7
    top_k: int = 8
    max_paths: int = 48
    accept_threshold: int = 0
    token_offset: int = 0
    threshold_lower: int | None = None
    threshold_schedule: Literal["linear", "exponential"] = "linear"
    rollout_steps: int = 1
    kalman_interval: int = 0
    kalman_process_variance: float = 1.0
    kalman_measurement_variance: float = 1e-3
    kalman_history_window: int | None = None

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.candidate_depth <= 0:
            raise ValueError("candidate_depth must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_paths <= 0:
            raise ValueError("max_paths must be positive")
        if self.accept_threshold < 0:
            raise ValueError("accept_threshold must be non-negative")
        if self.threshold_lower is not None:
            if self.threshold_lower < 0:
                raise ValueError("threshold_lower must be non-negative")
            if self.threshold_lower > self.accept_threshold:
                raise ValueError("threshold_lower must not exceed accept_threshold")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if self.kalman_interval < 0:
            raise ValueError("kalman_interval must be non-negative")
        if self.kalman_process_variance < 0:
            raise ValueError("kalman_process_variance must be non-negative")
        if self.kalman_measurement_variance <= 0:
            raise ValueError("kalman_measurement_variance must be positive")


@dataclass(frozen=True)
class CandidateTree:
    """A compact tree plus the path view consumed by a verifier.

    Attributes:
        tokens: Token stored at each unique node, shape ``[nodes]``.
        parent_indices: Parent node for each node.  The root parent is ``-1``.
        attention_mask: Ancestor mask, shape ``[1, 1, nodes, nodes]``.
        position_ids: Tree depth for each node, shape ``[1, nodes]``.
        path_indices: Node indices for every leaf path, padded with ``-1``.
        candidate_paths: Token paths corresponding to ``path_indices``.
        path_scores: Cumulative draft log probability for every path.
    """

    tokens: torch.LongTensor
    parent_indices: torch.LongTensor
    attention_mask: torch.Tensor
    position_ids: torch.LongTensor
    path_indices: torch.LongTensor
    candidate_paths: torch.LongTensor
    path_scores: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def num_nodes(self) -> int:
        return int(self.tokens.numel())

    @property
    def num_paths(self) -> int:
        return int(self.path_indices.shape[0])

    @classmethod
    def from_paths(
        cls,
        paths: torch.LongTensor,
        path_scores: torch.Tensor | None = None,
    ) -> CandidateTree:
        """Deduplicate equal prefixes and construct a verifier tree."""

        if paths.ndim != 2 or paths.shape[0] == 0 or paths.shape[1] < 2:
            raise ValueError("paths must have shape [num_paths, length>=2]")
        if paths.dtype != torch.long:
            paths = paths.to(torch.long)
        if not torch.all(paths[:, 0] == paths[0, 0]):
            raise ValueError("all candidate paths must share one root token")

        device = paths.device
        node_tokens = [int(paths[0, 0].item())]
        parents = [-1]
        prefix_to_node: dict[tuple[int, ...], int] = {
            (node_tokens[0],): 0,
        }
        path_rows: list[list[int]] = []

        for row in paths.detach().to("cpu").tolist():
            node_path = [0]
            for depth in range(1, len(row)):
                prefix = tuple(int(token) for token in row[: depth + 1])
                node = prefix_to_node.get(prefix)
                if node is None:
                    parent_prefix = prefix[:-1]
                    parent = prefix_to_node[parent_prefix]
                    node = len(node_tokens)
                    prefix_to_node[prefix] = node
                    node_tokens.append(int(row[depth]))
                    parents.append(parent)
                node_path.append(node)
            path_rows.append(node_path)

        tokens = torch.tensor(node_tokens, dtype=torch.long, device=device)
        parent_indices = torch.tensor(parents, dtype=torch.long, device=device)
        path_indices = torch.tensor(path_rows, dtype=torch.long, device=device)
        return build_candidate_tree(
            tokens,
            parent_indices,
            path_indices=path_indices,
            path_scores=path_scores,
        )


@dataclass(frozen=True)
class VerificationResult:
    """Selected verification path and its accepted prefix."""

    best_path: int
    accept_length: int
    accepted_tokens: torch.LongTensor
    next_token: torch.LongTensor
    predicted_tokens: torch.LongTensor
    accept_lengths: torch.LongTensor


@dataclass(frozen=True)
class KERVStepResult:
    """Output of one KERV candidate-generation and verification round."""

    output_tokens: torch.LongTensor
    tree: CandidateTree
    verification: VerificationResult
    threshold: int
    used_kalman: bool


def _validate_tree_parents(parent_indices: torch.LongTensor) -> None:
    if parent_indices.ndim != 1 or parent_indices.numel() == 0:
        raise ValueError("parent_indices must be a non-empty 1D tensor")
    if int(parent_indices[0].item()) not in {-1, 0}:
        raise ValueError("the root parent must be -1 or 0")
    for node in range(1, int(parent_indices.numel())):
        parent = int(parent_indices[node].item())
        if parent < 0 or parent >= node:
            raise ValueError("every non-root parent must precede its child")


def _derive_leaf_paths(parent_indices: torch.LongTensor) -> torch.LongTensor:
    parent_cpu = parent_indices.detach().to("cpu").tolist()
    has_child = [False] * len(parent_cpu)
    for node in range(1, len(parent_cpu)):
        has_child[int(parent_cpu[node])] = True
    leaves = [node for node in range(1, len(parent_cpu)) if not has_child[node]]
    if not leaves:
        leaves = [0]
    paths: list[list[int]] = []
    for leaf in leaves:
        path = [leaf]
        while path[-1] != 0:
            path.append(int(parent_cpu[path[-1]]))
        paths.append(list(reversed(path)))
    max_length = max(len(path) for path in paths)
    padded = [path + [-1] * (max_length - len(path)) for path in paths]
    return torch.tensor(padded, dtype=torch.long, device=parent_indices.device)


def build_candidate_tree(
    candidate_tokens: torch.LongTensor,
    parent_indices: torch.LongTensor,
    path_indices: torch.LongTensor | None = None,
    path_scores: torch.Tensor | None = None,
) -> CandidateTree:
    """Build KERV's mask, positions, retrieval indices, and path tokens.

    This is deliberately device agnostic and runs on CPU, CUDA, and other
    PyTorch backends.  Tree topology is small and can be cached by the caller.
    """

    if candidate_tokens.ndim != 1:
        raise ValueError("candidate_tokens must be a 1D tensor")
    if candidate_tokens.numel() != parent_indices.numel():
        raise ValueError("candidate_tokens and parent_indices must align")
    if candidate_tokens.device != parent_indices.device:
        raise ValueError("candidate_tokens and parent_indices must share a device")
    _validate_tree_parents(parent_indices)

    device = candidate_tokens.device
    node_count = int(candidate_tokens.numel())
    parent_cpu = parent_indices.detach().to("cpu").tolist()
    mask = torch.zeros((node_count, node_count), dtype=torch.bool, device=device)
    positions = torch.empty((node_count,), dtype=torch.long, device=device)
    for node in range(node_count):
        current = node
        depth = 0
        while True:
            mask[node, current] = True
            if current == 0:
                break
            current = int(parent_cpu[current])
            depth += 1
        positions[node] = depth

    if path_indices is None:
        path_indices = _derive_leaf_paths(parent_indices)
    else:
        if path_indices.ndim != 2 or path_indices.shape[0] == 0:
            raise ValueError("path_indices must have shape [num_paths, length]")
        path_indices = path_indices.to(device=device, dtype=torch.long)
        valid = path_indices >= 0
        if bool(valid.any()) and int(path_indices[valid].max().item()) >= node_count:
            raise ValueError("path_indices contains a node outside the tree")
        if not torch.all(path_indices[:, 0] == 0):
            raise ValueError("every path must start at the root node")

    safe_indices = path_indices.clamp_min(0)
    candidate_paths = candidate_tokens[safe_indices]
    candidate_paths = candidate_paths.masked_fill(path_indices < 0, -1)
    if path_scores is None:
        path_scores = torch.zeros((path_indices.shape[0],), dtype=torch.float32, device=device)
    else:
        if path_scores.ndim != 1 or path_scores.shape[0] != path_indices.shape[0]:
            raise ValueError("path_scores must contain one value per path")
        path_scores = path_scores.to(device=device)

    return CandidateTree(
        tokens=candidate_tokens.to(torch.long),
        parent_indices=parent_indices.to(torch.long),
        attention_mask=mask.unsqueeze(0).unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        path_indices=path_indices,
        candidate_paths=candidate_paths,
        path_scores=path_scores,
    )


def generate_candidate_tree(
    draft_fn: DraftCallable,
    root_token: int | torch.Tensor,
    config: KERVConfig,
    *,
    device: torch.device | str | None = None,
) -> CandidateTree:
    """Generate a fixed-depth candidate tree through batched beam expansion."""

    if isinstance(root_token, torch.Tensor):
        if root_token.numel() != 1:
            raise ValueError("root_token must contain exactly one token")
        resolved_device = root_token.device
        root = int(root_token.item())
    else:
        resolved_device = torch.device(device or "cpu")
        root = int(root_token)

    paths = torch.tensor([[root]], dtype=torch.long, device=resolved_device)
    scores = torch.zeros((1,), dtype=torch.float32, device=resolved_device)
    for _ in range(config.candidate_depth):
        logits = draft_fn(paths)
        if logits.ndim != 2 or logits.shape[0] != paths.shape[0]:
            raise ValueError("draft_fn must return [num_paths, vocab_size] logits")
        if logits.shape[1] == 0:
            raise ValueError("draft_fn returned an empty vocabulary")
        branch_count = min(config.top_k, int(logits.shape[1]))
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        # Stable ordering makes equal-probability candidates deterministic:
        # because vocabulary columns are ordered by token id, smaller ids win.
        branch_tokens = torch.argsort(log_probabilities, dim=-1, descending=True, stable=True)[
            :, :branch_count
        ]
        branch_scores = torch.gather(log_probabilities, 1, branch_tokens)
        expanded_scores = scores[:, None] + branch_scores
        parent_rows = torch.arange(paths.shape[0], device=paths.device)[:, None]
        parent_rows = parent_rows.expand(-1, branch_count).reshape(-1)
        expanded_paths = torch.cat(
            (paths[parent_rows], branch_tokens.reshape(-1, 1).to(torch.long)),
            dim=1,
        )
        expanded_scores = expanded_scores.reshape(-1)
        keep = min(config.max_paths, int(expanded_scores.numel()))
        order = torch.argsort(expanded_scores, descending=True, stable=True)[:keep]
        scores = expanded_scores[order]
        paths = expanded_paths[order]

    return CandidateTree.from_paths(paths, scores)


def verify_candidates(
    candidate_tokens: CandidateTree | torch.LongTensor,
    verifier_logits: torch.Tensor,
    accept_threshold: int | float | None = 0,
    *,
    token_offset: int = 0,
    scores: torch.Tensor | None = None,
) -> VerificationResult:
    """Select the path with the longest verifier-consistent prefix.

    Candidate position zero is the shared root.  Logits at position ``i``
    predict candidate position ``i + 1``.  Ties are resolved by input order,
    matching KERV's priority-ordered candidate layout.  ``scores`` is accepted
    for API compatibility and checked, but intentionally does not override
    KERV's longest-prefix rule.
    """

    paths = (
        candidate_tokens.candidate_paths
        if isinstance(candidate_tokens, CandidateTree)
        else candidate_tokens
    )
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("candidate paths must have shape [num_paths, length>=2]")
    if verifier_logits.ndim != 3:
        raise ValueError("verifier_logits must have shape [num_paths, length, vocab]")
    if verifier_logits.shape[0] != paths.shape[0]:
        raise ValueError("candidate and verifier path counts differ")
    if verifier_logits.shape[1] < paths.shape[1]:
        raise ValueError("verifier logits must cover candidate tokens and one next-token position")
    if scores is not None and scores.shape[0] != paths.shape[0]:
        raise ValueError("scores must contain one value per path")

    aligned_logits = verifier_logits[:, : paths.shape[1] - 1]
    predictions = torch.argmax(aligned_logits, dim=-1).to(torch.long)
    predictions = predictions + int(token_offset)
    targets = paths[:, 1:].to(predictions.device)
    valid = targets >= 0
    threshold = 0.0 if accept_threshold is None else float(accept_threshold)
    if threshold < 0:
        raise ValueError("accept_threshold must be non-negative")
    matches = valid & ((targets - predictions).abs() <= threshold)
    accept_lengths = torch.cumprod(matches.to(torch.long), dim=1).sum(dim=1)
    best_path_tensor = torch.argmax(accept_lengths)
    best_path = int(best_path_tensor.item())
    accept_length = int(accept_lengths[best_path].item())
    accepted = paths[best_path, 1 : 1 + accept_length].to(verifier_logits.device)
    next_token_logits = verifier_logits[best_path, accept_length]
    next_token = torch.argmax(next_token_logits).to(torch.long) + int(token_offset)
    return VerificationResult(
        best_path=best_path,
        accept_length=accept_length,
        accepted_tokens=accepted,
        next_token=next_token,
        predicted_tokens=predictions,
        accept_lengths=accept_lengths,
    )


def compute_dynamic_threshold(
    step_idx: int,
    total_steps: int,
    start: int | float,
    lower: int | float,
    schedule: Literal["linear", "exponential"] = "linear",
) -> float:
    """Compute KERV's kinematic-aware relaxed acceptance threshold."""

    step = max(int(step_idx), 0)
    horizon = max(int(total_steps), 1)
    start_value = float(start)
    lower_value = float(lower)
    if start_value < lower_value:
        raise ValueError("start threshold must be >= lower threshold")
    if schedule == "linear":
        # A stretched exponential decays slowly early in the manipulation and
        # approaches the strict lower bound near the expected final step.
        epsilon = 1e-30
        power = 3.0
        tau = horizon / ((-math.log(epsilon)) ** (1.0 / power))
        decay = math.exp(-((float(step) / tau) ** power))
        value = lower_value + (start_value - lower_value) * decay
    elif schedule == "exponential":
        if start_value == 0:
            return lower_value
        ratio = max(0.0, min(1.0, lower_value / start_value))
        value = start_value * (ratio ** (step / horizon))
    else:
        raise ValueError(f"unsupported threshold schedule: {schedule}")
    return max(lower_value, value)


def kalman_predict(
    history: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    *,
    process_variance: float = 1.0,
    measurement_variance: float = 1e-3,
    initial_estimate_error: float = 10.0,
) -> torch.Tensor:
    """Predict the next scalar or action vector with independent Kalman filters."""

    # The state dimension is tiny (seven for KERV) and belongs to the control
    # path.  Running it on CPU avoids a chain of small device kernels and keeps
    # the implementation portable to every PyTorch accelerator backend.
    values = torch.as_tensor(history, dtype=torch.float64, device="cpu")
    if values.numel() == 0:
        raise ValueError("history must contain at least one observation")
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise ValueError("history must have shape [steps] or [steps, dimensions]")
    if process_variance < 0:
        raise ValueError("process_variance must be non-negative")
    if measurement_variance <= 0:
        raise ValueError("measurement_variance must be positive")

    estimate = values[0].clone()
    error = torch.full_like(estimate, float(initial_estimate_error))
    for measurement in values:
        predicted_error = error + float(process_variance)
        gain = predicted_error / (predicted_error + float(measurement_variance))
        estimate = estimate + gain * (measurement - estimate)
        error = (1.0 - gain) * predicted_error
    estimate = estimate.to(torch.float32)
    return estimate[0] if squeeze else estimate


@dataclass
class KERVRuntime:
    """Small native orchestrator for one batched KERV verification round."""

    config: KERVConfig
    action_history: list[torch.LongTensor] = field(default_factory=list)

    def reset(self) -> None:
        """Clear rollout-local action history."""

        self.action_history.clear()

    def threshold_for_step(self, step_idx: int) -> int:
        """Return the integer token-distance threshold for a rollout step."""

        lower = self.config.threshold_lower
        if lower is None:
            return int(self.config.accept_threshold)
        value = compute_dynamic_threshold(
            step_idx,
            self.config.rollout_steps,
            self.config.accept_threshold,
            lower,
            self.config.threshold_schedule,
        )
        return int(round(value))

    def _complete_with_kalman(
        self,
        accepted: torch.LongTensor,
        history: Sequence[Sequence[int]] | torch.Tensor | None,
    ) -> torch.LongTensor:
        accepted = accepted[: self.config.action_dim]
        remaining = self.config.action_dim - int(accepted.numel())
        if remaining <= 0:
            return accepted
        source: Any = history if history is not None else self.action_history
        if len(source) == 0:
            fallback = (
                accepted[-1:]
                if accepted.numel()
                else torch.zeros((1,), dtype=torch.long, device=accepted.device)
            )
            return torch.cat((accepted, fallback.expand(remaining)))
        if isinstance(source, torch.Tensor):
            source_tensor = source.to(dtype=torch.float32)
        elif isinstance(source[0], torch.Tensor):
            source_tensor = torch.stack([item.detach().to(dtype=torch.float32) for item in source])
        else:
            source_tensor = torch.as_tensor(source, dtype=torch.float32)
        if self.config.kalman_history_window is not None:
            source_tensor = source_tensor[-self.config.kalman_history_window :]
        prediction = (
            kalman_predict(
                source_tensor,
                process_variance=self.config.kalman_process_variance,
                measurement_variance=self.config.kalman_measurement_variance,
            )
            .round()
            .to(device=accepted.device, dtype=torch.long)
        )
        start = int(accepted.numel())
        tail = prediction[start : start + remaining]
        if tail.numel() < remaining:
            fallback = (
                tail[-1:]
                if tail.numel()
                else (accepted[-1:] if accepted.numel() else prediction[-1:])
            )
            tail = torch.cat((tail, fallback.expand(remaining - tail.numel())))
        return torch.cat((accepted, tail))

    def step(
        self,
        draft_fn: DraftCallable,
        verify_fn: VerifyCallable,
        root_token: int | torch.Tensor,
        *,
        step_idx: int = 0,
        history: Sequence[Sequence[int]] | torch.Tensor | None = None,
        force_kalman: bool | None = None,
    ) -> KERVStepResult:
        """Generate, verify, and optionally Kalman-complete one action round."""

        tree = generate_candidate_tree(draft_fn, root_token, self.config)
        logits = verify_fn(tree)
        threshold = self.threshold_for_step(step_idx)
        verification = verify_candidates(
            tree,
            logits,
            threshold,
            token_offset=self.config.token_offset,
        )
        if force_kalman is None:
            interval = self.config.kalman_interval
            use_kalman = interval > 0 and (int(step_idx) + 1) % interval == 0
        else:
            use_kalman = bool(force_kalman)
        if use_kalman:
            output = self._complete_with_kalman(verification.accepted_tokens, history)
        else:
            output = torch.cat((verification.accepted_tokens, verification.next_token.reshape(1)))[
                : self.config.action_dim
            ]
        if output.numel() == self.config.action_dim:
            self.action_history.append(output.detach().to("cpu"))
        return KERVStepResult(
            output_tokens=output,
            tree=tree,
            verification=verification,
            threshold=threshold,
            used_kalman=use_kalman,
        )


__all__ = [
    "CandidateTree",
    "DraftCallable",
    "KERVConfig",
    "KERVRuntime",
    "KERVStepResult",
    "VerificationResult",
    "VerifyCallable",
    "build_candidate_tree",
    "compute_dynamic_threshold",
    "generate_candidate_tree",
    "kalman_predict",
    "verify_candidates",
]

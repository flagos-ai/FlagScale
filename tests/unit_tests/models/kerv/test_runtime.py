# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flagscale.models.kerv import (
    CandidateTree,
    KERVConfig,
    KERVRuntime,
    build_candidate_tree,
    compute_dynamic_threshold,
    generate_candidate_tree,
    kalman_predict,
    verify_candidates,
)


def _logits_for_paths(paths: torch.LongTensor, next_token: int) -> torch.Tensor:
    vocabulary = max(int(paths.max().item()), next_token) + 2
    logits = torch.full(
        (paths.shape[0], paths.shape[1], vocabulary),
        -100.0,
        device=paths.device,
    )
    for path_index, path in enumerate(paths.tolist()):
        for position, token in enumerate(path[1:]):
            logits[path_index, position, token] = 100.0
        logits[path_index, paths.shape[1] - 1, next_token] = 100.0
    return logits


def test_candidate_tree_deduplicates_prefixes_and_builds_ancestor_mask():
    paths = torch.tensor([[10, 11, 12], [10, 11, 13], [10, 14, 15]], dtype=torch.long)
    scores = torch.tensor([-0.1, -0.2, -0.3])

    tree = CandidateTree.from_paths(paths, scores)

    assert torch.equal(tree.tokens, torch.tensor([10, 11, 12, 13, 14, 15]))
    assert torch.equal(tree.parent_indices, torch.tensor([-1, 0, 1, 1, 0, 4]))
    assert torch.equal(
        tree.path_indices,
        torch.tensor([[0, 1, 2], [0, 1, 3], [0, 4, 5]]),
    )
    assert torch.equal(tree.candidate_paths, paths)
    assert torch.equal(tree.position_ids, torch.tensor([[0, 1, 2, 2, 1, 2]]))
    expected_row = torch.tensor([True, True, False, True, False, False])
    assert torch.equal(tree.attention_mask[0, 0, 3], expected_row)
    assert torch.equal(tree.path_scores, scores)


def test_build_candidate_tree_derives_and_pads_leaf_paths():
    tree = build_candidate_tree(
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([-1, 0, 1, 0]),
    )

    assert torch.equal(tree.path_indices, torch.tensor([[0, 1, 2], [0, 3, -1]]))
    assert torch.equal(tree.candidate_paths, torch.tensor([[10, 11, 12], [10, 13, -1]]))
    assert tree.attention_mask.dtype == torch.bool
    assert tuple(tree.attention_mask.shape) == (1, 1, 4, 4)


@pytest.mark.parametrize(
    "parents",
    (
        torch.tensor([1, 0]),
        torch.tensor([-1, -1]),
        torch.tensor([-1, 1]),
    ),
)
def test_build_candidate_tree_rejects_invalid_topology(parents):
    with pytest.raises(ValueError):
        build_candidate_tree(torch.arange(parents.numel()), parents)


def test_generate_candidate_tree_batches_expansion_and_has_stable_tie_break():
    observed_shapes = []

    def draft(paths):
        observed_shapes.append(tuple(paths.shape))
        # Equal probabilities deliberately exercise the documented token-id tie break.
        return torch.zeros((paths.shape[0], 4), dtype=torch.float32)

    config = KERVConfig(action_dim=3, candidate_depth=2, top_k=2, max_paths=3)
    tree = generate_candidate_tree(draft, root_token=9, config=config)

    assert observed_shapes == [(1, 1), (2, 2)]
    assert torch.equal(
        tree.candidate_paths,
        torch.tensor([[9, 0, 0], [9, 0, 1], [9, 1, 0]]),
    )
    assert tree.num_paths == 3
    assert tree.num_nodes == 6


def test_verify_candidates_selects_longest_prefix_and_returns_next_token():
    paths = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 5], [0, 6, 7, 8]], dtype=torch.long)
    logits = torch.full((3, 4, 10), -100.0)
    predictions = (
        (1, 2, 9, 7),
        (1, 4, 5, 8),
        (6, 9, 8, 7),
    )
    for path_index, tokens in enumerate(predictions):
        for position, token in enumerate(tokens):
            logits[path_index, position, token] = 100.0

    result = verify_candidates(paths, logits)

    assert result.best_path == 1
    assert result.accept_length == 3
    assert torch.equal(result.accepted_tokens, torch.tensor([1, 4, 5]))
    assert result.next_token.item() == 8
    assert torch.equal(result.accept_lengths, torch.tensor([2, 3, 1]))


def test_verify_candidates_relaxed_threshold_and_token_offset():
    paths = torch.tensor([[100, 102, 104], [100, 103, 105]])
    logits = torch.full((2, 3, 8), -100.0)
    # Projected vocabulary begins at token 100.  Path 0 is within distance one;
    # path 1 fails at its first candidate token.
    logits[0, 0, 1] = 10.0
    logits[0, 1, 3] = 10.0
    logits[0, 2, 7] = 10.0
    logits[1, 0, 0] = 10.0
    logits[1, 1, 5] = 10.0
    logits[1, 2, 6] = 10.0

    result = verify_candidates(paths, logits, accept_threshold=1, token_offset=100)

    assert result.best_path == 0
    assert result.accept_length == 2
    assert torch.equal(result.accepted_tokens, torch.tensor([102, 104]))
    assert result.next_token.item() == 107


def test_verify_candidates_ties_use_input_order_and_require_next_token_logits():
    paths = torch.tensor([[0, 1, 2], [0, 1, 3]])
    logits = _logits_for_paths(paths, next_token=4)
    result = verify_candidates(paths, logits)
    assert result.best_path == 0

    with pytest.raises(ValueError, match="next-token position"):
        verify_candidates(paths, logits[:, :-1])


def test_dynamic_threshold_is_bounded_and_monotonic():
    values = [
        compute_dynamic_threshold(step, 100, start=14, lower=5, schedule="linear")
        for step in (0, 25, 50, 75, 100)
    ]
    assert values[0] == pytest.approx(14.0)
    assert values[-1] == pytest.approx(5.0)
    assert all(left >= right for left, right in zip(values, values[1:]))
    assert compute_dynamic_threshold(100, 100, 10, 2, "exponential") == pytest.approx(2)


def test_kalman_predict_handles_scalar_and_vector_history():
    scalar = kalman_predict([5.0, 5.0, 5.0])
    vector = kalman_predict([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]])

    assert scalar.item() == pytest.approx(5.0)
    assert tuple(vector.shape) == (2,)
    assert 2.0 < vector[0].item() <= 3.0
    assert 11.0 < vector[1].item() <= 12.0


def test_native_runtime_smoke_executes_draft_verify_and_kalman_paths():
    config = KERVConfig(
        action_dim=3,
        candidate_depth=2,
        top_k=2,
        max_paths=3,
        accept_threshold=0,
    )
    runtime = KERVRuntime(config)

    def draft(paths):
        logits = torch.full((paths.shape[0], 10), -20.0)
        logits[:, 1] = 20.0
        logits[:, 2] = 19.0
        return logits

    def verify(tree):
        return _logits_for_paths(tree.candidate_paths, next_token=9)

    direct = runtime.step(draft, verify, root_token=0)
    assert torch.equal(direct.output_tokens, torch.tensor([1, 1, 9]))
    assert direct.verification.accept_length == 2
    assert direct.used_kalman is False

    completed = runtime.step(
        draft,
        verify,
        root_token=0,
        history=torch.tensor([[10, 20, 30], [12, 22, 32]]),
        force_kalman=True,
    )
    assert torch.equal(completed.output_tokens[:2], torch.tensor([1, 1]))
    assert completed.output_tokens[2].item() == 32
    assert completed.used_kalman is True

    runtime.reset()
    assert runtime.action_history == []


def test_config_rejects_invalid_runtime_parameters():
    with pytest.raises(ValueError, match="action_dim"):
        KERVConfig(action_dim=0)
    with pytest.raises(ValueError, match="threshold_lower"):
        KERVConfig(accept_threshold=2, threshold_lower=3)
    with pytest.raises(ValueError, match="measurement_variance"):
        KERVConfig(kalman_measurement_variance=0)

import json
from argparse import ArgumentParser
from itertools import islice

import numpy as np
import pytest
from torch.utils.data import DataLoader

from megatron.training.arguments import _add_sft_args
from megatron.training.datasets.streaming_sft_dataset import (
    StreamingPackedSFTDataset,
    _rows,
    make_batch_item,
    window_packs,
)


class ToyTokenizer:
    pad = 0

    def tokenize_conversation(self, messages, return_target, add_generation_prompt):
        ids = np.asarray(messages[-1]["content"], dtype=np.int64)
        return ids, ids.copy()


def test_opt_in_and_capacity():
    args = _add_sft_args(ArgumentParser()).parse_args([])
    assert not args.sft_online_packing
    assert args.sft_packing_window_size == 8
    sizes = [6, 6, 4, 4, 11]
    samples = [(range(n), idx) for idx, n in enumerate(sizes)]

    def groups(w):
        return [[idx for _, idx in pack] for pack in window_packs(samples, 10, w)]

    assert groups(1) == [[0], [1, 2], [3]]
    assert groups(8) == [[0, 2], [1, 3]]
    # Overflow an actual window, check both carry and the final flush.
    samples *= 5
    packed = groups(2)
    assert sorted(i for group in packed for i in group) == sorted(list(range(4)) * 5)
    assert all(sum(sizes[i] for i in group) <= 10 for group in packed)


def test_labels_positions_and_padding():
    rows = [(np.array([2, 3, 4]), np.array([-100, -100, 4])), (np.array([5, 6]), np.array([5, 6]))]
    result = make_batch_item(rows, 8, 0)
    assert result["tokens"].tolist() == [2, 3, 4, 5, 6, 0, 0, 0]
    assert result["labels"].tolist() == [-100, 4, -100, 6, -100, -100, -100, -100]
    assert result["loss_mask"].tolist() == [0, 1, 0, 1, 0, 0, 0, 0]
    assert result["position_ids"].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    assert result["cu_seqlens"].tolist() == [0, 3, 8]
    assert result["max_seqlen"].item() == 5


@pytest.mark.parametrize("size, aligned", [(3, 4), (4, 4), (5, 8), (14, 16)])
def test_variable_pack_preserves_targets_and_boundaries(size, aligned):
    samples = [(np.arange(1, 3), np.arange(1, 3)), (np.arange(1, size - 1), np.arange(1, size - 1))]
    fixed = make_batch_item(samples, 16, 0)
    packed = make_batch_item(samples, 16, 0, pad_to_multiple=4)
    for key in ("tokens", "labels", "position_ids", "loss_mask"):
        assert packed[key].equal(fixed[key][:aligned])
    assert packed["cu_seqlens"].tolist() == [0, 2, aligned]
    assert packed["max_seqlen"].item() == max(2, aligned - 2)


@pytest.mark.parametrize("workers", [0, 2])
def test_sharding_and_resume(tmp_path, workers):
    paths = [tmp_path / name for name in ["a.jsonl", "b.jsonl"]]
    for source, path in enumerate(paths):
        with path.open("w") as stream:
            for index in range(37):
                value = 1 + index + 37 * source
                message = {"role": "assistant", "content": [value] * (2 + index % 4)}
                stream.write(json.dumps({"messages": [message]}) + "\n")
    ids = [
        messages[0]["content"][0]
        for p in paths
        for split in [(0, 0.7), (0.7, 1)]
        for shard in range(4)
        for messages in _rows(p, shard, 4, split)
    ]
    assert sorted(ids) == list(range(1, 75))

    def read(rank, consumed, count):
        ds = StreamingPackedSFTDataset((paths, [1, 2]), ToyTokenizer(), 10, 3, 42, rank, 2)
        ds.consumed_samples = consumed
        return [b["tokens"].tolist() for b in islice(DataLoader(ds, num_workers=workers), count)]

    for rank in (0, 1):
        baseline = read(rank, 0, 27)
        assert baseline[5:] == read(rank, 10, 22)

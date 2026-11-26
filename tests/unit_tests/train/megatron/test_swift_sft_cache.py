# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Focused tests for the shared Swift SFT encoded and packing caches."""

import json
import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("datasets")

_MEGATRON_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "flagscale", "train", "megatron"
    )
)
sys.path.insert(0, _MEGATRON_ROOT)


class _FakeTemplate:
    max_length = 128

    def __init__(self, calls):
        self.calls = calls
        self.packing = False
        self.padding_free = True

    def encode(self, row, return_length=False):
        self.calls.append(row["messages"][0]["content"])
        length = len(row["messages"][0]["content"]) + 1
        encoded = {
            "input_ids": list(range(length)),
            "labels": [-100, *range(1, length)],
            "length": length,
        }
        if return_length:
            encoded["lengths"] = [length]
        return encoded


def test_packing_budgets_per_row_cp_padding():
    """Rows that fit by raw length must split if CP padding crosses the cap."""
    from data.dataset import PackingDataset

    class LengthDataset:
        lengths = [[33], [80]]

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, index):
            if index == "lengths":
                return self.lengths
            return {"index": index}

    class Template:
        max_length = 128
        packing = False
        padding_free = False

    packed = PackingDataset(
        Template(),
        LengthDataset(),
        packing_length=128,
        packing_alignment=32,
        packing_num_proc=1,
        load_from_cache_file=False,
        multiprocessing_context="fork",
    )

    # Raw total 113 fits, but per-row CP padding makes it
    # round_up(33, 32) + round_up(80, 32) = 160, so it must split.
    assert len(packed) == 2
    assert sorted(packed.packed_length) == [64, 96]
    assert all(length <= 128 for length in packed.packed_length)


def test_second_build_reuses_encoded_and_packing_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "modelscope-cache"))
    from data import swift_sft_dataset as module

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    data_file = tmp_path / "train.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "user", "content": "three"}, {"role": "assistant", "content": "b"}]},
    ]
    data_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    calls = []
    monkeypatch.setattr(module, "_build_template", lambda config: _FakeTemplate(calls))
    config = module.SwiftSFTConfig(
        model_dir=str(model_dir),
        dataset=[str(data_file)],
        max_length=128,
        packing_length=128,
        # None keeps this focused unit test in-process; production values >1
        # exercise datasets' multiprocessing workers.
        dataset_num_proc=None,
        packing_num_proc=1,
        cache_dir=str(tmp_path / "shared-cache"),
    )

    first = module.SwiftSFTDataset(config, config.dataset, is_train=True)
    assert len(calls) == len(rows)
    first_signature = [[row["input_ids"] for row in first[i]] for i in range(len(first))]

    second = module.SwiftSFTDataset(config, config.dataset, is_train=True)
    assert len(calls) == len(rows), "the second build unexpectedly tokenized the source again"
    second_signature = [[row["input_ids"] for row in second[i]] for i in range(len(second))]
    assert second_signature == first_signature
    assert list((tmp_path / "shared-cache" / "encoded").glob("*/_SUCCESS"))
    assert list((tmp_path / "shared-cache" / "packing").glob("*.pkl"))


def test_cache_key_changes_when_encoding_inputs_change(tmp_path):
    from data.swift_sft_dataset import SwiftSFTConfig, _cache_manifest, _manifest_key

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("{}\n", encoding="utf-8")

    config = SwiftSFTConfig(model_dir=str(model_dir), dataset=[str(data_file)], max_length=128)
    key = _manifest_key(_cache_manifest(config, config.dataset, is_train=True))
    config.max_length = 256
    changed_key = _manifest_key(_cache_manifest(config, config.dataset, is_train=True))
    assert changed_key != key

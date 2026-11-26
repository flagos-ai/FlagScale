# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Tests for SFT real and physical packed-sequence metadata."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_MEGATRON_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "flagscale", "train", "megatron"
    )
)
_TRAIN_ROOT = os.path.dirname(_MEGATRON_ROOT)
sys.path.insert(0, _TRAIN_ROOT)
sys.path.insert(0, _MEGATRON_ROOT)


def test_dual_cu_seqlens_reconstruct_physical_padding_mask():
    from data.packed_seq_utils import padding_mask_from_cu_seqlens

    mask = padding_mask_from_cu_seqlens(
        torch.tensor([0, 3, 8, 8], dtype=torch.int32),
        torch.tensor([0, 4, 12, 16], dtype=torch.int32),
    )

    assert torch.equal(
        mask,
        torch.tensor(
            [
                False, False, False, True,
                False, False, False, False, False, True, True, True,
                True, True, True, True,
            ]
        ),
    )


def test_swift_collator_emits_real_and_physical_cu_seqlens():
    """Real lengths exclude CP/tail padding while physical lengths retain it."""
    from data.swift_collator import swift_sft_collate

    class FakeTemplate:
        # Per-row physical lengths are rounded to 2 * CP = 4.
        sequence_parallel_size = 2

        def data_collator(self, batch, padding_to=None):
            del batch, padding_to
            return {
                "input_ids": torch.arange(16).unsqueeze(0),
                # The first two real tokens are prompts and therefore ignored
                # by CE, but they must remain valid for router/indexer losses.
                "labels": torch.tensor(
                    [[-100, -100, 2, -100, -100, 5, 6, 7, 8, -100, -100, -100,
                      -100, -100, -100, -100]]
                ),
                "position_ids": torch.tensor(
                    [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]]
                ),
                "seq_lens": [3, 5],
            }

    out = swift_sft_collate([{}], template=FakeTemplate(), padding_to=16)
    assert "real_token_mask" not in out
    assert torch.equal(
        out["cu_seqlens"], torch.tensor([[0, 3, 8, 8]], dtype=torch.int32)
    )
    assert torch.equal(
        out["cu_seqlens_padded"], torch.tensor([[0, 4, 12, 16]], dtype=torch.int32)
    )
    # Prompt rows remain absent from CE supervision; their validity for MoE/DSA
    # is represented by the real segment length, not by loss_mask.
    assert not bool(out["loss_mask"][0, 0])


def test_megatron_sft_dataset_emits_real_and_physical_cu_seqlens():
    """The offline backend must expose the same dual-boundary semantics."""
    from megatron.training.datasets.sft_dataset import SFTDataset

    class FakeTokenizer:
        pad = 99
        eod = 0

        def tokenize_conversation(self, conversation, return_target, add_generation_prompt):
            del conversation, return_target, add_generation_prompt
            # One prompt token followed by two supervised tokens.
            return torch.tensor([10, 11, 12]), torch.tensor([-100, 11, 12])

    # Bypass MegatronDataset.__init__; __getitem__ only needs these fields.
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.dataset = [[
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "answer"},
    ]]
    dataset.indices = np.array([0], dtype=np.int64)
    dataset.num_samples = 1
    dataset.config = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        sequence_length=8,
        context_parallel_size=2,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
    )

    out = dataset[0]

    assert "real_token_mask" not in out
    assert torch.equal(out["cu_seqlens"], torch.tensor([0, 3], dtype=torch.int32))
    assert torch.equal(out["cu_seqlens_padded"], torch.tensor([0, 8], dtype=torch.int32))


def test_non_binary_loss_scale_uses_contiguous_dsv4_cp_shard(monkeypatch):
    from model_specific_batch import slice_loss_scale_on_cp_rank

    cp_group = object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 1)
    batch = {"loss_scale": torch.arange(16, dtype=torch.float32).unsqueeze(0)}

    result = slice_loss_scale_on_cp_rank(
        batch,
        cp_group,
        use_contiguous_cp=True,
    )

    assert torch.equal(
        result["loss_scale"],
        torch.arange(8, 16, dtype=torch.float32).unsqueeze(0),
    )


def test_dsv4_padding_mask_uses_contiguous_cp_shard(monkeypatch):
    import train_gpt

    cp_group = object()
    monkeypatch.setattr(train_gpt, "get_context_parallel_group", lambda: cp_group)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 1)

    mask = train_gpt._padding_mask_from_cu_seqlens(
        torch.tensor([0, 3, 8, 8], dtype=torch.int32),
        torch.tensor([0, 4, 12, 16], dtype=torch.int32),
        local_seq_length=8,
        micro_batch_size=1,
        use_contiguous_cp=True,
    )

    assert torch.equal(
        mask,
        torch.tensor([[False, True, True, True, True, True, True, True]]),
    )

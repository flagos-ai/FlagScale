# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Tests for the DeepSeek-V4-specific tokenizer vocabulary contract."""

import os
import sys

_MEGATRON_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "flagscale", "train", "megatron"
    )
)
sys.path.insert(0, _MEGATRON_ROOT)


def test_deepseek_v4_vocab_includes_added_tokens_without_changing_generic_hf():
    from megatron.training.tokenizer.tokenizer import _DeepSeekV4TokenizerFS, _HFTokenizerFS

    class FakeTokenizer:
        vocab_size = 100

        def __len__(self):
            return 103

    generic = _HFTokenizerFS.__new__(_HFTokenizerFS)
    generic.tokenizer = FakeTokenizer()
    deepseek = _DeepSeekV4TokenizerFS.__new__(_DeepSeekV4TokenizerFS)
    deepseek.tokenizer = FakeTokenizer()

    assert generic.vocab_size == 100
    assert deepseek.vocab_size == 103

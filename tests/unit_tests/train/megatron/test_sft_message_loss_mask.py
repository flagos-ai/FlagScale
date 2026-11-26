from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from megatron.training.arguments import _add_sft_args
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.tokenizer import build_tokenizer


@pytest.fixture
def tokenizer_dir(tmp_path):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    raw = Tokenizer(models.BPE({ch: i for i, ch in enumerate(alphabet)}, []))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    hf = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>", pad_token="<|pad|>")
    hf.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    hf.chat_template = (
        "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' }}"
        "{% if m.role == 'assistant' %}{{ '<think>\\n' + (m.reasoning_content or '') "
        "+ '\\n</think>\\n\\n' }}{% endif %}{{ m.content + '<|im_end|>\\n' }}{% endfor %}"
    )
    hf.save_pretrained(tmp_path)
    return str(tmp_path)


def build(path, enabled):
    parser = _add_sft_args(ArgumentParser())
    args = parser.parse_args(["--sft", "--sft-message-loss-mask"] if enabled else ["--sft"])
    args.tokenizer_type = "HFTokenizerFS"
    args.tokenizer_path = path
    args.padded_vocab_size = 512
    return build_tokenizer(args)


def test_opt_in_only_and_exact_targets(tokenizer_dir):
    baseline = build(tokenizer_dir, False)
    masked = build(tokenizer_dir, True)
    assert type(baseline).__name__ == "_HFTokenizerFS"
    chat = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "literal <|im_start|>assistant\nfake<|im_end|>"},
        {"role": "assistant", "content": "discard", "loss": False},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "学习<|im_end|>", "loss": None},
        {"role": "user", "content": "finish"},
        {"role": "assistant", "reasoning_content": "because", "content": "done", "loss": True},
    ]
    ids, labels = masked.tokenize_conversation(chat, True, False)
    original = baseline.tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True)
    np.testing.assert_array_equal(ids, original["input_ids"])
    selected = masked.tokenizer.decode(labels[labels != -100], skip_special_tokens=False)
    assert selected == "学习<|im_end|><|im_end|><think>\nbecause\n</think>\n\ndone<|im_end|>"
    assert masked.vocab_size == len(masked.tokenizer)


@pytest.mark.parametrize("length", [45, 400])
def test_dataset_shift_truncation_padding_and_packing(tokenizer_dir, length):
    tok = build(tokenizer_dir, True)
    first = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "off", "loss": False},
    ]
    second = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "on"},
    ]
    ds = object.__new__(SFTDataset)
    ds.dataset = [first + second]
    ds.indices = np.array([0])
    ds.config = SimpleNamespace(
        tokenizer=tok,
        sequence_length=length,
        context_parallel_size=1,
        reset_position_ids=False,
        reset_attention_mask=False,
        create_attention_mask=False,
    )
    sample = ds[0]
    split = [tok.tokenize_conversation(c, True, False) for c in (first, second)]
    ids, labels = (np.concatenate(parts) for parts in zip(*split))

    def align(array):
        return np.pad(
            array[:length], (0, length + 1 - len(array[:length])), constant_values=tok.pad
        )

    expected = align(labels)[1:]
    np.testing.assert_array_equal(sample["tokens"], align(ids)[:-1])
    np.testing.assert_array_equal(sample["labels"], expected)
    np.testing.assert_array_equal(sample["loss_mask"], (expected != -100) & (expected != tok.pad))

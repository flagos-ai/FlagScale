# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from megatron.core.tokenizers.text.libraries import sft_tokenizer as core_sft_tokenizer
from megatron.training.arguments import _add_sft_args, _add_tokenizer_args
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.tokenizer import sft_tokenizer
from megatron.training.tokenizer.tokenizer import build_tokenizer

CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}
{{ '<' + message['role'] + '>' }}{% if message['role'] == 'assistant' %}
{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}
{% else %}{{ message['content'] }}{% endif %}{% endfor %}
{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"""
CONVERSATION = [
    {"role": "system", "content": "instructions"},
    {"role": "user", "content": "answer one"},
    {"role": "assistant", "content": "answer one"},
    {"role": "tool", "content": "tool result"},
    {"role": "assistant", "content": "answer two"},
    {"role": "user", "content": "question"},
]


@pytest.fixture
def hf_tokenizer(monkeypatch):
    special = ["<pad>", "<unk>", "<eos>", "<bos>", "<system>", "<user>", "<assistant>", "<tool>"]
    words = ["instructions", "answer", "one", "two", "tool", "result", "question"]
    vocab = {token: index for index, token in enumerate(special + words)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<bos>",
        additional_special_tokens=special[4:],
        chat_template=CHAT_TEMPLATE,
    )
    monkeypatch.setattr(
        core_sft_tokenizer.transformers.AutoTokenizer, "from_pretrained", lambda **kwargs: tokenizer
    )
    return tokenizer


@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_default_masks_only_assistant_bodies_and_eos(hf_tokenizer, add_generation_prompt):
    tokenizer = sft_tokenizer.CustomSFTTokenizer("unused", "default")
    tokens, targets = tokenizer.tokenize_conversation(CONVERSATION, True, add_generation_prompt)
    expected_words = [
        "<bos>",
        "<system>",
        "instructions",
        "<user>",
        "answer",
        "one",
        "<assistant>",
        "answer",
        "one",
        "<eos>",
        "<tool>",
        "tool",
        "result",
        "<assistant>",
        "answer",
        "two",
        "<eos>",
        "<user>",
        "question",
    ]
    if add_generation_prompt:
        expected_words.append("<assistant>")
    expected_tokens = np.array(hf_tokenizer.convert_tokens_to_ids(expected_words))
    expected_targets = np.full_like(expected_tokens, sft_tokenizer.IGNORE_INDEX)
    # Supervision belongs to these assistant positions even when user text is identical.
    assistant_positions = [7, 8, 9, 14, 15, 16]
    expected_targets[assistant_positions] = expected_tokens[assistant_positions]
    np.testing.assert_array_equal(tokens, expected_tokens)
    np.testing.assert_array_equal(targets, expected_targets)
    supervised = targets[targets != sft_tokenizer.IGNORE_INDEX].tolist()
    assert hf_tokenizer.convert_ids_to_tokens(supervised) == [
        "answer",
        "one",
        "<eos>",
        "answer",
        "two",
        "<eos>",
    ]
    np.testing.assert_array_equal(tokens[targets != sft_tokenizer.IGNORE_INDEX], supervised)
    assert hf_tokenizer.convert_tokens_to_ids("<assistant>") not in supervised
    # The user repeats the same answer text: masking must depend on position, not text.
    assert (tokens == hf_tokenizer.convert_tokens_to_ids("one")).sum() == 2
    assert (targets == hf_tokenizer.convert_tokens_to_ids("one")).sum() == 1


@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_switch_does_not_change_inputs_or_inference(hf_tokenizer, add_generation_prompt):
    enabled = sft_tokenizer.CustomSFTTokenizer("unused", "default")
    disabled = sft_tokenizer.CustomSFTTokenizer("unused", "default", assistant_only_loss=False)
    core = core_sft_tokenizer.SFTTokenizer("unused", "default")
    tokens, _ = enabled.tokenize_conversation(CONVERSATION, True, add_generation_prompt)
    old_tokens, old_targets = disabled.tokenize_conversation(
        CONVERSATION, True, add_generation_prompt
    )
    np.testing.assert_array_equal(tokens, old_tokens)
    np.testing.assert_array_equal(old_tokens, old_targets)
    core_tokens, core_targets = core.tokenize_conversation(
        CONVERSATION, True, add_generation_prompt
    )
    np.testing.assert_array_equal(tokens, core_tokens)
    np.testing.assert_array_equal(old_targets, core_targets)
    np.testing.assert_array_equal(
        tokens, enabled.tokenize_conversation(CONVERSATION, False, add_generation_prompt)
    )


def test_missing_generation_spans_fail_only_when_masking(hf_tokenizer):
    hf_tokenizer.chat_template = CHAT_TEMPLATE.replace("{% generation %}", "").replace(
        "{% endgeneration %}", ""
    )
    enabled = sft_tokenizer.CustomSFTTokenizer("unused", "default")
    with pytest.raises(ValueError, match="requires.*generation"):
        enabled.tokenize_conversation(CONVERSATION, True, False)
    assert enabled.tokenize_conversation(CONVERSATION, False, False).size > 0
    disabled = sft_tokenizer.CustomSFTTokenizer("unused", "default", assistant_only_loss=False)
    tokens, targets = disabled.tokenize_conversation(CONVERSATION, True, False)
    np.testing.assert_array_equal(tokens, targets)


def test_empty_generation_mask_is_reported(hf_tokenizer):
    hf_tokenizer.chat_template = (
        "{% generation %}{% endgeneration %}"
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    tokenizer = sft_tokenizer.CustomSFTTokenizer("unused", "default")
    with pytest.raises(ValueError, match="no assistant tokens"):
        tokenizer.tokenize_conversation(CONVERSATION, True, False)


def test_no_assistant_turn_has_no_targets(hf_tokenizer):
    tokenizer = sft_tokenizer.CustomSFTTokenizer("unused", "default")
    _, targets = tokenizer.tokenize_conversation(CONVERSATION[:2], True, True)
    assert (targets == sft_tokenizer.IGNORE_INDEX).all()


@pytest.mark.parametrize("prompt_format", ["identity", "nemotron-nano-v2", "nemotron-h-aligned"])
@pytest.mark.parametrize("assistant_only_loss", [False, True])
def test_other_prompt_formats_are_unchanged(hf_tokenizer, prompt_format, assistant_only_loss):
    # These formats use their existing templates and masking rules, regardless of the flag.
    conversation = CONVERSATION[:3]
    tokenizer = sft_tokenizer.CustomSFTTokenizer(
        "unused", prompt_format, assistant_only_loss=assistant_only_loss
    )
    core = core_sft_tokenizer.SFTTokenizer("unused", prompt_format)
    for return_target in (False, True):
        actual = tokenizer.tokenize_conversation(conversation, return_target, False)
        expected = core.tokenize_conversation(conversation, return_target, False)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("option", [None, True, False])
def test_factory_passes_switch_and_defaults_on(hf_tokenizer, option):
    args = SimpleNamespace(
        tokenizer_type="CustomSFTTokenizer",
        tokenizer_model="unused",
        sft_tokenizer_prompt_format="default",
        metadata_path=None,
        pad_vocab_size=False,
        padded_vocab_size=len(hf_tokenizer),
    )
    if option is not None:
        args.sft_assistant_only_loss = option
    tokenizer = build_tokenizer(args)
    tokens, targets = tokenizer.tokenize_conversation(CONVERSATION, True, False)
    assert (targets == sft_tokenizer.IGNORE_INDEX).any() == (option is not False)
    if option is False:
        np.testing.assert_array_equal(tokens, targets)


@pytest.mark.parametrize(
    "argv,expected",
    [([], True), (["--sft-assistant-only-loss"], True), (["--no-sft-assistant-only-loss"], False)],
)
def test_training_argument_switch(argv, expected):
    parser = _add_sft_args(argparse.ArgumentParser())
    assert parser.parse_args(argv).sft_assistant_only_loss is expected


@pytest.fixture
def make_sft_dataset(hf_tokenizer):
    def make(messages, sequence_length, context_parallel_size=2):
        tokenizer = sft_tokenizer.CustomSFTTokenizer("unused", "default")
        dataset = SFTDataset.__new__(SFTDataset)
        dataset.config = SimpleNamespace(
            tokenizer=SimpleNamespace(
                tokenize_conversation=tokenizer.tokenize_conversation,
                eod=hf_tokenizer.eos_token_id,
                pad=hf_tokenizer.pad_token_id,
            ),
            sequence_length=sequence_length,
            reset_position_ids=False,
            context_parallel_size=context_parallel_size,
            preserve_packed_sequence_lengths=False,
            create_attention_mask=False,
            reset_attention_mask=False,
        )
        dataset.dataset = [messages]
        dataset.indices = np.array([0])
        return dataset

    return make


@pytest.mark.parametrize("sequence_length", [12, 128])
@pytest.mark.parametrize("pad_is_eos", [False, True])
def test_dataset_alignment_truncation_and_padding(
    hf_tokenizer, make_sft_dataset, sequence_length, pad_is_eos
):
    if pad_is_eos:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    # Two packed conversations also exercise the per-conversation CP padding path.
    messages = CONVERSATION[:3] + CONVERSATION[:3]
    dataset = make_sft_dataset(messages, sequence_length)
    sample = dataset[0]
    labels = sample["labels"].numpy()
    mask = sample["loss_mask"].numpy()
    np.testing.assert_array_equal(mask, labels != -100)
    expected = hf_tokenizer.convert_tokens_to_ids(["answer", "one", "<eos>"])
    assert labels[mask.astype(bool)].tolist() == expected * (1 if sequence_length == 12 else 2)
    # Every supervised label predicts the next assistant token, including the first answer token.
    for position in np.flatnonzero(mask):
        if position + 1 < sequence_length:
            assert labels[position] == sample["tokens"][position + 1].item()
    assert sample["tokens"].numel() == sequence_length


@pytest.mark.parametrize("pad_is_eos", [False, True])
@pytest.mark.parametrize(
    "sequence_length,supervised_words",
    [(7, []), (8, ["answer"]), (9, ["answer", "one"]), (10, ["answer", "one", "<eos>"])],
)
def test_dataset_truncation_at_assistant_and_eos_boundaries(
    hf_tokenizer, make_sft_dataset, pad_is_eos, sequence_length, supervised_words
):
    if pad_is_eos:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    dataset = make_sft_dataset(CONVERSATION[:3], sequence_length, context_parallel_size=1)
    sample = dataset[0]
    expected_tokens = hf_tokenizer.convert_tokens_to_ids(
        [
            "<bos>",
            "<system>",
            "instructions",
            "<user>",
            "answer",
            "one",
            "<assistant>",
            "answer",
            "one",
            "<eos>",
        ]
    )[:sequence_length]
    expected_labels = np.full(sequence_length, sft_tokenizer.IGNORE_INDEX)
    # After the one-token shift, the assistant header predicts the first answer token.
    expected_labels[6 : 6 + len(supervised_words)] = hf_tokenizer.convert_tokens_to_ids(
        supervised_words
    )
    np.testing.assert_array_equal(sample["tokens"].numpy(), expected_tokens)
    np.testing.assert_array_equal(sample["labels"].numpy(), expected_labels)
    np.testing.assert_array_equal(
        sample["loss_mask"].numpy(), expected_labels != sft_tokenizer.IGNORE_INDEX
    )
    # Truncation must never invent supervision for a dropped token or padding.
    assert sample["labels"][-1].item() == sft_tokenizer.IGNORE_INDEX


@pytest.mark.parametrize("option,expected", [(None, True), (True, True), (False, False)])
def test_yaml_switch_reaches_argument_parser(option, expected):
    from omegaconf import OmegaConf

    from flagscale.runner.runner_train import _get_args_megatron

    data = {"tokenizer_type": "CustomSFTTokenizer"}
    if option is not None:
        data["sft_assistant_only_loss"] = option
    config = OmegaConf.create(
        {
            "experiment": {"task": {"backend": "megatron"}},
            "train": {"system": {}, "model": {}, "data": data},
        }
    )
    argv = _get_args_megatron(config)
    parser = _add_tokenizer_args(_add_sft_args(argparse.ArgumentParser()))
    args = parser.parse_args(argv)
    assert args.tokenizer_type == "CustomSFTTokenizer"
    assert args.sft_assistant_only_loss is expected
    assert OmegaConf.to_container(config.train.data) == data


def test_factory_uses_local_subclass_and_core_wrapper(hf_tokenizer):
    from unittest.mock import patch

    from megatron.training.tokenizer import tokenizer as factory

    args = SimpleNamespace(
        tokenizer_type="CustomSFTTokenizer",
        tokenizer_model="unused",
        sft_tokenizer_prompt_format="default",
        metadata_path=None,
        padded_vocab_size=len(hf_tokenizer),
    )
    with patch.object(factory, "_upstream_build_tokenizer", side_effect=AssertionError):
        tokenizer = build_tokenizer(args)
    assert isinstance(tokenizer._tokenizer, sft_tokenizer.CustomSFTTokenizer)
    assert isinstance(tokenizer._tokenizer, core_sft_tokenizer.SFTTokenizer)
    assert tokenizer.pad == hf_tokenizer.pad_token_id
    assert tokenizer.eod == hf_tokenizer.eos_token_id
    assert tokenizer.vocab_size == len(hf_tokenizer)
    assert tokenizer.unique_identifiers["assistant_only_loss"] == "True"


def test_core_constructor_does_not_need_new_argument(hf_tokenizer, monkeypatch):
    original = core_sft_tokenizer.SFTTokenizer.__init__

    def legacy_init(self, tokenizer_path, prompt_format):
        original(self, tokenizer_path, prompt_format)

    monkeypatch.setattr(core_sft_tokenizer.SFTTokenizer, "__init__", legacy_init)
    tokenizer = sft_tokenizer.CustomSFTTokenizer("unused", "default")
    _, targets = tokenizer.tokenize_conversation(CONVERSATION, True, False)
    assert (targets == sft_tokenizer.IGNORE_INDEX).any()


@pytest.mark.parametrize("bad_mask", ["missing", "shape", "binary"])
def test_missing_or_invalid_assistant_mask_is_reported(hf_tokenizer, monkeypatch, bad_mask):
    tokenizer = sft_tokenizer.CustomSFTTokenizer("unused", "default")
    encoded = hf_tokenizer.apply_chat_template(
        CONVERSATION,
        return_dict=True,
        return_tensors="np",
        return_assistant_tokens_mask=True,
    )
    if bad_mask == "missing":
        del encoded["assistant_masks"]
    elif bad_mask == "shape":
        encoded["assistant_masks"] = [[0]]
    else:
        encoded["assistant_masks"] = np.full_like(encoded["assistant_masks"], 2)
    monkeypatch.setattr(hf_tokenizer, "apply_chat_template", lambda *a, **kw: encoded)
    with pytest.raises(ValueError, match="mask"):
        tokenizer.tokenize_conversation(CONVERSATION, True, False)


def test_standard_sft_tokenizer_still_delegates_to_core(hf_tokenizer):
    from unittest.mock import patch

    from megatron.training.tokenizer import tokenizer as factory

    args = SimpleNamespace(tokenizer_type="SFTTokenizer", padded_vocab_size=len(hf_tokenizer))
    with patch.object(factory, "_upstream_build_tokenizer", return_value=hf_tokenizer) as build:
        assert build_tokenizer(args) is hf_tokenizer
    build.assert_called_once_with(args)


@pytest.mark.parametrize("hf_template", [None, CHAT_TEMPLATE])
def test_custom_template_has_priority(hf_tokenizer, hf_template):
    hf_tokenizer.chat_template = hf_template
    template = CHAT_TEMPLATE.replace("{{ bos_token }}", "{{ bos_token + 'question' }}")
    tokenizer = sft_tokenizer.CustomSFTTokenizer("unused", "default", custom_chat_template=template)
    tokens, targets = tokenizer.tokenize_conversation(CONVERSATION, True, False)
    expected = hf_tokenizer.apply_chat_template(
        CONVERSATION, chat_template=template, return_tensors="np"
    )[0]
    np.testing.assert_array_equal(tokens, expected)
    assert (targets == sft_tokenizer.IGNORE_INDEX).any()
    np.testing.assert_array_equal(
        tokenizer.tokenize_conversation(CONVERSATION, False, False), tokens
    )


def test_hf_template_dictionary_uses_default(hf_tokenizer):
    hf_tokenizer.chat_template = {"default": CHAT_TEMPLATE, "other": "unused"}
    tokenizer = sft_tokenizer.CustomSFTTokenizer("unused", "default")
    _, targets = tokenizer.tokenize_conversation(CONVERSATION, True, False)
    assert (targets == sft_tokenizer.IGNORE_INDEX).any()


def test_factory_passes_custom_template_from_yaml(hf_tokenizer):
    from omegaconf import OmegaConf

    from flagscale.runner.runner_train import _get_args_megatron

    template = CHAT_TEMPLATE.replace("{{ bos_token }}", "{{ bos_token + 'question' }}")
    config = OmegaConf.create(
        {
            "experiment": {"task": {"backend": "megatron"}},
            "train": {
                "system": {},
                "model": {},
                "data": {
                    "tokenizer_type": "CustomSFTTokenizer",
                    "tokenizer_model": "unused",
                    "sft_tokenizer_prompt_format": "default",
                    "custom_chat_template": template,
                },
            },
        }
    )
    parser = _add_tokenizer_args(_add_sft_args(argparse.ArgumentParser()))
    args = parser.parse_args(_get_args_megatron(config))
    args.padded_vocab_size = len(hf_tokenizer)
    tokenizer = build_tokenizer(args)
    tokens, _ = tokenizer.tokenize_conversation(CONVERSATION, True, False)
    expected = hf_tokenizer.apply_chat_template(
        CONVERSATION, chat_template=template, return_tensors="np"
    )[0]
    np.testing.assert_array_equal(tokens, expected)
    assert tokenizer.chat_template == template


@pytest.mark.parametrize("hf_template", [None, "", {}])
@pytest.mark.parametrize("assistant_only_loss", [True, False])
def test_missing_templates_fail_during_initialization(
    hf_tokenizer, hf_template, assistant_only_loss
):
    hf_tokenizer.chat_template = hf_template
    with pytest.raises(ValueError, match="CustomSFTTokenizer.*custom_chat_template"):
        sft_tokenizer.CustomSFTTokenizer(
            "unused", "default", assistant_only_loss=assistant_only_loss
        )


def test_factory_reports_missing_templates(hf_tokenizer):
    hf_tokenizer.chat_template = None
    args = SimpleNamespace(
        tokenizer_type="CustomSFTTokenizer",
        tokenizer_model="unused",
        sft_tokenizer_prompt_format="default",
        metadata_path=None,
        padded_vocab_size=len(hf_tokenizer),
    )
    with pytest.raises(ValueError, match="CustomSFTTokenizer.*custom_chat_template"):
        build_tokenizer(args)


@pytest.mark.parametrize("metadata_source", ["dict", "json"])
def test_factory_loads_sft_metadata_without_mutating_it(hf_tokenizer, tmp_path, metadata_source):
    metadata = {"library": "sft", "metadata_marker": "preserved"}
    metadata_path = tmp_path / "tokenizer_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    args = SimpleNamespace(
        tokenizer_type="CustomSFTTokenizer",
        tokenizer_model="unused",
        sft_tokenizer_prompt_format="default",
        metadata_path=metadata if metadata_source == "dict" else str(metadata_path),
        padded_vocab_size=len(hf_tokenizer),
    )
    tokenizer = build_tokenizer(args)
    assert isinstance(tokenizer._tokenizer, sft_tokenizer.CustomSFTTokenizer)
    assert tokenizer.library == "sft"
    assert tokenizer.metadata_marker == "preserved"
    _, targets = tokenizer.tokenize_conversation(CONVERSATION, True, False)
    assert hf_tokenizer.convert_ids_to_tokens(targets[targets != -100].tolist()) == [
        "answer",
        "one",
        "<eos>",
        "answer",
        "two",
        "<eos>",
    ]
    assert metadata == {"library": "sft", "metadata_marker": "preserved"}
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata


@pytest.mark.parametrize("metadata_source", ["dict", "json"])
@pytest.mark.parametrize("library", [None, "huggingface"])
def test_factory_rejects_non_sft_metadata(hf_tokenizer, tmp_path, metadata_source, library):
    metadata = {} if library is None else {"library": library}
    # Keep missing-library metadata nonempty so it cannot select the default config.
    metadata["metadata_marker"] = "invalid"
    metadata_path = tmp_path / "tokenizer_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    args = SimpleNamespace(
        tokenizer_type="CustomSFTTokenizer",
        tokenizer_model="unused",
        sft_tokenizer_prompt_format="default",
        metadata_path=metadata if metadata_source == "dict" else str(metadata_path),
        padded_vocab_size=len(hf_tokenizer),
    )
    with pytest.raises(ValueError, match="metadata must use the 'sft' library"):
        build_tokenizer(args)


@pytest.mark.parametrize("use_kwargs_template", [False, True])
def test_factory_template_kwargs_override_args(hf_tokenizer, use_kwargs_template):
    args_template = CHAT_TEMPLATE.replace("{{ bos_token }}", "{{ bos_token + 'question' }}")
    kwargs_template = CHAT_TEMPLATE.replace("{{ bos_token }}", "{{ bos_token + 'instructions' }}")
    args = SimpleNamespace(
        tokenizer_type="CustomSFTTokenizer",
        tokenizer_model="unused",
        sft_tokenizer_prompt_format="default",
        custom_chat_template=args_template,
        padded_vocab_size=len(hf_tokenizer),
    )
    tokenizer = build_tokenizer(
        args, custom_chat_template=kwargs_template if use_kwargs_template else ""
    )
    expected_template = kwargs_template if use_kwargs_template else args_template
    assert tokenizer.chat_template == expected_template
    tokens, _ = tokenizer.tokenize_conversation(CONVERSATION, True, False)
    expected = hf_tokenizer.apply_chat_template(
        CONVERSATION, chat_template=expected_template, return_tensors="np"
    )[0]
    np.testing.assert_array_equal(tokens, expected)
    assert tokenizer.unique_identifiers["custom_chat_template"] == expected_template

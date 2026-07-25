# Copyright (c) 2025, FlagScale Team. All rights reserved.

"""FlagScale-specific tokenizer types and build_tokenizer wrapper.

Standard tokenizer types are delegated to the upstream factory at
megatron.core.tokenizers.utils.build_tokenizer. This module provides:
1. A registration system for FlagScale-specific tokenizer factories
2. Each registered tokenizer owns its special token definitions internally
"""

from collections import OrderedDict

from megatron.core.tokenizers.base_tokenizer import MegatronTokenizerBase
from megatron.core.tokenizers.utils.build_tokenizer import (
    build_tokenizer as _upstream_build_tokenizer,
    vocab_size_with_padding,
)

from .gpt2_tokenization import AquilaTokenizer
from .rwkv_tokenization import RWKVTokenizer


# ---------------------------------------------------------------------------
# Tokenizer factory registration system
# ---------------------------------------------------------------------------

_TOKENIZER_FACTORY_REGISTRY = {}


def register_tokenizer_factory(tokenizer_type, factory_fn):
    """Register a tokenizer factory function.

    Args:
        tokenizer_type: String name of the tokenizer type
        factory_fn: Callable that takes (args, **kwargs) and returns a tokenizer instance
    """
    _TOKENIZER_FACTORY_REGISTRY[tokenizer_type] = factory_fn


def build_tokenizer(args, **kwargs):
    """Initialize tokenizer.

    Checks the FlagScale registry first; falls through to upstream if not found.
    """
    from megatron.training.utils import print_rank_0

    if args.tokenizer_type in _TOKENIZER_FACTORY_REGISTRY:
        print_rank_0(f"> building {args.tokenizer_type} tokenizer ...")
        tokenizer = _TOKENIZER_FACTORY_REGISTRY[args.tokenizer_type](args, **kwargs)
    else:
        tokenizer = _upstream_build_tokenizer(args, **kwargs)

    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


# ---------------------------------------------------------------------------
# FlagScale-specific tokenizer classes
# ---------------------------------------------------------------------------


class _FlagScaleTokenizerBase(MegatronTokenizerBase):
    """Convenience base for FlagScale tokenizers."""

    def __init__(self, path, config=None, **kwargs):
        if config is None:
            config = {}
        super().__init__(path=path, config=config, **kwargs)

    @property
    def unique_identifiers(self) -> OrderedDict:
        uid = OrderedDict()
        uid["class"] = f"{type(self).__module__}.{type(self).__qualname__}"
        uid["tokenizer_path"] = self.path
        return uid

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError("This tokenizer does not support chat templates.")


class _AquilaTokenizerFS(_FlagScaleTokenizerBase):
    """Aquila tokenizer using GPT2 BPE with custom special tokens."""

    def __init__(self, vocab_file, merge_file, special_tokens_file):
        super().__init__(path=vocab_file)
        special_tokens = []
        if special_tokens_file:
            special_tokens = open(special_tokens_file, encoding="utf-8").read().split("\n")[:-1]
        self.tokenizer = AquilaTokenizer(
            vocab_file, merge_file, errors="replace", special_tokens=special_tokens, max_len=None
        )
        self.eod_id = self.tokenizer.encoder["</s>"]
        self.cls_id = self.tokenizer.encoder["[CLS]"]
        self.pad_id = self.tokenizer.encoder["<|endoftext|>"]

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def cls(self):
        return self.cls_id

    @property
    def pad(self):
        return self.pad_id


class _HFTokenizerFS(_FlagScaleTokenizerBase):
    """HuggingFace AutoTokenizer wrapper."""

    def __init__(self, tokenizer_path):
        super().__init__(path=tokenizer_path)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.eod_id = self.tokenizer.eos_token_id
        self.cls_id = self.tokenizer.bos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self._inv_vocab = None

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in self.vocab.items()}
        return self._inv_vocab

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def cls(self):
        return self.cls_id

    @property
    def pad(self):
        return self.pad_id


class _Llama3TokenizerFS(_HFTokenizerFS):
    """Llama3 tokenizer with added vocab."""

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + len(self.tokenizer.get_added_vocab())


class _QwenTokenizerFS(_HFTokenizerFS):
    """Qwen tokenizer with custom special tokens."""

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.eod_id = self.tokenizer.encode("<|extra_204|>")[0]
        self.cls_id = self.tokenizer.encode("<|extra_203|>")[0]
        self.pad_id = self.tokenizer.encode("<|endoftext|>")[0]


class _HFTokenizersTokenizerFS(_FlagScaleTokenizerBase):
    """Tokenizer using HuggingFace tokenizers library (not transformers)."""

    def __init__(self, json_file):
        super().__init__(path=json_file)
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(json_file)
        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")
        self.eod_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_id = self.tokenizer.token_to_id("<|padding|>")
        self._inv_vocab = None

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in self.vocab.items()}
        return self._inv_vocab

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def pad(self):
        return self.pad_id


class _Qwen2TokenizerFS(_HFTokenizerFS):
    """Qwen2 tokenizer with explicit vocab_size from args."""

    def __init__(self, tokenizer_path, args):
        super().__init__(tokenizer_path)
        self.eod_id = self.tokenizer.encode("<|extra_204|>")[0]
        self.cls_id = self.tokenizer.encode("<|extra_203|>")[0]
        self.pad_id = self.tokenizer.encode("<|endoftext|>")[0]
        assert args.vocab_size is not None
        self._vocab_size = args.vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size


class _Qwen2VLTokenizer(_FlagScaleTokenizerBase):
    def __init__(self, tokenizer_path, extra_vocab_size):
        super().__init__(tokenizer_path)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            use_fast=True,
            split_special_tokens=False,
            trust_remote_code=True,
            revision="main",
            token=None,
        )
        self.extra_vocab_size = extra_vocab_size
        self.special_tokens_map = {
            k: v for k, v in zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)
        }
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            revision="main",
            token=None,
        )
        # NOTE: In Qwen2-VL, template in chat_template.json is same within tokenizer_config.json and both can be used.
        # However, in Qwen 2.5-VL, the two templates are different and only the one in chat_template.json is OK.
        self.chat_template = self.processor.chat_template

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=None,
        max_length=None,
        truncation=None,
        add_special_tokens=None,
    ):
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )

    def apply_chat_template(self, conversations, tokenize: bool = True, **kwargs):
        return self.tokenizer.apply_chat_template(
            conversations, tokenize=tokenize, chat_template=self.chat_template, **kwargs
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + self.extra_vocab_size

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def image_token_id(self):
        return self.special_tokens_map[self.image_token]

    @property
    def video_token_id(self):
        return self.special_tokens_map[self.video_token]

    @property
    def vision_start_token_id(self):
        return self.special_tokens_map[self.vision_start_token]

    @property
    def vision_end_token_id(self):
        return self.special_tokens_map[self.vision_end_token]

    def encode(self, x):
        return self.tokenizer.encode(x)


class _RWKVTokenizerFS(_FlagScaleTokenizerBase):
    """RWKV Trie-based tokenizer."""

    def __init__(self, tokenizer_path):
        super().__init__(path=tokenizer_path)
        self._rwkv = RWKVTokenizer(tokenizer_path)
        self.eod = self._rwkv.eod

    @property
    def vocab_size(self):
        return self._rwkv.vocab_size

    @property
    def vocab(self):
        return self._rwkv.token2idx

    @property
    def inv_vocab(self):
        return self._rwkv.idx2token

    def tokenize(self, text):
        return self._rwkv.encode(text)

    def detokenize(self, token_ids):
        return self._rwkv.decode(token_ids)

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError("RWKVTokenizer does not support chat templates.")


# ---------------------------------------------------------------------------
# Factory functions and registration
# ---------------------------------------------------------------------------


def _build_aquila(args, **kwargs):
    assert args.vocab_file and args.merge_file and args.special_tokens_file
    return _AquilaTokenizerFS(args.vocab_file, args.merge_file, args.special_tokens_file)


def _build_hf(args, **kwargs):
    assert args.tokenizer_path
    return _HFTokenizerFS(args.tokenizer_path)


def _build_llama3(args, **kwargs):
    assert args.tokenizer_path
    return _Llama3TokenizerFS(args.tokenizer_path)


def _build_qwen(args, **kwargs):
    assert args.tokenizer_path
    return _QwenTokenizerFS(args.tokenizer_path)


def _build_hftokenizers(args, **kwargs):
    assert args.tokenizer_path
    return _HFTokenizersTokenizerFS(args.tokenizer_path)


def _build_qwen2(args, **kwargs):
    assert args.tokenizer_path
    return _Qwen2TokenizerFS(args.tokenizer_path, args)


def _build_qwen2vl(args, **kwargs):
    assert args.tokenizer_path
    tok = _Qwen2VLTokenizer(args.tokenizer_path, args.extra_vocab_size)
    args.padded_vocab_size = tok.vocab_size
    return tok


def _build_rwkv(args, **kwargs):
    assert args.tokenizer_path
    return _RWKVTokenizerFS(args.tokenizer_path)


register_tokenizer_factory("AquilaTokenizerFS", _build_aquila)
register_tokenizer_factory("HFTokenizerFS", _build_hf)
register_tokenizer_factory("Llama3TokenizerFS", _build_llama3)
register_tokenizer_factory("QwenTokenizerFS", _build_qwen)
register_tokenizer_factory("HFTokenizersTokenizerFS", _build_hftokenizers)
register_tokenizer_factory("Qwen2TokenizerFS", _build_qwen2)
register_tokenizer_factory("Qwen2VLTokenizer", _build_qwen2vl)
register_tokenizer_factory("RWKVTokenizer", _build_rwkv)

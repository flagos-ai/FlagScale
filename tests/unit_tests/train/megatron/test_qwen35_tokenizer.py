"""VL checkpoint vocab padding must survive tokenizer construction."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from megatron.training.tokenizer import tokenizer


@pytest.mark.parametrize("model_vocab,expected", [(248320, 248320), (None, 248064)])
def test_explicit_model_vocab_is_preserved(model_vocab, expected):
    tok = MagicMock(vocab_size=248064)
    tok.tokenizer.__len__.return_value = 248064
    args = SimpleNamespace(tokenizer_path="unused", extra_vocab_size=0, vocab_size=model_vocab)
    with patch.object(tokenizer, "_Qwen2VLTokenizer", return_value=tok):
        assert tokenizer._build_qwen2vl(args) is tok
    assert args.padded_vocab_size == expected


def test_model_vocab_cannot_drop_special_tokens():
    tok = MagicMock(vocab_size=248044)
    tok.tokenizer.__len__.return_value = 248064
    args = SimpleNamespace(tokenizer_path="unused", extra_vocab_size=0, vocab_size=248044)
    with (
        patch.object(tokenizer, "_Qwen2VLTokenizer", return_value=tok),
        pytest.raises(ValueError, match="special tokens"),
    ):
        tokenizer._build_qwen2vl(args)


def test_chatml_path_decoder_does_not_initialize_av():
    import pickle

    from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory
    from tools.datasets.qwenvl.data.energon.chatml import ChatMLWebdataset

    def initialize(instance, path, **kwargs):
        assert kwargs["auto_decode"] is False
        assert "av_decode" not in kwargs
        instance.image_decode = kwargs["image_decode"]

    with patch.object(DefaultDecoderWebdatasetFactory, "__init__", initialize):
        dataset = ChatMLWebdataset("unused")
    sample = dataset._decoder(
        {
            "__key__": "example",
            "jpgs": pickle.dumps(["image.jpg"]),
            "videos": pickle.dumps([]),
            "json": b'{"conversations": []}',
        }
    )
    assert sample["jpgs"] == ["image.jpg"]
    assert sample["videos"] == []
    assert sample["json"] == {"conversations": []}


def test_qwen35_mtp_mrope_arguments(monkeypatch):
    import sys
    from pathlib import Path

    from omegaconf import OmegaConf

    from flagscale.runner.runner_train import _get_args_megatron

    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "flagscale/train/megatron"))
    from megatron.training.arguments import parse_args, validate_args

    from flagscale.train.megatron.train_qwen35 import add_qwen35_extra_args

    cfg = OmegaConf.create({"experiment": {"task": {"backend": "megatron"}}})
    cfg.train = OmegaConf.load(root / "examples/qwen35/conf/train/4b.yaml")
    cfg.train.system.tensor_model_parallel_size = 2
    cfg.train.system.pipeline_model_parallel_size = 1
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train_qwen35.py", *_get_args_megatron(cfg)])
    args = parse_args(extra_args_provider=add_qwen35_extra_args)
    validate_args(args)
    assert args.mtp_num_layers == 1
    assert args.position_embedding_type == "mrope"

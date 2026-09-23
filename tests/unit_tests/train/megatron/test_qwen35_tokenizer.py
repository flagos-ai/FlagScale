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


@pytest.mark.parametrize("device_arch", [None, 9, 10])
def test_qwen35_mtp_mrope_arguments(monkeypatch, device_arch):
    import sys
    from pathlib import Path

    from omegaconf import OmegaConf

    from flagscale.runner.runner_train import _get_args_megatron

    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "flagscale/train/megatron"))
    import megatron.training.arguments as training_arguments
    from megatron.training.arguments import parse_args, validate_args

    from flagscale.train.megatron.train_qwen35 import add_qwen35_extra_args

    cfg = OmegaConf.create({"experiment": {"task": {"backend": "megatron"}}})
    cfg.train = OmegaConf.load(root / "examples/qwen35/conf/train/4b.yaml")
    cfg.train.system.tensor_model_parallel_size = 2
    cfg.train.system.pipeline_model_parallel_size = 1
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(training_arguments, "get_device_arch_version", lambda: device_arch)
    monkeypatch.setattr(sys, "argv", ["train_qwen35.py", *_get_args_megatron(cfg)])
    args = parse_args(extra_args_provider=add_qwen35_extra_args)
    validate_args(args)
    assert args.mtp_num_layers == 1
    assert args.position_embedding_type == "mrope"


@pytest.mark.parametrize("enable_hetero", [False, True])
def test_hetero_mesh_validation_precedes_common_validation(monkeypatch, enable_hetero):
    import megatron.backend_config as backend_config
    import megatron.training.arguments as training_arguments
    from megatron.training.arguments_fs import FSTrainArguments

    args = SimpleNamespace(enable_hetero=enable_hetero, use_checkpoint_args=False, yaml_cfg=None)
    events = []

    def pre_validate(instance):
        assert instance.args is args
        assert not hasattr(args, "data_parallel_size")
        args.data_parallel_size = 2
        events.append("pre")

    def validate(actual, defaults):
        assert actual is args
        if enable_hetero:
            assert args.data_parallel_size == 2
        events.append("validate")

    monkeypatch.setattr(training_arguments, "parse_args", lambda *a: args)
    monkeypatch.setattr(training_arguments, "validate_args", validate)
    monkeypatch.setattr(
        training_arguments, "set_global_variables", lambda a: events.append("globals")
    )
    monkeypatch.setattr(
        backend_config, "configure_backend_environment", lambda a: events.append("backend")
    )
    monkeypatch.setattr(FSTrainArguments, "pre_validate_args", pre_validate)
    monkeypatch.setattr(FSTrainArguments, "post_validate_args", lambda a: events.append("post"))

    assert training_arguments.parse_and_validate_args() is args
    expected = ["backend", "pre", "validate", "post", "globals"]
    assert events == (expected if enable_hetero else ["validate", "globals"])

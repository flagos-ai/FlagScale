import importlib
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from flagscale.compress.algo.algo_base import BaseALGO
from flagscale.compress.combined_algo import prepare_compress_methods
from flagscale.patches_utils import add_patches_module
from flagscale.utils import flatten_dict_to_args


def test_flatten_dict_to_args_handles_nested_lists_bools_and_ignore_keys():
    config = {
        "model_path": "Qwen",
        "skip_me": "ignored",
        "nested": {"use_cache": True, "disabled": False, "layers": [1, "2"]},
    }

    assert flatten_dict_to_args(config, ignore_keys=["skip_me"]) == [
        "--model-path",
        "Qwen",
        "--use-cache",
        "--layers",
        "1",
        "2",
    ]


def test_flatten_dict_to_args_keeps_empty_values_and_default_ignore_list_is_safe():
    assert flatten_dict_to_args({"empty": [], "flag": False, "value": 0}) == [
        "--empty",
        "--value",
        "0",
    ]
    assert flatten_dict_to_args({"again": "ok"}) == ["--again", "ok"]


def test_add_patches_module_replaces_existing_module_attributes(monkeypatch):
    module = types.ModuleType("fakepkg.target")
    module.target_func = lambda: "old"
    module.missing = None
    monkeypatch.setitem(sys.modules, "fakepkg.target", module)

    replacement = lambda: "new"
    add_patches_module("fakepkg", {"target_func": replacement, "missing": object()})

    assert module.target_func is replacement
    assert module.missing is None


def test_add_patches_module_rejects_empty_module_dict():
    with pytest.raises(Exception, match="module dict is None"):
        add_patches_module("fakepkg", {})


def test_prepare_compress_methods_returns_deepcopy():
    cfg = {"method": [{"targets": ["linear"]}]}

    copied = prepare_compress_methods(cfg)
    copied["method"][0]["targets"].append("mlp")

    assert cfg == {"method": [{"targets": ["linear"]}]}
    assert copied is not cfg


def test_base_algo_initial_state_and_abstract_methods():
    algo = BaseALGO("gptq")

    assert algo.name == "gptq"
    assert algo._observer is False
    assert algo._compress is False
    with pytest.raises(NotImplementedError):
        algo.preprocess_weight()
    with pytest.raises(NotImplementedError):
        algo.add_batch()


def test_inference_parse_config_loads_yaml_and_validates_required_sections(tmp_path):
    from flagscale.inference import arguments

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "llm:\n  model: qwen\ngenerate:\n  max_new_tokens: 8\n", encoding="utf-8"
    )

    with patch("sys.argv", ["prog", "--config-path", str(config_file)]):
        config = arguments.parse_config()

    assert config.llm.model == "qwen"
    assert config.generate.max_new_tokens == 8

    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("llm:\n  model: qwen\n", encoding="utf-8")
    with (
        patch("sys.argv", ["prog", "--config-path", str(bad_file)]),
        pytest.raises(AssertionError),
    ):
        arguments.parse_config()


def test_serve_parse_config_reads_yaml_and_ignores_log_dir_argument(tmp_path):
    from flagscale.serve import arguments

    config_file = tmp_path / "serve.yaml"
    config_file.write_text("model:\n  path: qwen\n", encoding="utf-8")

    with patch("sys.argv", ["prog", "--config-path", str(config_file), "--log-dir", "logs"]):
        config = arguments.parse_config()

    assert config.model.path == "qwen"


def test_serve_load_args_runs_only_once_and_updates_task_config(tmp_path, monkeypatch):
    import flagscale.serve.utils as serve_utils

    serve_utils = importlib.reload(serve_utils)
    config_file = tmp_path / "serve.yaml"
    config_file.write_text("model:\n  path: qwen\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--config-path", str(config_file), "--log-dir", "logs1"],
    )

    serve_utils.load_args()
    serve_utils.task_config.model.path = "changed"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--config-path", str(config_file), "--log-dir", "logs2"],
    )
    serve_utils.load_args()

    assert serve_utils.task_config.model.path == "changed"
    assert serve_utils.task_config.log_dir == "logs1"


def test_serve_check_and_get_port_returns_requested_or_free_port(monkeypatch):
    import flagscale.serve.dag_utils as dag_utils

    class FakeSocket:
        next_fail = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def bind(self, address):
            self.last_bind = address
            if FakeSocket.next_fail and address[1] != 0:
                FakeSocket.next_fail = False
                raise OSError("occupied")

        def getsockname(self):
            return ("0.0.0.0", 45678)

    monkeypatch.setattr(dag_utils.socket, "socket", lambda *args, **kwargs: FakeSocket())

    assert dag_utils.check_and_get_port(12345) == 12345
    FakeSocket.next_fail = True
    assert dag_utils.check_and_get_port(None) == 45678


def test_serve_auto_remote_wraps_actor_methods(monkeypatch):
    fake_ray = types.ModuleType("ray")
    calls = []

    class RemoteMethod:
        def __init__(self, value):
            self.value = value

        def remote(self, *args, **kwargs):
            return (self.value, args, kwargs)

    class Actor:
        generate = RemoteMethod("generate")
        status = RemoteMethod("status")

    class RemoteClass:
        @staticmethod
        def remote(*args, **kwargs):
            calls.append((args, kwargs))
            return Actor()

    def remote(**resources):
        calls.append(resources)

        def decorator(cls):
            calls.append(cls)
            return RemoteClass

        return decorator

    fake_ray.remote = remote
    fake_ray.get = lambda value: ("got", value)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    core = importlib.reload(importlib.import_module("flagscale.serve.core"))

    @core.auto_remote(gpu=2, cpu=3, custom={"accelerator": 1})
    class Worker:
        pass

    worker = Worker("model", dtype="bf16")

    assert worker.generate("hello") == ("got", ("generate", ("hello",), {}))
    assert worker.status.remote() == ("status", (), {})
    assert calls[0] == {"num_gpus": 2, "num_cpus": 3, "resources": {"accelerator": 1}}
    assert calls[2] == (("model",), {"dtype": "bf16"})


def test_serve_run_serve_main_delegates_to_engine(monkeypatch):
    fake_arguments = types.ModuleType("flagscale.serve.arguments")
    fake_engine = types.ModuleType("flagscale.serve.engine")
    config = OmegaConf.create({"model": "qwen"})
    engine_instance = MagicMock()
    fake_arguments.parse_config = MagicMock(return_value=config)
    fake_engine.ServeEngine = MagicMock(return_value=engine_instance)
    monkeypatch.setitem(sys.modules, "flagscale.serve.arguments", fake_arguments)
    monkeypatch.setitem(sys.modules, "flagscale.serve.engine", fake_engine)

    run_serve = importlib.reload(importlib.import_module("flagscale.serve.run_serve"))
    run_serve.main()

    fake_arguments.parse_config.assert_called_once()
    fake_engine.ServeEngine.assert_called_once_with(config)
    engine_instance.run_task.assert_called_once()


def test_inference_parse_torch_dtype_supports_aliases_and_invalid_values(monkeypatch):
    fake_torch = types.ModuleType("torch")

    class FakeDType:
        pass

    fake_torch.dtype = FakeDType
    fake_torch.bfloat16 = FakeDType()
    fake_torch.float16 = FakeDType()
    fake_torch.float32 = FakeDType()
    fake_torch.float64 = FakeDType()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    inference_utils = importlib.reload(importlib.import_module("flagscale.inference.utils"))

    assert inference_utils.parse_torch_dtype(None) is None
    assert inference_utils.parse_torch_dtype(fake_torch.float16) is fake_torch.float16
    assert inference_utils.parse_torch_dtype(" bf16 ") is fake_torch.bfloat16
    assert inference_utils.parse_torch_dtype("torch.float16") is fake_torch.float16
    assert inference_utils.parse_torch_dtype("half") is fake_torch.float16
    assert inference_utils.parse_torch_dtype("FP32") is fake_torch.float32
    assert inference_utils.parse_torch_dtype("fp64") is fake_torch.float64
    assert inference_utils.parse_torch_dtype("unknown") is None
    assert inference_utils.parse_torch_dtype(123) is None


def test_logger_replaces_handlers_and_delegates_levels(monkeypatch):
    import flagscale.logger as logger_module

    logger_module.GLOBAL_LOGGER = None
    first = logger_module.get_logger()
    second = logger_module.get_logger()

    assert first is second
    assert first.logger.name == "FlagScale"
    assert len(first.logger.handlers) == 1
    assert first.logger.propagate is False

    custom = logger_module.Logger("unit-logger", level=logging.DEBUG)
    custom.logger.info = MagicMock()
    custom.logger.warning = MagicMock()
    custom.logger.error = MagicMock()
    custom.logger.critical = MagicMock()
    custom.logger.debug = MagicMock()

    custom.info("info")
    custom.warning("warning")
    custom.error("error")
    custom.critical("critical")
    custom.debug("debug")

    custom.logger.info.assert_called_once_with("info", stacklevel=2)
    custom.logger.warning.assert_called_once_with("warning", stacklevel=2)
    custom.logger.error.assert_called_once_with("error", stacklevel=2)
    custom.logger.critical.assert_called_once_with("critical", stacklevel=2)
    custom.logger.debug.assert_called_once_with("debug", stacklevel=2)


def test_args_mapping_helpers_converter_and_errors(tmp_path, monkeypatch):
    from flagscale.serve.args_mapping import mapping

    assert mapping.args2func("backend", "max_len") == "backend_max_len_converter"
    assert mapping.func2args("backend", "backend_max_len_converter") == "max_len"
    with pytest.raises(ValueError, match="Backend name"):
        mapping.func2args("vllm", "backend_max_len_converter")
    with pytest.raises(ValueError, match="does not end"):
        mapping.func2args("backend", "backend_max_len")

    converter = mapping.ArgsConverter()
    assert converter.convert(
        "sglang",
        {"model": "qwen", "max_num_seqs": 4, "unchanged": True},
    ) == {"model_path": "qwen", "max_running_requests": 4, "unchanged": True}
    with pytest.raises(ValueError, match="not found"):
        converter.convert("missing", {})
    with pytest.raises(ValueError, match="not found"):
        converter.load_funcs("missing")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    gguf_file = model_dir / "model.gguf"
    gguf_file.write_text("fake", encoding="utf-8")
    converted = converter.convert(
        "llama_cpp",
        {
            "disable_uvicorn_access_log": True,
            "uvicorn_log_level": "info",
            "max_model_len": "2K",
            "kv_cache_dtype": "fp8",
            "model": str(model_dir),
            "plain": "kept",
        },
    )

    assert converted["log_disable"] is True
    assert converted["log_verbosity"] == 1
    assert converted["ctx_size"] == 2048
    assert converted["cache_type_k"] == "q8_0"
    assert converted["cache_type_v"] == "q8_0"
    assert converted["model"] == str(gguf_file)
    assert converted["plain"] == "kept"

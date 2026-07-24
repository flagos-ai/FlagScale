import importlib
import sys
import types

import pytest
from omegaconf import OmegaConf


class DummyBackend:
    instances = []

    def __init__(self, config):
        self.config = config
        DummyBackend.instances.append(self)


class DummyLauncher:
    instances = []

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.calls = []
        DummyLauncher.instances.append(self)

    def run(self, *args, **kwargs):
        self.calls.append(("run", args, kwargs))
        return "run-result"

    def stop(self, *args, **kwargs):
        self.calls.append(("stop", args, kwargs))
        return "stop-result"

    def query(self, *args, **kwargs):
        self.calls.append(("query", args, kwargs))
        return "query-result"


class FakeRunnerFactory:
    backend_requests = []
    launcher_requests = []

    @classmethod
    def get_backend(cls, backend_type):
        cls.backend_requests.append(backend_type)
        return DummyBackend

    @classmethod
    def get_launcher(cls, launcher_type):
        cls.launcher_requests.append(launcher_type)
        return DummyLauncher


_MISSING = object()


@pytest.fixture
def runner_base_module(monkeypatch):
    DummyBackend.instances.clear()
    DummyLauncher.instances.clear()
    FakeRunnerFactory.backend_requests.clear()
    FakeRunnerFactory.launcher_requests.clear()

    factory_module = types.ModuleType("flagscale.runner.runner_factory")
    factory_module.RunnerFactory = FakeRunnerFactory
    previous_module = sys.modules.pop("flagscale.runner.runner_base", _MISSING)
    monkeypatch.setitem(sys.modules, "flagscale.runner.runner_factory", factory_module)

    module = importlib.import_module("flagscale.runner.runner_base")
    yield module

    sys.modules.pop("flagscale.runner.runner_base", None)
    if previous_module is not _MISSING:
        sys.modules["flagscale.runner.runner_base"] = previous_module


def make_config(task_type="train", backend="native", runner=None, task_extra=None, extra=None):
    task = {"type": task_type}
    if backend is not _MISSING:
        task["backend"] = backend
    if task_extra:
        task.update(task_extra)

    config = {
        "experiment": {
            "task": task,
            "runner": runner or {"type": "ssh", "hostfile": None},
        }
    }
    if extra:
        config.update(extra)
    return OmegaConf.create(config)


def test_runner_normalizes_native_backend_and_delegates_actions(runner_base_module):
    runner = runner_base_module.Runner(make_config())

    assert runner.backend_type == "native_train"
    assert FakeRunnerFactory.backend_requests == ["native_train"]
    assert FakeRunnerFactory.launcher_requests == ["ssh"]
    assert runner.run("arg", key="value") == "run-result"
    assert runner.stop() == "stop-result"
    assert runner.query() == "query-result"
    assert DummyLauncher.instances[-1].calls == [
        ("run", ("arg",), {"key": "value"}),
        ("stop", (), {}),
        ("query", (), {}),
    ]


def test_runner_requires_backend_for_train_tasks(runner_base_module):
    config = make_config(backend=_MISSING)

    with pytest.raises(AssertionError, match="backend_type is required"):
        runner_base_module.Runner(config)


def test_runner_rejects_unsupported_task_type(runner_base_module):
    config = make_config(task_type="unknown", backend="native")

    with pytest.raises(AssertionError, match="Unsupported task type"):
        runner_base_module.Runner(config)


def test_runner_rejects_backend_not_allowed_for_task(runner_base_module):
    config = make_config(task_type="inference", backend="native")

    with pytest.raises(AssertionError, match="Unsupported backend type"):
        runner_base_module.Runner(config)


def test_runner_serve_cloud_forces_vllm_backend(runner_base_module):
    config = make_config(
        task_type="serve",
        backend="native",
        runner={"type": "cloud", "hostfile": None},
        extra={"serve": [{"serve_id": "svc", "engine": "sglang"}]},
    )

    runner = runner_base_module.Runner(config)

    assert runner.backend_type == "vllm"
    assert FakeRunnerFactory.backend_requests == ["vllm"]
    assert FakeRunnerFactory.launcher_requests == ["cloud"]


def test_runner_serve_engine_selects_backend_when_no_entrypoint(runner_base_module):
    config = make_config(
        task_type="serve",
        backend=None,
        extra={"serve": [{"serve_id": "svc", "engine": "sglang"}]},
    )

    runner = runner_base_module.Runner(config)

    assert runner.backend_type == "sglang"
    assert FakeRunnerFactory.backend_requests == ["sglang"]


def test_runner_native_serve_requires_fs_serve_enabled(runner_base_module):
    config = make_config(
        task_type="serve",
        backend="native",
        runner={"type": "ssh", "hostfile": None, "deploy": {"use_fs_serve": False}},
        task_extra={"entrypoint": "serve.py"},
        extra={"serve": [{"serve_id": "svc", "engine": "vllm"}]},
    )

    with pytest.raises(ValueError, match="use_fs_serve"):
        runner_base_module.Runner(config)


def test_runner_parses_hostfile_when_configured(runner_base_module, mocker):
    parse = mocker.patch(
        "flagscale.runner.runner_base.parse_hostfile",
        return_value={"worker0": {"slots": 8, "type": "A100"}},
    )
    config = make_config(runner={"type": "ssh", "hostfile": "/tmp/hostfile"})

    runner = runner_base_module.Runner(config)

    parse.assert_called_once_with("/tmp/hostfile")
    assert runner.resources == {"worker0": {"slots": 8, "type": "A100"}}

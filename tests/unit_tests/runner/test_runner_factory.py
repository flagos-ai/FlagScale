import importlib
import sys
import types

import pytest


class DummyBackend:
    pass


class AnotherDummyBackend(DummyBackend):
    pass


class DummyLauncher:
    pass


class AnotherDummyLauncher(DummyLauncher):
    pass


_MISSING = object()


@pytest.fixture
def runner_factory(monkeypatch):
    """Import RunnerFactory with lightweight fake backend/launcher modules.

    runner_factory.py registers built-in classes at import time. The real backend
    package pulls in heavyweight optional dependencies, so this fixture replaces
    those imports with simple stand-ins and keeps the test focused on registry
    behavior.
    """
    backend_module = types.ModuleType("flagscale.runner.backend")
    for name in (
        "BackendBase",
        "LlamaCppBackend",
        "MegatronBackend",
        "NativeCompressBackend",
        "NativeServeBackend",
        "NativeTrainBackend",
        "SglangBackend",
        "VerlBackend",
        "VllmBackend",
    ):
        setattr(backend_module, name, type(name, (), {}))

    launcher_module = types.ModuleType("flagscale.runner.launcher")
    for name in ("CloudLauncher", "LauncherBase", "SshLauncher"):
        setattr(launcher_module, name, type(name, (), {}))

    monkeypatch.setitem(sys.modules, "flagscale.runner.backend", backend_module)
    monkeypatch.setitem(sys.modules, "flagscale.runner.launcher", launcher_module)
    previous_factory_module = sys.modules.pop("flagscale.runner.runner_factory", _MISSING)

    module = importlib.import_module("flagscale.runner.runner_factory")
    factory = module.RunnerFactory
    factory._backend_registry = {}
    factory._launcher_registry = {}
    yield factory

    sys.modules.pop("flagscale.runner.runner_factory", None)
    if previous_factory_module is not _MISSING:
        sys.modules["flagscale.runner.runner_factory"] = previous_factory_module


def test_register_and_get_backend(runner_factory):
    runner_factory.register_backend("dummy", DummyBackend)

    assert runner_factory.get_backend("dummy") is DummyBackend


def test_register_backend_rejects_duplicate_name(runner_factory):
    runner_factory.register_backend("dummy", DummyBackend)

    with pytest.raises(ValueError, match="already registered"):
        runner_factory.register_backend("dummy", AnotherDummyBackend)


def test_get_backend_rejects_unknown_name(runner_factory):
    with pytest.raises(ValueError, match="Unknown backend type"):
        runner_factory.get_backend("missing")


def test_register_and_get_launcher(runner_factory):
    runner_factory.register_launcher("dummy", DummyLauncher)

    assert runner_factory.get_launcher("dummy") is DummyLauncher


def test_register_launcher_rejects_duplicate_name(runner_factory):
    runner_factory.register_launcher("dummy", DummyLauncher)

    with pytest.raises(ValueError, match="already registered"):
        runner_factory.register_launcher("dummy", AnotherDummyLauncher)


def test_get_launcher_rejects_unknown_name(runner_factory):
    with pytest.raises(ValueError, match="Unknown launcher type"):
        runner_factory.get_launcher("missing")

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


def _ensure_torch_module():
    try:
        importlib.import_module("torch")
    except ModuleNotFoundError:
        fake_torch = types.ModuleType("torch")
        fake_torch.device = MagicMock(side_effect=lambda kind, index=None: (kind, index))
        fake_torch.cuda = types.SimpleNamespace(
            is_available=MagicMock(return_value=False),
            device_count=MagicMock(return_value=0),
            set_device=MagicMock(),
            manual_seed_all=MagicMock(),
        )
        sys.modules["torch"] = fake_torch
    return sys.modules["torch"]


_ensure_torch_module()

from flagscale.platforms.platform_base import PlatformBase
from flagscale.platforms.platform_cuda import PlatformCUDA
from flagscale.platforms.platform_musa import PlatformMUSA
from flagscale.platforms.platform_npu import PlatformNPU


class DummyPlatform(PlatformBase):
    def __init__(self, platform_name="dummy", available=True):
        self.platform_name = platform_name
        self.available = available

    def name(self) -> str:
        return self.platform_name

    def is_available(self) -> bool:
        return self.available

    def set_device(self, device_index):
        self.device_index = device_index

    def device(self, device_index=None):
        return (self.platform_name, device_index)

    def device_count(self) -> int:
        return 1 if self.available else 0

    def dist_backend(self) -> str:
        return f"{self.platform_name}_backend"

    def manual_seed_all(self, seed):
        self.seed = seed

    def amp_device_type(self) -> str:
        return self.platform_name


@pytest.fixture(autouse=True)
def reset_platform_manager(monkeypatch):
    from flagscale.platforms import platform_manager, platform_register

    monkeypatch.delenv("FS_PLATFORM", raising=False)
    platform_manager._current_platform = None
    platform_register.PLATFORMS.clear()
    yield
    platform_manager._current_platform = None
    platform_register.PLATFORMS.clear()


def test_platform_base_defaults_and_abstract_contract():
    with pytest.raises(TypeError):
        PlatformBase()

    assert DummyPlatform().supports_distributions_on_device() is True


def test_cuda_platform_device_mapping_backend_seed_and_availability(monkeypatch):
    torch = _ensure_torch_module()
    cuda = types.SimpleNamespace(
        is_available=MagicMock(return_value=True),
        device_count=MagicMock(return_value=2),
        set_device=MagicMock(),
        manual_seed_all=MagicMock(),
    )
    monkeypatch.setattr(torch, "cuda", cuda, raising=False)
    monkeypatch.setattr(
        torch, "device", MagicMock(side_effect=lambda kind, index=None: (kind, index))
    )

    platform = PlatformCUDA()

    assert platform.name() == "cuda"
    assert platform.is_available() is True
    assert platform.device(1) == ("cuda", 1)
    assert platform.device_count() == 2
    assert platform.dist_backend() == "nccl"
    assert platform.amp_device_type() == "cuda"

    platform.set_device(1)
    platform.manual_seed_all(123)
    cuda.set_device.assert_called_once_with(1)
    cuda.manual_seed_all.assert_called_once_with(123)


def test_cuda_platform_unavailable_on_zero_devices_and_exceptions(monkeypatch):
    torch = _ensure_torch_module()
    cuda = types.SimpleNamespace(
        is_available=MagicMock(return_value=True),
        device_count=MagicMock(return_value=0),
    )
    monkeypatch.setattr(torch, "cuda", cuda, raising=False)
    assert PlatformCUDA().is_available() is False

    cuda.device_count.side_effect = RuntimeError("driver error")
    assert PlatformCUDA().is_available() is False


def test_cuda_platform_unavailable_short_circuits_when_cuda_is_unavailable(monkeypatch):
    torch = _ensure_torch_module()
    cuda = types.SimpleNamespace(
        is_available=MagicMock(return_value=False),
        device_count=MagicMock(return_value=8),
    )
    monkeypatch.setattr(torch, "cuda", cuda, raising=False)

    assert PlatformCUDA().is_available() is False
    cuda.device_count.assert_not_called()


def test_npu_platform_methods_and_availability(monkeypatch):
    torch = _ensure_torch_module()
    npu = types.SimpleNamespace(
        is_available=MagicMock(return_value=True),
        device_count=MagicMock(return_value=4),
        set_device=MagicMock(),
        manual_seed_all=MagicMock(),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(
        torch, "device", MagicMock(side_effect=lambda kind, index=None: (kind, index))
    )
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))

    platform = PlatformNPU()

    assert platform.name() == "npu"
    assert platform.is_available() is True
    assert platform.device(2) == ("npu", 2)
    assert platform.device_count() == 4
    assert platform.dist_backend() == "hccl"
    assert platform.amp_device_type() == "npu"

    platform.set_device(2)
    platform.manual_seed_all(321)
    npu.set_device.assert_called_once_with(2)
    npu.manual_seed_all.assert_called_once_with(321)


def test_npu_platform_unavailable_without_extension_or_on_exception(monkeypatch):
    torch = _ensure_torch_module()
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    assert PlatformNPU().is_available() is False

    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(
            is_available=MagicMock(return_value=True),
            device_count=MagicMock(side_effect=RuntimeError("bad npu")),
        ),
        raising=False,
    )
    assert PlatformNPU().is_available() is False


def test_npu_platform_unavailable_short_circuits_when_npu_is_unavailable(monkeypatch):
    torch = _ensure_torch_module()
    npu = types.SimpleNamespace(
        is_available=MagicMock(return_value=False),
        device_count=MagicMock(return_value=4),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))

    assert PlatformNPU().is_available() is False
    npu.device_count.assert_not_called()


def test_musa_platform_methods_availability_and_distribution_support(monkeypatch):
    torch = _ensure_torch_module()
    musa = types.SimpleNamespace(
        is_available=MagicMock(return_value=True),
        device_count=MagicMock(return_value=3),
        set_device=MagicMock(),
        manual_seed_all=MagicMock(),
    )
    monkeypatch.setattr(torch, "musa", musa, raising=False)
    monkeypatch.setattr(
        torch, "device", MagicMock(side_effect=lambda kind, index=None: (kind, index))
    )
    monkeypatch.setitem(sys.modules, "torch_musa", types.ModuleType("torch_musa"))

    platform = PlatformMUSA()

    assert platform.name() == "musa"
    assert platform.is_available() is True
    assert platform.device(0) == ("musa", 0)
    assert platform.device_count() == 3
    assert platform.dist_backend() == "mccl"
    assert platform.amp_device_type() == "musa"
    assert platform.supports_distributions_on_device() is False

    platform.set_device(0)
    platform.manual_seed_all(999)
    musa.set_device.assert_called_once_with(0)
    musa.manual_seed_all.assert_called_once_with(999)


def test_musa_platform_unavailable_without_extension_or_devices(monkeypatch):
    torch = _ensure_torch_module()
    monkeypatch.delitem(sys.modules, "torch_musa", raising=False)
    assert PlatformMUSA().is_available() is False

    monkeypatch.setitem(sys.modules, "torch_musa", types.ModuleType("torch_musa"))
    monkeypatch.setattr(
        torch,
        "musa",
        types.SimpleNamespace(
            is_available=MagicMock(return_value=True),
            device_count=MagicMock(return_value=0),
        ),
        raising=False,
    )
    assert PlatformMUSA().is_available() is False


def test_musa_platform_unavailable_short_circuits_when_musa_is_unavailable(monkeypatch):
    torch = _ensure_torch_module()
    musa = types.SimpleNamespace(
        is_available=MagicMock(return_value=False),
        device_count=MagicMock(return_value=3),
    )
    monkeypatch.setattr(torch, "musa", musa, raising=False)
    monkeypatch.setitem(sys.modules, "torch_musa", types.ModuleType("torch_musa"))

    assert PlatformMUSA().is_available() is False
    musa.device_count.assert_not_called()


def test_register_platforms_registers_only_available_platforms(monkeypatch):
    from flagscale.platforms import (
        platform_cuda,
        platform_musa,
        platform_npu,
        platform_register,
    )

    monkeypatch.setattr(platform_cuda, "PlatformCUDA", lambda: DummyPlatform("cuda", True))
    monkeypatch.setattr(platform_npu, "PlatformNPU", lambda: DummyPlatform("npu", False))
    monkeypatch.setattr(platform_musa, "PlatformMUSA", lambda: DummyPlatform("musa", True))

    platform_register.register_platforms()

    assert sorted(platform_register.PLATFORMS) == ["cuda", "musa"]
    assert platform_register.PLATFORMS["cuda"].name() == "cuda"
    assert platform_register.PLATFORMS["musa"].name() == "musa"


def test_register_platforms_is_idempotent_for_repeated_registration(monkeypatch):
    from flagscale.platforms import (
        platform_cuda,
        platform_musa,
        platform_npu,
        platform_register,
    )

    monkeypatch.setattr(platform_cuda, "PlatformCUDA", lambda: DummyPlatform("cuda", True))
    monkeypatch.setattr(platform_npu, "PlatformNPU", lambda: DummyPlatform("npu", False))
    monkeypatch.setattr(platform_musa, "PlatformMUSA", lambda: DummyPlatform("musa", False))

    platform_register.register_platforms()
    first_cuda = platform_register.PLATFORMS["cuda"]
    platform_register.register_platforms()

    assert list(platform_register.PLATFORMS) == ["cuda"]
    assert platform_register.PLATFORMS["cuda"] is not first_cuda


def test_register_platforms_keeps_registry_empty_when_all_platforms_unavailable(
    monkeypatch,
):
    from flagscale.platforms import (
        platform_cuda,
        platform_musa,
        platform_npu,
        platform_register,
    )

    monkeypatch.setattr(platform_cuda, "PlatformCUDA", lambda: DummyPlatform("cuda", False))
    monkeypatch.setattr(platform_npu, "PlatformNPU", lambda: DummyPlatform("npu", False))
    monkeypatch.setattr(platform_musa, "PlatformMUSA", lambda: DummyPlatform("musa", False))

    platform_register.register_platforms()

    assert platform_register.PLATFORMS == {}


def test_get_platform_uses_env_override_and_caches_result(monkeypatch):
    from flagscale.platforms import platform_manager, platform_register

    cuda = DummyPlatform("cuda")
    npu = DummyPlatform("npu")
    platform_register.PLATFORMS.update({"cuda": cuda, "npu": npu})
    monkeypatch.setenv("FS_PLATFORM", "NPU")

    assert platform_manager.get_platform() is npu
    monkeypatch.setenv("FS_PLATFORM", "cuda")
    assert platform_manager.get_platform() is npu


def test_get_platform_unknown_env_reports_registered_platforms(monkeypatch):
    from flagscale.platforms import platform_manager, platform_register

    platform_register.PLATFORMS.update({"cuda": DummyPlatform("cuda")})
    monkeypatch.setenv("FS_PLATFORM", "unknown")

    with pytest.raises(ValueError, match="FS_PLATFORM='unknown' is not available"):
        platform_manager.get_platform()


def test_get_platform_auto_detects_priority_and_reports_no_available_platform(
    monkeypatch,
):
    from flagscale.platforms import platform_manager, platform_register

    platform_register.PLATFORMS.update({"musa": DummyPlatform("musa"), "npu": DummyPlatform("npu")})
    assert platform_manager.get_platform().name() == "npu"

    platform_manager._current_platform = None
    platform_register.PLATFORMS.clear()
    with pytest.raises(RuntimeError, match="No available platform detected"):
        platform_manager.get_platform()


def test_get_platform_auto_detects_cuda_before_other_registered_platforms():
    from flagscale.platforms import platform_manager, platform_register

    platform_register.PLATFORMS.update(
        {
            "musa": DummyPlatform("musa"),
            "npu": DummyPlatform("npu"),
            "cuda": DummyPlatform("cuda"),
        }
    )

    assert platform_manager.get_platform().name() == "cuda"


def test_set_platform_overrides_current_platform():
    from flagscale.platforms import platform_manager

    platform = DummyPlatform("manual")
    platform_manager.set_platform(platform)

    assert platform_manager.get_platform() is platform


def test_set_platform_none_reenables_auto_detection():
    from flagscale.platforms import platform_manager, platform_register

    platform_register.PLATFORMS.update({"cuda": DummyPlatform("cuda")})
    platform_manager.set_platform(DummyPlatform("manual"))
    platform_manager.set_platform(None)

    assert platform_manager.get_platform().name() == "cuda"


def test_package_exports_and_registers_on_import(monkeypatch):
    import flagscale.platforms as platforms_pkg

    assert platforms_pkg.PlatformBase is PlatformBase
    assert platforms_pkg.get_platform
    assert platforms_pkg.set_platform
    assert "PlatformBase" in platforms_pkg.__all__

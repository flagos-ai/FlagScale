# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the real profiler override dispatch with mocked backend SDKs."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import megatron.plugin_flagscale
from megatron.plugin import decorators
from megatron.plugin.platform import platform_manager
from megatron.training import training, utils


@pytest.fixture
def profiler_env(monkeypatch, tmp_path):
    # Dispatch is cached on first use, so each scenario needs fresh lookup state.
    monkeypatch.setattr(decorators, "_plugin_registry", {})
    monkeypatch.setattr(decorators, "_lazy_registry", {})
    monkeypatch.setattr(decorators, "_plugin_impl_cache", {})
    monkeypatch.setattr(decorators, "_original_impl_cache", set())
    runpy.run_path(str(Path(megatron.plugin_flagscale.__file__).with_name("override_registry.py")))
    monkeypatch.setenv("MG_FL_PREFER", "cuda")

    standard = Mock()
    npu = Mock()
    observer = standard.ExecutionTraceObserver.return_value
    observer.register_callback.return_value = observer
    standard.profile.return_value.execution_trace_observer = None
    npu.profile.return_value = Mock(spec=["start", "step", "stop", "export_chrome_trace"])
    monkeypatch.setattr(training.torch, "profiler", standard)
    # A vendor SDK being installed must not by itself select that vendor.
    npu_module = ModuleType("torch_npu")
    npu_module.profiler = npu
    monkeypatch.setitem(sys.modules, "torch_npu", npu_module)
    monkeypatch.setitem(sys.modules, "torch_npu.profiler", npu)
    monkeypatch.delitem(sys.modules, "megatron.plugin_flagscale.npu_plugin", raising=False)
    monkeypatch.setattr(megatron.plugin_flagscale, "npu_plugin", None, raising=False)
    monkeypatch.setattr(training, "print_rank_0", Mock())
    monkeypatch.setattr(utils, "print_rank_0", Mock())
    reports = Mock(return_value=("details.csv", "summary.csv", "operators.csv"))
    monkeypatch.setattr(training, "export_kernel_reports", reports)

    args = SimpleNamespace(
        profile_step_start=3,
        profile_step_end=8,
        tensorboard_dir=str(tmp_path / "tensorboard"),
        pytorch_profiler_collect_shapes=True,
        pytorch_profiler_collect_memory=True,
        pytorch_profiler_collect_callstack=False,
        pytorch_profiler_collect_chakra=False,
    )
    return SimpleNamespace(args=args, standard=standard, npu=npu, reports=reports)


def test_default_uses_standard_profiler_and_preserves_reports(profiler_env):
    env = profiler_env
    prof = training.create_pytorch_profiler(env.args, rank=7)
    assert prof is env.standard.profile.return_value
    assert "megatron.plugin_flagscale.npu_plugin" not in sys.modules
    env.npu.profile.assert_not_called()
    prof.start.assert_not_called()
    env.standard.schedule.assert_called_once_with(wait=2, warmup=1, active=5, repeat=1)
    kwargs = env.standard.profile.call_args.kwargs
    assert kwargs["execution_trace_observer"] is None
    assert kwargs["record_shapes"] is True
    assert kwargs["profile_memory"] is True
    assert kwargs["with_stack"] is False

    kwargs["on_trace_ready"](prof)
    trace_path = f"{env.args.tensorboard_dir}/../torch_profile/rank-7.json.gz"
    prof.export_chrome_trace.assert_called_once_with(trace_path)
    env.reports.assert_called_once_with(
        prof.events.return_value, Path(trace_path).parent, 7, trace_path
    )
    training.stop_pytorch_profiler(prof)
    prof.stop.assert_called_once_with()


@pytest.mark.parametrize("selector", ["explicit", "platform"])
def test_npu_dispatch_exports_trace_without_events(profiler_env, monkeypatch, selector):
    env = profiler_env
    if selector == "explicit":
        monkeypatch.setenv("MG_FL_PREFER", "npu")
    else:
        monkeypatch.delenv("MG_FL_PREFER")
        monkeypatch.setattr(
            platform_manager, "cur_platform", SimpleNamespace(platform_name=lambda: "npu")
        )
    assert "megatron.plugin_flagscale.npu_plugin" not in sys.modules
    prof = training.create_pytorch_profiler(env.args, rank=7)
    assert prof is env.npu.profile.return_value
    assert "megatron.plugin_flagscale.npu_plugin" in sys.modules
    env.standard.profile.assert_not_called()
    env.standard.ExecutionTraceObserver.assert_not_called()
    prof.start.assert_not_called()
    env.npu.schedule.assert_called_once_with(wait=2, warmup=1, active=5, repeat=1)
    env.npu.tensorboard_trace_handler.assert_not_called()
    kwargs = env.npu.profile.call_args.kwargs
    assert "execution_trace_observer" not in kwargs
    assert kwargs["record_shapes"] is True
    assert kwargs["profile_memory"] is True
    assert kwargs["with_stack"] is False

    prof.start()
    prof.step()
    training.stop_pytorch_profiler(prof)
    assert prof.mock_calls == [("start", (), {}), ("step", (), {}), ("stop", (), {})]
    kwargs["on_trace_ready"](prof)
    trace_dir = Path(env.args.tensorboard_dir) / ".." / "torch_profile"
    assert trace_dir.is_dir()
    prof.export_chrome_trace.assert_called_once_with(f"{trace_dir}/rank-7.json.gz")
    env.reports.assert_not_called()
    assert decorators._plugin_impl_cache[training.stop_pytorch_profiler.__wrapped__].__module__ == (
        "megatron.plugin_flagscale.npu_plugin"
    )


def test_npu_does_not_create_standard_chakra_observer(profiler_env, monkeypatch):
    env = profiler_env
    monkeypatch.setenv("MG_FL_PREFER", "npu")
    env.args.pytorch_profiler_collect_chakra = True
    training.create_pytorch_profiler(env.args, rank=0)
    env.standard.ExecutionTraceObserver.assert_not_called()
    assert "execution_trace_observer" not in env.npu.profile.call_args.kwargs


@pytest.mark.parametrize("failure", [None, "stop"])
def test_standard_chakra_observer_is_released(profiler_env, failure):
    env = profiler_env
    env.args.pytorch_profiler_collect_chakra = True
    observer = env.standard.ExecutionTraceObserver.return_value
    prof = env.standard.profile.return_value
    prof.execution_trace_observer = observer
    assert training.create_pytorch_profiler(env.args, rank=2) is prof
    assert env.standard.profile.call_args.kwargs["execution_trace_observer"] is observer
    if failure == "stop":
        prof.stop.side_effect = RuntimeError("trace export failed")
        with pytest.raises(RuntimeError, match="trace export failed"):
            training.stop_pytorch_profiler(prof)
    else:
        training.stop_pytorch_profiler(prof)
    observer.register_callback.assert_called_once_with(
        f"{env.args.tensorboard_dir}/../chakra/rank-2.json.gz"
    )
    observer.unregister_callback.assert_called_once_with()


@pytest.mark.parametrize("vendor", ["cuda", "npu"])
def test_zero_start_has_no_warmup(profiler_env, monkeypatch, vendor):
    env = profiler_env
    monkeypatch.setenv("MG_FL_PREFER", vendor)
    env.args.profile_step_start = 0
    env.args.profile_step_end = 2
    training.create_pytorch_profiler(env.args, rank=0)
    backend = env.npu if vendor == "npu" else env.standard
    backend.schedule.assert_called_once_with(wait=0, warmup=0, active=2, repeat=1)

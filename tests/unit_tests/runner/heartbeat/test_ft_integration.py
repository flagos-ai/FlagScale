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

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def ft_integration(monkeypatch):
    """Load ft_integration with only the small dependency surface used by these tests."""
    package_name = "heartbeat_ft_integration_test"
    package = types.ModuleType(package_name)
    package.__path__ = []

    arguments = types.ModuleType(f"{package_name}.arguments")
    global_vars = types.ModuleType(f"{package_name}.global_vars")
    global_vars.get_args = lambda: SimpleNamespace(curr_iteration=17)
    utils = types.ModuleType(f"{package_name}.utils")
    utils.is_rank0 = lambda: True
    utils.print_rank_0 = lambda *args, **kwargs: None
    async_utils = types.ModuleType(f"{package_name}.async_utils")
    async_utils.is_empty_async_queue = lambda: True

    megatron = types.ModuleType("megatron")
    megatron.__path__ = []
    plugin = types.ModuleType("megatron.plugin")
    plugin.__path__ = []
    platform = types.ModuleType("megatron.plugin.platform")
    platform.get_platform = lambda: SimpleNamespace()
    torch = types.ModuleType("torch")

    modules = {
        package_name: package,
        f"{package_name}.arguments": arguments,
        f"{package_name}.global_vars": global_vars,
        f"{package_name}.utils": utils,
        f"{package_name}.async_utils": async_utils,
        "megatron": megatron,
        "megatron.plugin": plugin,
        "megatron.plugin.platform": platform,
        "torch": torch,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = (
        Path(__file__).parents[4]
        / "flagscale"
        / "train"
        / "megatron"
        / "training"
        / "ft_integration.py"
    )
    spec = importlib.util.spec_from_file_location(f"{package_name}.ft_integration", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_actual_checkpoint_is_always_tracked(ft_integration, monkeypatch):
    starts = []
    monkeypatch.setattr(
        ft_integration.gpu_heartbeat,
        "checkpoint_start",
        lambda iteration: starts.append(iteration),
    )
    monkeypatch.setattr(
        ft_integration,
        "is_empty_async_queue",
        lambda: pytest.fail("actual saves must not inspect the async queue"),
    )

    ft_integration.on_checkpointing_start()

    assert starts == [17]


def test_nonblocking_async_poll_does_not_publish_checkpoint(ft_integration, monkeypatch):
    monkeypatch.setattr(
        ft_integration.gpu_heartbeat,
        "checkpoint_start",
        lambda iteration: pytest.fail("non-blocking polling must not publish a checkpoint"),
    )
    monkeypatch.setattr(
        ft_integration,
        "is_empty_async_queue",
        lambda: pytest.fail("non-blocking polling must not inspect the async queue"),
    )
    monkeypatch.setattr(
        ft_integration.gpu_heartbeat,
        "checkpoint_end",
        lambda: pytest.fail("non-blocking polling must not end a checkpoint"),
    )

    for _ in range(1000):
        ft_integration.on_checkpointing_start(is_async_finalization=True, blocking=False)
        ft_integration.on_checkpointing_end(is_async_finalization=True, blocking=False)


@pytest.mark.parametrize(
    ("queue_is_empty", "expected_starts"),
    [(True, []), (False, [17])],
)
def test_blocking_async_finalization_tracks_only_pending_work(
    ft_integration,
    monkeypatch,
    queue_is_empty,
    expected_starts,
):
    starts = []
    monkeypatch.setattr(ft_integration, "is_empty_async_queue", lambda: queue_is_empty)
    monkeypatch.setattr(
        ft_integration.gpu_heartbeat,
        "checkpoint_start",
        lambda iteration: starts.append(iteration),
    )

    ft_integration.on_checkpointing_start(is_async_finalization=True, blocking=True)

    assert starts == expected_starts

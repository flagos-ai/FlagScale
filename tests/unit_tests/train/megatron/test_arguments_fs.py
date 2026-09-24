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

import pytest
import torch


def _load_arguments_fs(monkeypatch):
    parallel_context = types.ModuleType("megatron.plugin.hetero.parallel_context")
    parallel_context.RankMapper = object
    platform = types.ModuleType("megatron.plugin.platform")
    platform.get_platform = lambda: object()

    monkeypatch.setitem(sys.modules, "flagcx", types.ModuleType("flagcx"))
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.plugin", types.ModuleType("megatron.plugin"))
    monkeypatch.setitem(sys.modules, "megatron.plugin.hetero.parallel_context", parallel_context)
    monkeypatch.setitem(sys.modules, "megatron.plugin.platform", platform)

    source = (
        Path(__file__).resolve().parents[4] / "flagscale/train/megatron/training/arguments_fs.py"
    )
    spec = importlib.util.spec_from_file_location("arguments_fs_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "devices,expected",
    [
        (["cuda"], "cpu:gloo,cuda:flagcx"),
        (["txda"], "cpu:gloo,txda:flagcx"),
        (["cpu", "cuda", "txda"], "cpu:gloo,cuda:flagcx,txda:flagcx"),
    ],
)
def test_flagcx_backend_config_uses_registered_capabilities(monkeypatch, devices, expected):
    arguments_fs = _load_arguments_fs(monkeypatch)
    monkeypatch.setitem(torch.distributed.Backend.backend_capability, "flagcx", devices)

    assert arguments_fs._get_flagcx_backend_config() == expected


@pytest.mark.parametrize("devices", [[], ["cpu"]])
def test_flagcx_backend_config_requires_accelerator(monkeypatch, devices):
    arguments_fs = _load_arguments_fs(monkeypatch)
    monkeypatch.setitem(torch.distributed.Backend.backend_capability, "flagcx", devices)

    with pytest.raises(RuntimeError, match="did not register an accelerator device"):
        arguments_fs._get_flagcx_backend_config()

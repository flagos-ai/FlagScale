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

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_distributed_backends():
    source = (
        Path(__file__).resolve().parents[4]
        / "flagscale/train/megatron/training/distributed_backends.py"
    )
    spec = importlib.util.spec_from_file_location(
        "distributed_backends_under_test", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_flagcx_backend_is_unchanged():
    module = _load_distributed_backends()

    assert module.resolve_distributed_backend("nccl") == "nccl"


@pytest.mark.parametrize(
    "devices,expected",
    [
        (["cuda"], "cpu:gloo,cuda:flagcx"),
        (["txda"], "cpu:gloo,txda:flagcx"),
        (["cpu", "cuda", "txda"], "cpu:gloo,cuda:flagcx,txda:flagcx"),
    ],
)
def test_flagcx_backend_uses_registered_capabilities(monkeypatch, devices, expected):
    module = _load_distributed_backends()
    monkeypatch.setitem(sys.modules, "flagcx", types.ModuleType("flagcx"))
    monkeypatch.setitem(
        torch.distributed.Backend.backend_capability, "flagcx", devices
    )

    assert module.resolve_distributed_backend("flagcx") == expected


@pytest.mark.parametrize("devices", [[], ["cpu"]])
def test_flagcx_backend_requires_accelerator(monkeypatch, devices):
    module = _load_distributed_backends()
    monkeypatch.setitem(sys.modules, "flagcx", types.ModuleType("flagcx"))
    monkeypatch.setitem(
        torch.distributed.Backend.backend_capability, "flagcx", devices
    )

    with pytest.raises(RuntimeError, match="did not register an accelerator device"):
        module.resolve_distributed_backend("flagcx")


@pytest.mark.parametrize("filename", ["arguments_fs.py", "initialize.py"])
def test_training_initializers_use_shared_backend_resolver(filename):
    training_dir = (
        Path(__file__).resolve().parents[4]
        / "flagscale/train/megatron/training"
    )
    tree = ast.parse((training_dir / filename).read_text())

    resolver_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_distributed_backend"
    ]

    assert resolver_calls, f"{filename} bypasses the shared backend resolver"

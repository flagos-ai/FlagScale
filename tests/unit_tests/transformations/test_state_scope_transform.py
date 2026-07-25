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

import unittest
from typing import Any

import torch
from torch import nn

from flagscale.inference.runtime_context import RuntimeContext
from flagscale.transformations.hook import ModelHook, ModuleHookRegistry
from flagscale.transformations.state_scope_transformation import (
    StateScopeHook,
    StateScopeTransformation,
)
from flagscale.transformations.state_store import StateStore


class DummyPipeline:
    def __init__(self) -> None:
        self.unet = nn.Sequential(nn.Linear(2, 2))


class DummyHook(ModelHook):
    def __init__(self) -> None:
        super().__init__()

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple[Any], dict[str, Any]]:
        if self._stateful:
            self._stateful[0].get_or_create_state()
        return args, kwargs


class TestStateScopeTransform(unittest.TestCase):
    def test_apply_registers_hook_on_backbone(self):
        pipeline = DummyPipeline()
        transform = StateScopeTransformation()

        applied = transform.apply(pipeline.unet)
        self.assertTrue(applied)

        reg = ModuleHookRegistry.get_registry_if_present(pipeline.unet)
        self.assertIsNotNone(reg)
        self.assertIsInstance(reg.get_hook("state_scope"), StateScopeHook)

    def test_hook_sets_and_resets_state_context_during_forward(self):
        pipeline = DummyPipeline()
        backbone = pipeline.unet

        reg = ModuleHookRegistry.get_or_create_registry(backbone)
        store = StateStore(dict)

        # To make it work, we need to register the dummy hook first
        hook = DummyHook()
        reg.register_hook(hook, "dummy")
        hook.register_stateful(store)

        transform = StateScopeTransformation()
        transform.apply(backbone)

        x = torch.zeros(1, 2)
        ctx = RuntimeContext()
        ctx.state_scope_provider = lambda: "ctxA"
        with ctx.session():
            _ = backbone(x)

        self.assertIn("ctxA", store._state_by_scope)

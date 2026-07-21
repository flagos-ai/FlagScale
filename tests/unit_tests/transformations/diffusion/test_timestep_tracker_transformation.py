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

import torch
import torch.nn as nn

from flagscale.inference.core.diffusion.timestep_tracker_transformation import (
    TimestepTrackerHook,
    TimestepTrackerTransformation,
)
from flagscale.inference.runtime_context import RuntimeContext
from flagscale.transformations.hook import ModuleHookRegistry


class TrackerModule(nn.Module):
    def forward(self, x, timestep):
        return x


class TestTimestepTrackerTransformation(unittest.TestCase):
    def test_apply_registers_hook(self):
        module = TrackerModule()
        transform = TimestepTrackerTransformation()

        applied = transform.apply(module)
        self.assertTrue(applied)

        reg = ModuleHookRegistry.get_registry_if_present(module)
        self.assertIsNotNone(reg)
        hook = reg.get_hook("timestep_tracker")
        self.assertIsInstance(hook, TimestepTrackerHook)

    def test_pre_forward_updates_runtime_context(self):
        module = TrackerModule()
        TimestepTrackerTransformation().apply(module)

        ctx = RuntimeContext()
        with ctx.session():
            self.assertEqual(ctx.timestep_index, -1)
            self.assertEqual(ctx.timestep, -1)

            _ = module(torch.zeros(1), timestep=0)
            self.assertEqual(ctx.timestep_index, 0)
            self.assertEqual(ctx.timestep, 0.0)

            # Same timestep should not advance index
            _ = module(torch.zeros(1), timestep=0)
            self.assertEqual(ctx.timestep_index, 0)
            self.assertEqual(ctx.timestep, 0.0)

            # Tensor input timestep should be parsed and advance index
            _ = module(torch.zeros(1), timestep=torch.tensor(1))
            self.assertEqual(ctx.timestep_index, 1)
            self.assertEqual(ctx.timestep, 1.0)

    def test_no_context_raises(self):
        module = TrackerModule()
        TimestepTrackerTransformation().apply(module)

        with self.assertRaises(ValueError):
            _ = module(torch.zeros(1), timestep=0)

    def test_missing_timestep_raises(self):
        module = TrackerModule()
        TimestepTrackerTransformation().apply(module)

        ctx = RuntimeContext()
        with ctx.session(), self.assertRaises(ValueError):
            _ = module(torch.zeros(1))

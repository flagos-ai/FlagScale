import unittest

import torch
import torch.nn as nn

from flagscale.inference.runtime_context import RuntimeContext
from flagscale.transformations.diffusion.taylorseer_transformation import (
    TaylorSeerHook,
    TaylorSeerTransformation,
)
from flagscale.transformations.hook import ModuleHookRegistry


class TestTaylorSeerTransformation(unittest.TestCase):
    def test_apply_registers_hook(self):
        module = nn.Linear(1, 1)
        transform = TaylorSeerTransformation(order=1, warmup_steps=1, skip_interval_steps=2)

        applied = transform.apply(module)
        self.assertTrue(applied)

        reg = ModuleHookRegistry.get_registry_if_present(module)
        self.assertIsNotNone(reg)
        hook = reg.get_hook("taylorseer")
        self.assertIsInstance(hook, TaylorSeerHook)

    def test_exact_then_approx_then_exact_with_index_delta(self):
        module = nn.Linear(1, 1)
        with torch.no_grad():
            module.weight.fill_(2.0)
            module.bias.fill_(1.0)

        transform = TaylorSeerTransformation(order=1, warmup_steps=1, skip_interval_steps=2)
        transform.apply(module)

        # Ensure the state scope is set so StateStore can create/access its state
        reg = ModuleHookRegistry.get_registry_if_present(module)
        reg.set_state_scope("scopeA")

        ctx = RuntimeContext()
        with ctx.session():
            # Step 0: exact forward
            ctx.update_timestep(0)
            y0 = module(torch.tensor([[2.0]]))
            self.assertAlmostEqual(float(y0.item()), 5.0, places=5)

            # Step 1: approximate forward (order=1 -> returns previous output)
            ctx.update_timestep(1)
            y1 = module(torch.tensor([[3.0]]))
            self.assertAlmostEqual(float(y1.item()), 5.0, places=5)

            # Step 2: exact forward again due to skip interval
            ctx.update_timestep(2)
            y2 = module(torch.tensor([[4.0]]))
            self.assertAlmostEqual(float(y2.item()), 9.0, places=5)

    def test_tuple_output_supported(self):
        class TupleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(1, 1)
                with torch.no_grad():
                    self.lin.weight.fill_(1.5)
                    self.lin.bias.fill_(0.5)

            def forward(self, x):
                return (self.lin(x),)

        module = TupleModule()
        transform = TaylorSeerTransformation(order=1, warmup_steps=1, skip_interval_steps=2)
        transform.apply(module)

        reg = ModuleHookRegistry.get_registry_if_present(module)
        reg.set_state_scope("scopeB")

        ctx = RuntimeContext()
        with ctx.session():
            ctx.update_timestep(0)
            out0 = module(torch.tensor([[2.0]]))
            self.assertIsInstance(out0, tuple)
            self.assertEqual(len(out0), 1)
            self.assertAlmostEqual(float(out0[0].item()), 3.5, places=5)

            ctx.update_timestep(1)
            out1 = module(torch.tensor([[4.0]]))
            self.assertIsInstance(out1, tuple)
            self.assertEqual(len(out1), 1)
            # order=1 approximation returns previous exact output
            self.assertAlmostEqual(float(out1[0].item()), float(out0[0].item()), places=5)

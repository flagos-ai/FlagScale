# Modified from https://github.com/huggingface/diffusers/blob/main/tests/hooks/test_hooks.py

import gc
import unittest

import torch

from flagscale.runner.utils import logger
from flagscale.transformations.hook import ModelHook, ModuleHookRegistry


class DummyBlock(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()

        self.proj_in = torch.nn.Linear(in_features, hidden_features)
        self.activation = torch.nn.ReLU()
        self.proj_out = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.proj_out(x)
        return x


class DummyModel(torch.nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, num_layers: int
    ) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_features, hidden_features)
        self.activation = torch.nn.ReLU()
        self.blocks = torch.nn.ModuleList(
            [
                DummyBlock(hidden_features, hidden_features, hidden_features)
                for _ in range(num_layers)
            ]
        )
        self.linear_2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        for block in self.blocks:
            x = block(x)
        x = self.linear_2(x)
        return x


class AddHook(ModelHook):
    def __init__(self, value: int):
        super().__init__()
        self.value = value

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        logger.debug("AddHook pre_forward")
        args = ((x + self.value) if torch.is_tensor(x) else x for x in args)
        return args, kwargs

    def post_forward(self, module, output):
        logger.debug("AddHook post_forward")
        return output


class MultiplyHook(ModelHook):
    def __init__(self, value: int):
        super().__init__()
        self.value = value

    def pre_forward(self, module, *args, **kwargs):
        logger.debug("MultiplyHook pre_forward")
        args = ((x * self.value) if torch.is_tensor(x) else x for x in args)
        return args, kwargs

    def post_forward(self, module, output):
        logger.debug("MultiplyHook post_forward")
        return output

    def __repr__(self):
        return f"MultiplyHook(value={self.value})"


class StatefulAddHook(ModelHook):
    _is_stateful = True

    def __init__(self, value: int):
        super().__init__()
        self.value = value
        self.increment = 0

    def pre_forward(self, module, *args, **kwargs):
        logger.debug("StatefulAddHook pre_forward")
        add_value = self.value + self.increment
        self.increment += 1
        args = ((x + add_value) if torch.is_tensor(x) else x for x in args)
        return args, kwargs

    def reset_state(self, module):
        self.increment = 0


class SkipLayerHook(ModelHook):
    def __init__(self, skip_layer: bool):
        super().__init__()
        self.skip_layer = skip_layer

    def pre_forward(self, module, *args, **kwargs):
        logger.debug("SkipLayerHook pre_forward")
        return args, kwargs

    def custom_forward(self, module, *args, **kwargs):
        logger.debug("SkipLayerHook new_forward")
        if self.skip_layer:
            return args[0]
        return self.fn_ref.original_forward(*args, **kwargs)

    def post_forward(self, module, output):
        logger.debug("SkipLayerHook post_forward")
        return output


class HookTests(unittest.TestCase):
    in_features = 4
    hidden_features = 8
    out_features = 4
    num_layers = 2

    def setUp(self):
        self.model = DummyModel(
            self.in_features, self.hidden_features, self.out_features, self.num_layers
        )

    def tearDown(self):
        super().tearDown()

        del self.model
        gc.collect()

    def get_generator(self):
        return torch.manual_seed(42)

    def test_hook_registry(self):
        registry = ModuleHookRegistry.get_or_create_registry(self.model)
        registry.register_hook(AddHook(1), "add_hook")
        registry.register_hook(MultiplyHook(2), "multiply_hook")

        registry_repr = repr(registry)
        expected_repr = "ModuleHookRegistry(\n  (0) add_hook - AddHook\n  (1) multiply_hook - MultiplyHook(value=2)\n)"

        self.assertEqual(len(registry._hooks), 2)
        self.assertEqual(registry._order, ["add_hook", "multiply_hook"])
        self.assertEqual(registry_repr, expected_repr)

        registry.remove_hook("add_hook")

        self.assertEqual(len(registry._hooks), 1)
        self.assertEqual(registry._order, ["multiply_hook"])

    def test_inference(self):
        registry = ModuleHookRegistry.get_or_create_registry(self.model)
        registry.register_hook(AddHook(1), "add_hook")
        registry.register_hook(MultiplyHook(2), "multiply_hook")

        input = torch.randn(1, 4, generator=self.get_generator())
        output1 = self.model(input).mean().item()

        registry.remove_hook("multiply_hook")
        new_input = input * 2
        output2 = self.model(new_input).mean().item()

        registry.remove_hook("add_hook")
        new_input = input * 2 + 1
        output3 = self.model(new_input).mean().item()

        self.assertAlmostEqual(output1, output2, places=5)
        self.assertAlmostEqual(output1, output3, places=5)
        self.assertAlmostEqual(output2, output3, places=5)

    def test_skip_layer_hook(self):
        registry = ModuleHookRegistry.get_or_create_registry(self.model)
        registry.register_hook(SkipLayerHook(skip_layer=True), "skip_layer_hook")

        input = torch.zeros(1, 4)
        output = self.model(input).mean().item()
        self.assertEqual(output, 0.0)

        registry.remove_hook("skip_layer_hook")
        registry.register_hook(SkipLayerHook(skip_layer=False), "skip_layer_hook")
        output = self.model(input).mean().item()
        self.assertNotEqual(output, 0.0)

import unittest

from unittest.mock import patch

import torch.nn as nn

from omegaconf import OmegaConf

from flagscale.transformations.torch_compile_transformation import TorchCompileTransformation


def _empty_pass_config():
    return OmegaConf.create({})


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.compile_called = False
        self.compile_kwargs = None

    def compile(self, **kwargs):
        self.compile_called = True
        self.compile_kwargs = kwargs


class TestTorchCompileTransformation(unittest.TestCase):
    def test_preflight_returns_false_when_torch_too_old(self):
        transform = TorchCompileTransformation(passes=_empty_pass_config())

        with (
            patch(
                "flagscale.transformations.torch_compile_transformation.is_torch_equal_or_newer",
                return_value=False,
            ) as mock_version,
            patch("flagscale.transformations.torch_compile_transformation.logger") as mock_logger,
        ):
            result = transform.preflight()

        self.assertFalse(result)
        mock_version.assert_called_once_with("2.6.0")
        mock_logger.error.assert_called_once()

    def test_preflight_returns_true_when_version_satisfied(self):
        transform = TorchCompileTransformation(passes=_empty_pass_config())

        with patch(
            "flagscale.transformations.torch_compile_transformation.is_torch_equal_or_newer",
            return_value=True,
        ) as mock_version:
            result = transform.preflight()

        self.assertTrue(result)
        mock_version.assert_called_once_with("2.6.0")

    def test_apply_skips_when_disabled(self):
        module = DummyModule()
        transform = TorchCompileTransformation(
            options={"disable": True}, passes=_empty_pass_config()
        )

        applied = transform.apply(module)

        self.assertTrue(applied)
        self.assertFalse(module.compile_called)

    def test_apply_invokes_compile_with_expected_options(self):
        module = DummyModule()
        options = {"disable": False, "mode": "reduce-overhead", "dynamic": False, "fullgraph": True}
        transform = TorchCompileTransformation(options=options, passes=_empty_pass_config())

        applied = transform.apply(module)

        self.assertTrue(applied)
        self.assertTrue(module.compile_called)
        self.assertIsNotNone(module.compile_kwargs)
        self.assertIn("backend", module.compile_kwargs)
        self.assertTrue(callable(module.compile_kwargs["backend"]))
        self.assertEqual(module.compile_kwargs["mode"], options["mode"])
        self.assertEqual(module.compile_kwargs["dynamic"], options["dynamic"])
        self.assertEqual(module.compile_kwargs["fullgraph"], options["fullgraph"])

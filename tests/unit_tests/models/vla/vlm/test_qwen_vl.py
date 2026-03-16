import unittest

from flagscale.models.vla.registry import VLM_REGISTRY


class TestQwenVLRegistration(unittest.TestCase):
    def test_qwen25_vl_registered(self):
        from flagscale.models.vla.vlm import qwen_vl  # noqa: F401

        self.assertIn("qwen2.5-vl", VLM_REGISTRY)

    def test_qwen3_vl_registered(self):
        from flagscale.models.vla.vlm import qwen_vl  # noqa: F401

        self.assertIn("qwen3-vl", VLM_REGISTRY)

    def test_qwen25_has_required_methods(self):
        from flagscale.models.vla.vlm.qwen_vl import Qwen25VLBackbone

        self.assertTrue(hasattr(Qwen25VLBackbone, "model_config"))
        self.assertTrue(hasattr(Qwen25VLBackbone, "prepare_input"))
        self.assertTrue(hasattr(Qwen25VLBackbone, "forward"))

    def test_qwen3_has_required_methods(self):
        from flagscale.models.vla.vlm.qwen_vl import Qwen3VLBackbone

        self.assertTrue(hasattr(Qwen3VLBackbone, "model_config"))
        self.assertTrue(hasattr(Qwen3VLBackbone, "prepare_input"))
        self.assertTrue(hasattr(Qwen3VLBackbone, "forward"))

import unittest


class TestVLMInit(unittest.TestCase):
    def test_imports(self):
        from flagscale.models.vla.vlm import Qwen3VLBackbone, Qwen25VLBackbone

        self.assertIsNotNone(Qwen25VLBackbone)
        self.assertIsNotNone(Qwen3VLBackbone)

import unittest

from flagscale.models.vla.registry import (
    ACTION_MODEL_REGISTRY,
    VLM_REGISTRY,
    build_action_model,
    build_vlm,
    register_action_model,
    register_vlm,
)


class TestRegistry(unittest.TestCase):
    def test_register_vlm(self):
        @register_vlm("test-vlm")
        class TestVLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        self.assertIn("test-vlm", VLM_REGISTRY)
        vlm = build_vlm("test-vlm", model_id="test")
        self.assertEqual(vlm.kwargs["model_id"], "test")

    def test_register_action_model(self):
        @register_action_model("test-model")
        class TestModel:
            def __init__(self, config):
                self.config = config

        self.assertIn("test-model", ACTION_MODEL_REGISTRY)
        model = build_action_model("test-model", config={"action_dim": 7})
        self.assertEqual(model.config["action_dim"], 7)

    def test_build_unknown_vlm_raises(self):
        with self.assertRaises(ValueError):
            build_vlm("nonexistent-vlm-xyz")

    def test_build_unknown_action_model_raises(self):
        with self.assertRaises(ValueError):
            build_action_model("nonexistent-model-xyz")

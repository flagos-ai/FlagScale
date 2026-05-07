import unittest

import torch


class MockVLM:
    @property
    def config(self):
        return {"hidden_size": 1024}

    def prepare_input(self, batch):
        return batch

    def forward(self, batch, **kwargs):
        return {"hidden_states": (torch.randn(1, 10, 1024),)}


class MockActionModel:
    def forward(self, vlm_output, action_input, **kwargs):
        return {"loss": torch.tensor(0.5)}

    def predict(self, vlm_output, action_input, **kwargs):
        return {"actions": torch.randn(1, 16, 7)}


class TestVLMBackboneProtocol(unittest.TestCase):
    def test_mock_vlm_has_protocol_methods(self):
        vlm = MockVLM()
        self.assertTrue(hasattr(vlm, "config"))
        self.assertTrue(hasattr(vlm, "prepare_input"))
        self.assertTrue(hasattr(vlm, "forward"))

        output = vlm.forward({})
        self.assertIn("hidden_states", output)


class TestActionModelProtocol(unittest.TestCase):
    def test_mock_action_model_has_protocol_methods(self):
        model = MockActionModel()
        self.assertTrue(hasattr(model, "forward"))
        self.assertTrue(hasattr(model, "predict"))

    def test_forward_returns_loss(self):
        model = MockActionModel()
        vlm_output = {"hidden_states": (torch.randn(1, 10, 1024),)}
        action_input = {"actions": torch.randn(1, 16, 7)}

        output = model.forward(vlm_output, action_input)
        self.assertIn("loss", output)

    def test_predict_returns_actions(self):
        model = MockActionModel()
        vlm_output = {"hidden_states": (torch.randn(1, 10, 1024),)}
        action_input = {}

        pred = model.predict(vlm_output, action_input)
        self.assertIn("actions", pred)

import unittest

from flagscale.models.vla.utils import get_vlm_config


class MockConfigDirect:
    hidden_size = 2048
    num_hidden_layers = 28


class MockConfigNested:
    class text_config:
        hidden_size = 1536
        num_hidden_layers = 24


class MockConfigInvalid:
    pass


class TestGetVlmConfig(unittest.TestCase):
    def test_direct_config(self):
        info = get_vlm_config(MockConfigDirect())
        self.assertEqual(info["hidden_size"], 2048)
        self.assertEqual(info["num_hidden_layers"], 28)

    def test_nested_config(self):
        info = get_vlm_config(MockConfigNested())
        self.assertEqual(info["hidden_size"], 1536)
        self.assertEqual(info["num_hidden_layers"], 24)

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            get_vlm_config(MockConfigInvalid())

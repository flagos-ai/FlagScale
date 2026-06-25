import unittest

from flagscale.models.configs.types import FeatureType, PolicyFeature
from flagscale.models.vla.utils import get_vlm_config, reorder_visual_input_features


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


class TestOrderVisualInputFeatures(unittest.TestCase):
    def test_prefers_explicit_image_order(self):
        input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.wrist_image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }

        reordered = reorder_visual_input_features(
            input_features,
            ["observation.images.image", "observation.images.wrist_image"],
        )

        visual_keys = [key for key, ft in reordered.items() if ft.type == FeatureType.VISUAL]
        self.assertEqual(
            visual_keys,
            ["observation.images.image", "observation.images.wrist_image"],
        )
        self.assertIn("observation.state", reordered)

    def test_appends_unlisted_visual_keys(self):
        input_features = {
            "observation.images.left_wrist_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.right_wrist_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
        }

        reordered = reorder_visual_input_features(
            input_features,
            ["observation.images.image"],
        )

        visual_keys = [key for key, ft in reordered.items() if ft.type == FeatureType.VISUAL]
        self.assertEqual(
            visual_keys,
            [
                "observation.images.image",
                "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist_0_rgb",
            ],
        )

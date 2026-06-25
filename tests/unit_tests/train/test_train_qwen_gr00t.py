import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flagscale.models.configs.types import FeatureType
from flagscale.train.train_qwen_gr00t import make_policy


class FakePolicy:
    def __init__(self, config):
        self.config = config
        self.input_features = None
        self.output_features = None
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeMetadata:
    def __init__(self, features):
        self.features = features


def make_image_feature():
    return {
        "dtype": "image",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    }


class TestMakePolicyImageOrder(unittest.TestCase):
    def test_respects_explicit_image_key_order(self):
        ds_meta = FakeMetadata(
            {
                "observation.images.wrist_image": make_image_feature(),
                "observation.images.image": make_image_feature(),
                "observation.state": {"dtype": "float32", "shape": (7,)},
                "action": {"dtype": "float32", "shape": (7,)},
            }
        )
        config = SimpleNamespace(
            data=SimpleNamespace(
                image_key_order=[
                    "observation.images.image",
                    "observation.images.wrist_image",
                ]
            )
        )

        with patch("flagscale.train.train_qwen_gr00t.QwenGr00t", FakePolicy):
            policy = make_policy(config=config, ds_meta=ds_meta)

        visual_keys = [
            key for key, ft in policy.input_features.items() if ft.type == FeatureType.VISUAL
        ]
        self.assertEqual(
            visual_keys,
            ["observation.images.image", "observation.images.wrist_image"],
        )
        self.assertEqual(policy.device, "cuda")

    def test_keeps_dataset_visual_order_without_image_key_order(self):
        ds_meta = FakeMetadata(
            {
                "observation.images.wrist_image": make_image_feature(),
                "observation.images.image": make_image_feature(),
                "observation.state": {"dtype": "float32", "shape": (7,)},
                "action": {"dtype": "float32", "shape": (7,)},
            }
        )
        config = SimpleNamespace(data=SimpleNamespace())

        with patch("flagscale.train.train_qwen_gr00t.QwenGr00t", FakePolicy):
            policy = make_policy(config=config, ds_meta=ds_meta)

        visual_keys = [
            key for key, ft in policy.input_features.items() if ft.type == FeatureType.VISUAL
        ]
        self.assertEqual(
            visual_keys,
            ["observation.images.wrist_image", "observation.images.image"],
        )

    def test_appends_visual_keys_not_listed_in_config(self):
        ds_meta = FakeMetadata(
            {
                "observation.images.left_wrist_0_rgb": make_image_feature(),
                "observation.images.image": make_image_feature(),
                "observation.images.right_wrist_0_rgb": make_image_feature(),
                "action": {"dtype": "float32", "shape": (7,)},
            }
        )
        config = SimpleNamespace(data=SimpleNamespace(image_key_order=["observation.images.image"]))

        with patch("flagscale.train.train_qwen_gr00t.QwenGr00t", FakePolicy):
            policy = make_policy(config=config, ds_meta=ds_meta)

        visual_keys = [
            key for key, ft in policy.input_features.items() if ft.type == FeatureType.VISUAL
        ]
        self.assertEqual(
            visual_keys,
            [
                "observation.images.image",
                "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist_0_rgb",
            ],
        )

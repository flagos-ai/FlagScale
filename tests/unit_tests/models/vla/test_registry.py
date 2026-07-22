# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

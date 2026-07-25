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

from flagscale.models.vla.registry import VLM_REGISTRY


class TestQwenVLRegistration(unittest.TestCase):
    def test_qwen25_vl_registered(self):
        from flagscale.models.vla.vlm import qwenvl_backbone  # noqa: F401

        self.assertIn("qwen2.5-vl", VLM_REGISTRY)

    def test_qwen3_vl_registered(self):
        from flagscale.models.vla.vlm import qwenvl_backbone  # noqa: F401

        self.assertIn("qwen3-vl", VLM_REGISTRY)

    def test_qwen25_has_required_methods(self):
        from flagscale.models.vla.vlm.qwenvl_backbone import Qwen25VLBackbone

        self.assertTrue(hasattr(Qwen25VLBackbone, "model_config"))
        self.assertTrue(hasattr(Qwen25VLBackbone, "prepare_input"))
        self.assertTrue(hasattr(Qwen25VLBackbone, "forward"))

    def test_qwen3_has_required_methods(self):
        from flagscale.models.vla.vlm.qwenvl_backbone import Qwen3VLBackbone

        self.assertTrue(hasattr(Qwen3VLBackbone, "model_config"))
        self.assertTrue(hasattr(Qwen3VLBackbone, "prepare_input"))
        self.assertTrue(hasattr(Qwen3VLBackbone, "forward"))

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

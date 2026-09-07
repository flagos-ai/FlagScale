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

"""Unit tests for the Qwen3.5 checkpoint converter PP sharding rules.

Guards two contracts:

1. The converter's default (no explicit first/last) uneven-PP layer split must
   match the training-side allocation
   (``compute_qwen35_pipeline_layer_split``): the remainder goes to the FIRST
   stage (32 layers with PP3 -> [12, 10, 10]).  A divergent default silently
   shifts/drops decoder layers when merging a training-written checkpoint
   (``_convert_llm_meg2hf`` guards every layer lookup with ``if mk in sd``).
2. ``Config`` rejects grid-layout MIMO yamls (``mimo_module_specs`` /
   non-colocated ``mimo_layout``) instead of silently converting with the
   pinned legacy TP/PP fields.
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONVERTER_DIR = os.path.join(_REPO_ROOT, "tools", "checkpoint", "qwen35")
if _CONVERTER_DIR not in sys.path:
    sys.path.insert(0, _CONVERTER_DIR)

from qwen35.config import Config
from qwen35.sharding import merge_pp_layers, split_pp_layers

from flagscale.models.mimo.bridge.recipe.qwen35 import (
    compute_qwen35_pipeline_layer_split,
)


def _make_cfg(num_layers, pp, first=None, last=None):
    cfg = Config.__new__(Config)
    cfg.pp = pp
    cfg.num_layers = num_layers
    cfg.decoder_first_pipeline_num_layers = first
    cfg.decoder_last_pipeline_num_layers = last
    return cfg


class TestPPLayerCounts:
    @pytest.mark.parametrize("num_layers", [32, 33, 36, 48])
    @pytest.mark.parametrize("pp", [1, 2, 3, 4, 6, 8])
    def test_default_split_matches_training_rule(self, num_layers, pp):
        if num_layers < pp:
            pytest.skip("every stage needs at least one layer")
        assert _make_cfg(num_layers, pp).pp_layer_counts == (
            compute_qwen35_pipeline_layer_split(num_layers, pp)
        )

    def test_explicit_first_last_preserved(self):
        assert _make_cfg(32, 3, first=12, last=10).pp_layer_counts == [12, 10, 10]
        assert _make_cfg(32, 2, first=20, last=12).pp_layer_counts == [20, 12]

    def test_first_only_branch(self):
        assert _make_cfg(32, 3, first=12).pp_layer_counts == [12, 10, 10]


class TestPPRoundTrip:
    @pytest.mark.parametrize("num_layers,pp", [(32, 3), (32, 6)])
    def test_split_merge_roundtrip(self, num_layers, pp):
        cfg = _make_cfg(num_layers, pp)
        full_sd = {
            f"language_model.decoder.layers.{i}.w": torch.full((1,), float(i))
            for i in range(num_layers)
        }
        per_rank = split_pp_layers(full_sd, cfg)
        assert [len(sd) for sd in per_rank] == compute_qwen35_pipeline_layer_split(num_layers, pp)
        merged = merge_pp_layers(per_rank, cfg)
        assert set(merged) == set(full_sd)
        for key, value in full_sd.items():
            assert torch.equal(merged[key], value)


class TestGridYamlRejection:
    def test_grid_yaml_rejected(self):
        grid_yaml = os.path.join(
            _REPO_ROOT, "examples", "qwen35", "conf", "train", "4b_mimo_grid_2p6.yaml"
        )
        with pytest.raises(ValueError, match="[Gg]rid"):
            Config(grid_yaml)

    def test_colocated_mimo_yaml_accepted(self):
        legacy_yaml = os.path.join(
            _REPO_ROOT, "examples", "qwen35", "conf", "train", "4b_mimo.yaml"
        )
        cfg = Config(legacy_yaml)
        assert cfg.tp == 2
        assert cfg.vision_tp == 1

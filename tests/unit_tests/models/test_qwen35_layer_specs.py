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

"""CPU unit tests for flagscale.models.megatron.qwen35.layer_specs.

Validates the per-pipeline-stage slicing of the hybrid GDN + Attention
language layer spec:

- ``get_qwen35_language_model_spec(config, pp_rank=X)`` must build only the
  layers of PP stage X of the *language* module (grid MIMO mode): e.g. a
  24-layer model with language PP2 yields 12 layer specs per stage at global
  offsets 0 and 12.  Before the explicit ``pp_rank`` threading, grid mode
  built the spec while the global parallel state was TP=1/PP=1, so every
  stage sliced all 24 layer specs - duplicated weights and overlapping
  checkpoint keys (``decoder.layers.12..23`` saved by both stages).
- ``pp_rank=None`` must keep the non-grid fallback to the global parallel
  state (base / colocated paths).

The returned ``ModuleSpec`` list does not retain global layer numbers, so
non-overlapping global coverage is proven two ways:

1. by mocking ``get_num_layers_to_build`` / ``get_transformer_layer_offset``
   (the MCore helpers that decide the stage slicing) and asserting they are
   called with the explicit ``pp_rank``;
2. semantically, by matching each stage's GDN/Attention pattern (every 4th
   global layer is standard attention, ``linear_attention_freq=4``) against
   the global layer range the real helpers assign to that stage - e.g. PP2
   over 24 layers must give stage 0 = global [0, 12) and stage 1 = global
   [12, 24), together partitioning [0, 24) without overlap or gap.

Run (inside the container, from the FlagScale repo root):

    source /root/miniconda3/bin/activate flagscale
    PYTHONPATH=/workspace/multimodal/Megatron-LM-FL python -m pytest \
        tests/unit_tests/models/test_qwen35_layer_specs.py
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from flagscale.models.megatron.qwen35 import layer_specs as layer_specs_mod
from flagscale.models.megatron.qwen35.attention import Qwen35SelfAttention
from flagscale.models.megatron.qwen35.layer_specs import (
    get_qwen35_language_model_spec,
    get_qwen35_mtp_block_spec,
)
from flagscale.models.megatron.qwen35.transformer_config import Qwen35TransformerConfig
from flagscale.models.mimo.bridge.recipe.qwen35 import compute_qwen35_pipeline_layer_split

LINEAR_ATTENTION_FREQ = 4


def make_language_config(num_layers=24, pp_size=2, first=None, last=None):
    """Small Qwen35TransformerConfig with the grid-branch pipeline fields set."""
    config = Qwen35TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=256,
        use_cpu_initialization=True,
    )
    # Mirrors the grid branch in train_qwen35.model_provider: the language
    # module config gets the language PP size, and uneven splits are encoded
    # via the explicit first/last stage layer counts (middle stages split
    # evenly by MCore).
    config.pipeline_model_parallel_size = pp_size
    config.num_layers_in_first_pipeline_stage = first
    config.num_layers_in_last_pipeline_stage = last
    return config


def grid_branch_config(num_layers, pp_size):
    """Config exactly as the grid branch derives it (compute split -> fields)."""
    split = compute_qwen35_pipeline_layer_split(num_layers, pp_size)
    if len(set(split)) > 1:
        return make_language_config(
            num_layers=num_layers, pp_size=pp_size, first=split[0], last=split[-1]
        )
    return make_language_config(num_layers=num_layers, pp_size=pp_size)


def assert_stage_covers_global_range(testcase, spec, offset, count):
    """Assert ``spec`` holds exactly ``count`` layers of global [offset, offset+count).

    A layer at global index ``g`` is standard attention (Qwen35SelfAttention)
    iff ``(g + 1) % LINEAR_ATTENTION_FREQ == 0`` and GDN otherwise; the specs
    themselves carry no global index, so matching this pattern pins each spec
    to its global layer.
    """
    testcase.assertEqual(len(spec.layer_specs), count)
    for j, layer_spec in enumerate(spec.layer_specs):
        global_index = offset + j
        is_attention = layer_spec.submodules.self_attention.module is Qwen35SelfAttention
        testcase.assertEqual(
            is_attention,
            (global_index + 1) % LINEAR_ATTENTION_FREQ == 0,
            f"stage spec {j} does not match global layer {global_index}",
        )
        testcase.assertIs(
            layer_spec.submodules.self_attention.module,
            Qwen35SelfAttention if is_attention else GatedDeltaNet,
            f"stage spec {j} has an unexpected attention module",
        )


class TestQwen35LanguageModelSpecPipelineSlicing(unittest.TestCase):
    """Stage slicing with an explicit ``pp_rank`` (grid MIMO mode)."""

    def test_pp2_stages_build_12_layers_each_non_overlapping(self):
        config = make_language_config(num_layers=24, pp_size=2)

        # Real MCore stage math for the mutated language config.
        self.assertEqual(get_num_layers_to_build(config, pp_rank=0), 12)
        self.assertEqual(get_num_layers_to_build(config, pp_rank=1), 12)
        self.assertEqual(get_transformer_layer_offset(config, pp_rank=0), 0)
        self.assertEqual(get_transformer_layer_offset(config, pp_rank=1), 12)

        spec0 = get_qwen35_language_model_spec(config, pp_rank=0)
        spec1 = get_qwen35_language_model_spec(config, pp_rank=1)

        # Each stage builds exactly its own half.
        self.assertEqual(len(spec0.layer_specs), 12)
        self.assertEqual(len(spec1.layer_specs), 12)
        # Non-overlapping global coverage: stage 0 == layers 0..11, stage 1
        # == layers 12..23 (this was the bug: with the spec built under the
        # global PP=1 parallel state, both stages sliced all 24 specs).
        assert_stage_covers_global_range(self, spec0, offset=0, count=12)
        assert_stage_covers_global_range(self, spec1, offset=12, count=12)

    def test_pp3_uneven_stages_12_10_10_with_offsets_0_12_22(self):
        # 32 layers with PP3: first stage gets base+remainder (12), middle
        # and last 10 each (mirrors compute_qwen35_pipeline_layer_split).
        config = make_language_config(num_layers=32, pp_size=3, first=12, last=10)
        expected = [(0, 12, 0), (1, 10, 12), (2, 10, 22)]
        for pp_rank, count, offset in expected:
            with self.subTest(pp_rank=pp_rank):
                self.assertEqual(get_num_layers_to_build(config, pp_rank=pp_rank), count)
                self.assertEqual(get_transformer_layer_offset(config, pp_rank=pp_rank), offset)
                assert_stage_covers_global_range(
                    self,
                    get_qwen35_language_model_spec(config, pp_rank=pp_rank),
                    offset=offset,
                    count=count,
                )

    def test_grid_branch_splits_partition_global_layers_for_all_supported_pp(self):
        # End-to-end over the same derivation the grid branch uses
        # (compute_qwen35_pipeline_layer_split -> first/last config fields)
        # for every language PP supported this stage: PP1/2/3/6 over the 24
        # and 32-layer configs.  Each stage must slice its own layer range
        # and the ranges must partition [0, num_layers) without overlap/gap.
        for num_layers, pp_size in [(24, 1), (24, 2), (24, 3), (24, 6), (32, 3), (32, 6)]:
            with self.subTest(num_layers=num_layers, pp_size=pp_size):
                config = grid_branch_config(num_layers, pp_size)
                ranges = []
                for pp_rank in range(pp_size):
                    count = get_num_layers_to_build(config, pp_rank=pp_rank)
                    offset = get_transformer_layer_offset(config, pp_rank=pp_rank)
                    ranges.append((offset, offset + count))
                    assert_stage_covers_global_range(
                        self,
                        get_qwen35_language_model_spec(config, pp_rank=pp_rank),
                        offset=offset,
                        count=count,
                    )
                covered = sorted(g for start, end in ranges for g in range(start, end))
                self.assertEqual(covered, list(range(num_layers)))

    def test_pp_rank_propagated_to_mcore_offset_and_count_helpers(self):
        # The spec objects do not retain global numbering; prove the explicit
        # pp_rank reaches the MCore helpers that decide the slice.
        config = make_language_config(num_layers=24, pp_size=2)
        spec_module = "megatron.core.models.gpt.experimental_attention_variant_module_specs"
        with (
            mock.patch(
                f"{spec_module}.get_num_layers_to_build", wraps=get_num_layers_to_build
            ) as count_mock,
            mock.patch(
                f"{spec_module}.get_transformer_layer_offset", wraps=get_transformer_layer_offset
            ) as offset_mock,
        ):
            spec = get_qwen35_language_model_spec(config, pp_rank=1)

        count_mock.assert_called_with(config, vp_stage=None, pp_rank=1)
        offset_mock.assert_called_with(config, vp_stage=None, pp_rank=1)
        self.assertEqual(len(spec.layer_specs), 12)

    def test_none_pp_rank_falls_back_to_global_parallel_state(self):
        # Non-grid behavior: pp_rank=None (base / colocated paths) must
        # keep slicing by the global parallel state's PP rank - here a rank
        # whose global PP rank is 1 gets the second stage's 12 layers.
        config = make_language_config(num_layers=24, pp_size=2)
        with (
            mock.patch(
                "megatron.core.transformer.transformer_layer.parallel_state"
                ".get_pipeline_model_parallel_rank",
                return_value=1,
            ),
            mock.patch(
                "megatron.core.transformer.transformer_block.parallel_state"
                ".get_pipeline_model_parallel_rank",
                return_value=1,
            ),
        ):
            spec = get_qwen35_language_model_spec(config)
        assert_stage_covers_global_range(self, spec, offset=12, count=12)

    def test_mtp_block_spec_uses_unpatched_spec_without_pp_rank(self):
        # get_qwen35_mtp_block_spec must keep calling the language spec with
        # patch=False and no explicit pp_rank (MTP uses vanilla attention and
        # the global-parallel-state fallback, unchanged by this fix).
        config = make_language_config(num_layers=24, pp_size=2)
        args = SimpleNamespace(mtp_num_layers=1, transformer_impl="transformer_engine")
        with (
            mock.patch.object(
                layer_specs_mod, "get_qwen35_language_model_spec", return_value="unpatched-spec"
            ) as inner_mock,
            mock.patch.object(
                layer_specs_mod, "get_gpt_mtp_block_spec", return_value="mtp-block-spec"
            ) as mtp_mock,
        ):
            result = get_qwen35_mtp_block_spec(args, config)

        self.assertEqual(result, "mtp-block-spec")
        inner_mock.assert_called_once_with(config, patch=False)
        mtp_mock.assert_called_once_with(config, "unpatched-spec", use_transformer_engine=True)


if __name__ == "__main__":
    unittest.main()

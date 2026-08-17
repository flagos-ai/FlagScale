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

"""CPU unit tests for flagscale.models.mimo.bridge.providers.qwen35.

The heavy model constructors (``Qwen35LanguageModule``, ``Qwen3VisionModel``)
are mocked; the REAL ``megatron.core.models.mimo.MimoModel`` machinery is
exercised to validate:

- rank-role derivation from a (nullable) ``module_to_grid_map``,
- role-based module selection (encoder-only / language-only ranks),
- pg_collection injection into the language spec and the images submodule.

No process-group initialization is required: ``torch.distributed.get_rank``
is mocked for the non-colocated role paths, and ``WORLD_SIZE`` is injected via
the environment for the ``HyperCommGrid`` helper test.

Run (inside the container, from the FlagScale repo root):

    source /root/miniconda3/bin/activate flagscale
    PYTHONPATH=/workspace/multimodal/Megatron-LM-FL python -m pytest \
        tests/unit_tests/models/test_qwen35_grid_mimo_model.py
"""

import os
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ``flagscale.models.mimo`` imports cleanly without a backend shim
# (``mimo_optimizer`` pulls ``unwrap_model`` from ``megatron.core.utils``),
# so no global re-export patching of ``megatron.training.utils`` is needed.
import torch
import torch.nn as nn

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from flagscale.models.megatron.qwen35.transformer_config import Qwen35TransformerConfig
from flagscale.models.mimo.bridge.providers import qwen35 as grid_mimo
from flagscale.models.mimo.bridge.providers.qwen35 import (
    LANGUAGE_MODULE_NAME,
    MODULE_GRID_DIM_NAMES,
    VISION_ENCODER_NAME,
    VISION_MODALITY_NAME,
    Qwen35GridMIMOModel,
    Qwen35VisionSubmodules,
    build_qwen35_images_submodule_spec,
    build_qwen35_language_model_spec,
    build_qwen35_mimo_config,
    build_qwen35_module_to_grid_map,
    qwen35_grid_mimo_model_provider,
)

IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057
VISION_START_TOKEN_ID = 248053

WORLD_SIZE = 8


class FakeVisionModel(nn.Module):
    """Mock for ``Qwen3VisionModel``; records constructor kwargs."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.projection = nn.Linear(4, 4)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("mocked")


class FakeLanguageModel(nn.Module):
    """Mock for ``Qwen35LanguageModule``; records constructor kwargs."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, *args, **kwargs):
        raise NotImplementedError("mocked")


class FakeGrid:
    """Minimal ``HyperCommGrid``-compatible stub (rank range only)."""

    def __init__(self, rank_offset=0, size=1, dim_names=None, shape=None):
        self.rank_offset = rank_offset
        self.size = size
        self.dim_names = list(dim_names or [])
        self.shape = list(shape or [])

    def get_pg(self, dims):
        raise KeyError(f"no process group for {dims} in FakeGrid")


def make_language_config():
    return Qwen35TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=256,
        use_cpu_initialization=True,
    )


def make_vision_config():
    return TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=256,
        use_cpu_initialization=True,
    )


def make_layer_spec():
    return ModuleSpec(module=nn.Identity, params={})


class Qwen35GridMimoModelTestBase(unittest.TestCase):
    """Shared harness: patch heavy constructors + dist rank; silence warnings."""

    def setUp(self):
        self.language_config = make_language_config()
        self.vision_config = make_vision_config()

        self.language_patcher = mock.patch.object(
            grid_mimo, "Qwen35LanguageModule", FakeLanguageModel
        )
        self.vision_patcher = mock.patch.object(grid_mimo, "Qwen3VisionModel", FakeVisionModel)
        self.rank_patcher = mock.patch("torch.distributed.get_rank", return_value=0)
        for patcher in (self.language_patcher, self.vision_patcher, self.rank_patcher):
            patcher.start()
            self.addCleanup(patcher.stop)

        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.filterwarnings("ignore")
        self.addCleanup(self._warnings.__exit__)

    def build_provider_model(self, **kwargs):
        return qwen35_grid_mimo_model_provider(
            language_transformer_config=self.language_config,
            language_transformer_layer_spec=make_layer_spec(),
            language_vocab_size=512,
            language_max_sequence_length=2048,
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=make_layer_spec(),
            vision_projection_config=self.vision_config,
            vision_projection_layer_spec=make_layer_spec(),
            **kwargs,
        )


class TestQwen35GridMimoComponentNames(Qwen35GridMimoModelTestBase):
    """Component naming contract: "images" modality + "language" module."""

    def test_component_names(self):
        self.assertEqual(VISION_MODALITY_NAME, "images")
        self.assertEqual(VISION_ENCODER_NAME, "qwen3_vit")
        self.assertEqual(LANGUAGE_MODULE_NAME, "language")
        self.assertEqual(LANGUAGE_MODULE_NAME, MIMO_LANGUAGE_MODULE_KEY)


class TestQwen35GridMimoSpecs(Qwen35GridMimoModelTestBase):
    """Spec construction: classes, params, nullable pg_collection, grid keys."""

    def test_language_spec_module_and_params(self):
        spec = build_qwen35_language_model_spec(
            config=self.language_config,
            transformer_layer_spec=make_layer_spec(),
            vocab_size=512,
            max_sequence_length=2048,
            pg_collection=None,
        )
        self.assertIs(spec.module, grid_mimo.Qwen35LanguageModule)
        params = spec.params
        self.assertIs(params["config"], self.language_config)
        self.assertEqual(params["vocab_size"], 512)
        self.assertEqual(params["max_sequence_length"], 2048)
        self.assertEqual(params["position_embedding_type"], "mrope")
        self.assertEqual(params["rotary_percent"], 0.25)
        self.assertEqual(params["rotary_base"], 10000000)
        self.assertFalse(params["rope_scaling"])
        self.assertTrue(params["parallel_output"])
        self.assertTrue(params["pre_process"])
        self.assertTrue(params["post_process"])
        self.assertIsNone(params["mtp_block_spec"])
        self.assertIsNone(params["vp_stage"])
        self.assertIsNone(params["pg_collection"])

    def test_images_spec_module_and_params(self):
        spec = build_qwen35_images_submodule_spec(
            transformer_config=self.vision_config,
            transformer_layer_spec=make_layer_spec(),
            projection_config=self.vision_config,
            projection_layer_spec=make_layer_spec(),
            pg_collection=None,
        )
        self.assertIs(spec.module, Qwen35VisionSubmodules)
        encoder_spec = spec.submodules["encoders"][VISION_ENCODER_NAME]
        self.assertIs(encoder_spec.module, grid_mimo.Qwen3VisionModel)
        self.assertIs(encoder_spec.params["transformer_config"], self.vision_config)
        self.assertIs(encoder_spec.params["projection_config"], self.vision_config)
        self.assertEqual(encoder_spec.params["projection_type"], "mlp")
        self.assertTrue(encoder_spec.params["pre_process"])
        self.assertTrue(encoder_spec.params["post_process"])
        # No input projections: the ViT projector already maps to LM hidden size.
        self.assertNotIn("input_projections", spec.submodules)
        self.assertIsNone(spec.params["pg_collection"])

    def test_pg_collection_injected_into_specs(self):
        pg = SimpleNamespace()
        language_spec = build_qwen35_language_model_spec(
            config=self.language_config,
            transformer_layer_spec=make_layer_spec(),
            vocab_size=512,
            max_sequence_length=2048,
            pg_collection=pg,
        )
        self.assertIs(language_spec.params["pg_collection"], pg)

        images_spec = build_qwen35_images_submodule_spec(
            transformer_config=self.vision_config,
            transformer_layer_spec=make_layer_spec(),
            projection_config=self.vision_config,
            projection_layer_spec=make_layer_spec(),
            pg_collection=pg,
        )
        self.assertIs(images_spec.params["pg_collection"], pg)

    def test_mimo_config_default_special_token_ids_from_language_config(self):
        config = build_qwen35_mimo_config(
            language_transformer_config=self.language_config,
            language_transformer_layer_spec=make_layer_spec(),
            language_vocab_size=512,
            language_max_sequence_length=2048,
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=make_layer_spec(),
            vision_projection_config=self.vision_config,
            vision_projection_layer_spec=make_layer_spec(),
        )
        self.assertEqual(config.special_token_ids, {VISION_MODALITY_NAME: IMAGE_TOKEN_ID})
        self.assertIsNone(config.module_to_grid_map)
        self.assertEqual(config.kv_format, "sbhd")

    def test_mimo_config_special_token_ids_override_and_kv_format(self):
        config = build_qwen35_mimo_config(
            language_transformer_config=self.language_config,
            language_transformer_layer_spec=make_layer_spec(),
            language_vocab_size=512,
            language_max_sequence_length=2048,
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=make_layer_spec(),
            vision_projection_config=self.vision_config,
            vision_projection_layer_spec=make_layer_spec(),
            special_token_ids={"images": 12345},
            kv_format="thd",
        )
        self.assertEqual(config.special_token_ids, {"images": 12345})
        self.assertEqual(config.kv_format, "thd")

    def test_mimo_config_grid_map_passthrough_and_key_validation(self):
        grids = {
            VISION_MODALITY_NAME: FakeGrid(0, WORLD_SIZE),
            LANGUAGE_MODULE_NAME: FakeGrid(0, WORLD_SIZE),
        }
        config = build_qwen35_mimo_config(
            language_transformer_config=self.language_config,
            language_transformer_layer_spec=make_layer_spec(),
            language_vocab_size=512,
            language_max_sequence_length=2048,
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=make_layer_spec(),
            vision_projection_config=self.vision_config,
            vision_projection_layer_spec=make_layer_spec(),
            module_to_grid_map=grids,
        )
        self.assertIs(config.module_to_grid_map, grids)

        # Wrong grid keys must be rejected by MimoModelConfig.__post_init__.
        with self.assertRaises(ValueError):
            build_qwen35_mimo_config(
                language_transformer_config=self.language_config,
                language_transformer_layer_spec=make_layer_spec(),
                language_vocab_size=512,
                language_max_sequence_length=2048,
                vision_transformer_config=self.vision_config,
                vision_transformer_layer_spec=make_layer_spec(),
                vision_projection_config=self.vision_config,
                vision_projection_layer_spec=make_layer_spec(),
                module_to_grid_map={VISION_MODALITY_NAME: FakeGrid(), "vision": FakeGrid()},
            )

    def test_module_to_grid_map_helper(self):
        parallelisms = {
            LANGUAGE_MODULE_NAME: SimpleNamespace(
                tensor_model_parallel_size=2,
                context_parallel_size=1,
                data_parallel_size=2,
                expert_model_parallel_size=1,
                pipeline_model_parallel_size=2,
                rank_offset=0,
            ),
            VISION_MODALITY_NAME: SimpleNamespace(
                tensor_model_parallel_size=1,
                context_parallel_size=1,
                data_parallel_size=8,
                expert_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                rank_offset=0,
            ),
        }
        with mock.patch.dict(os.environ, {"WORLD_SIZE": str(WORLD_SIZE)}):
            grids = build_qwen35_module_to_grid_map(parallelisms)
        self.assertEqual(list(grids.keys()), [LANGUAGE_MODULE_NAME, VISION_MODALITY_NAME])
        self.assertIsInstance(grids[LANGUAGE_MODULE_NAME], HyperCommGrid)
        self.assertEqual(grids[LANGUAGE_MODULE_NAME].shape, [2, 1, 2, 1, 2])
        self.assertEqual(grids[LANGUAGE_MODULE_NAME].dim_names, list(MODULE_GRID_DIM_NAMES))
        self.assertEqual(grids[LANGUAGE_MODULE_NAME].rank_offset, 0)
        self.assertEqual(grids[VISION_MODALITY_NAME].shape, [1, 1, 8, 1, 1])


class TestQwen35GridMimoModelConstruction(Qwen35GridMimoModelTestBase):
    """Real MimoModel construction: rank role, module selection, pg injection."""

    def test_no_grids_builds_colocated_role_and_both_modules(self):
        model = self.build_provider_model()

        self.assertIsInstance(model, Qwen35GridMIMOModel)
        self.assertIs(model.config, self.language_config)
        self.assertEqual(model.role.mode, ModuleLayout.COLOCATED)
        self.assertTrue(model.role.has_language_module)
        self.assertTrue(model.role.has_modality_modules)

        # Language module built from the spec, with nullable pg_collection.
        self.assertIsInstance(model.language_model, FakeLanguageModel)
        self.assertIs(model.language_model.kwargs["config"], self.language_config)
        self.assertEqual(model.language_model.kwargs["vocab_size"], 512)
        self.assertIsNone(model.language_model.kwargs["pg_collection"])

        # Images submodule built from the spec, with the mocked ViT encoder.
        self.assertIn(VISION_MODALITY_NAME, model.modality_submodules)
        images = model.modality_submodules[VISION_MODALITY_NAME]
        self.assertIsInstance(images, Qwen35VisionSubmodules)
        self.assertIn(VISION_ENCODER_NAME, images.encoders)
        self.assertIsInstance(images.encoders[VISION_ENCODER_NAME], FakeVisionModel)
        vision_kwargs = images.encoders[VISION_ENCODER_NAME].kwargs
        self.assertIs(vision_kwargs["transformer_config"], self.vision_config)
        self.assertIs(vision_kwargs["projection_config"], self.vision_config)
        self.assertEqual(vision_kwargs["projection_type"], "mlp")
        self.assertIsNone(images.pg_collection)

        self.assertEqual(model.special_token_ids, {VISION_MODALITY_NAME: IMAGE_TOKEN_ID})

    def test_pg_collection_injected_into_built_modules(self):
        pg = SimpleNamespace(tp="TP", cp="CP")
        model = self.build_provider_model(pg_collection=pg)

        self.assertIs(model.language_model.kwargs["pg_collection"], pg)
        images = model.modality_submodules[VISION_MODALITY_NAME]
        self.assertIs(images.pg_collection, pg)

    def test_encoder_only_rank_selects_images_only(self):
        grids = {
            VISION_MODALITY_NAME: FakeGrid(0, WORLD_SIZE),
            LANGUAGE_MODULE_NAME: FakeGrid(WORLD_SIZE, WORLD_SIZE),
        }
        model = self.build_provider_model(module_to_grid_map=grids)

        self.assertEqual(model.role.mode, ModuleLayout.NON_COLOCATED)
        self.assertTrue(model.role.has_modality_modules)
        self.assertFalse(model.role.has_language_module)
        self.assertIsNone(model.language_model)
        self.assertIn(VISION_MODALITY_NAME, model.modality_submodules)

    def test_language_only_rank_selects_language_only(self):
        grids = {
            VISION_MODALITY_NAME: FakeGrid(WORLD_SIZE, WORLD_SIZE),
            LANGUAGE_MODULE_NAME: FakeGrid(0, WORLD_SIZE),
        }
        model = self.build_provider_model(module_to_grid_map=grids)

        self.assertEqual(model.role.mode, ModuleLayout.NON_COLOCATED)
        self.assertFalse(model.role.has_modality_modules)
        self.assertTrue(model.role.has_language_module)
        self.assertIsInstance(model.language_model, FakeLanguageModel)
        self.assertNotIn(VISION_MODALITY_NAME, model.modality_submodules)

    def test_colocated_grids_spanning_same_ranks(self):
        grids = {
            VISION_MODALITY_NAME: FakeGrid(0, WORLD_SIZE),
            LANGUAGE_MODULE_NAME: FakeGrid(0, WORLD_SIZE),
        }
        model = self.build_provider_model(module_to_grid_map=grids)

        self.assertEqual(model.role.mode, ModuleLayout.COLOCATED)
        self.assertTrue(model.role.has_language_module)
        self.assertTrue(model.role.has_modality_modules)
        self.assertIsInstance(model.language_model, FakeLanguageModel)
        self.assertIn(VISION_MODALITY_NAME, model.modality_submodules)
        # No TP/DP dims on the stub grids -> no colocated bridge communicators.
        self.assertEqual(model.colocated_comms, {})

    def test_provider_accepts_prebuilt_mimo_infra(self):
        pg = SimpleNamespace(tp="TP", cp="CP")
        infra = SimpleNamespace(
            module_to_grid_map={
                VISION_MODALITY_NAME: FakeGrid(WORLD_SIZE, WORLD_SIZE),
                LANGUAGE_MODULE_NAME: FakeGrid(0, WORLD_SIZE),
            },
            module_to_pg_collection={
                LANGUAGE_MODULE_NAME: pg,
                VISION_MODALITY_NAME: None,
            },
        )
        model = qwen35_grid_mimo_model_provider(
            language_transformer_config=self.language_config,
            language_transformer_layer_spec=make_layer_spec(),
            language_vocab_size=512,
            language_max_sequence_length=2048,
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=make_layer_spec(),
            vision_projection_config=self.vision_config,
            vision_projection_layer_spec=make_layer_spec(),
            mimo_infra=infra,
        )
        # Infra supplies both the grid map and the language pg_collection.
        self.assertEqual(model.role.mode, ModuleLayout.NON_COLOCATED)
        self.assertTrue(model.role.has_language_module)
        self.assertFalse(model.role.has_modality_modules)
        self.assertIsInstance(model.language_model, FakeLanguageModel)
        self.assertIs(model.language_model.kwargs["pg_collection"], pg)

    def test_provider_encoder_only_rank_gets_vision_pg_collection(self):
        """Encoder-only ranks must get the VISION collection in the images spec.

        Previously the infra path threaded only the language collection into
        both specs: on an encoder-only rank the language collection is None,
        so the images submodule (the one module the rank hosts) was built
        with pg_collection=None and fell back to the *global* parallel state
        for checkpoint metadata (wrong dp_cp_group in vision save/load).
        """
        vision_pg = SimpleNamespace(tp="VTP", cp="VCP")
        infra = SimpleNamespace(
            module_to_grid_map={
                VISION_MODALITY_NAME: FakeGrid(0, WORLD_SIZE),
                LANGUAGE_MODULE_NAME: FakeGrid(WORLD_SIZE, WORLD_SIZE),
            },
            module_to_pg_collection={
                LANGUAGE_MODULE_NAME: None,
                VISION_MODALITY_NAME: vision_pg,
            },
        )
        model = qwen35_grid_mimo_model_provider(
            language_transformer_config=self.language_config,
            language_transformer_layer_spec=make_layer_spec(),
            language_vocab_size=512,
            language_max_sequence_length=2048,
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=make_layer_spec(),
            vision_projection_config=self.vision_config,
            vision_projection_layer_spec=make_layer_spec(),
            mimo_infra=infra,
        )
        self.assertEqual(model.role.mode, ModuleLayout.NON_COLOCATED)
        self.assertFalse(model.role.has_language_module)
        self.assertTrue(model.role.has_modality_modules)
        self.assertIsNone(model.language_model)
        images = model.modality_submodules[VISION_MODALITY_NAME]
        self.assertIs(images.pg_collection, vision_pg)

    def test_provider_direct_path_threads_separate_collections(self):
        """Without infra, explicit per-spec collections are kept distinct."""
        language_pg = SimpleNamespace(tp="LTP", cp="LCP")
        vision_pg = SimpleNamespace(tp="VTP", cp="VCP")
        model = self.build_provider_model(pg_collection=language_pg, images_pg_collection=vision_pg)
        self.assertIs(model.language_model.kwargs["pg_collection"], language_pg)
        images = model.modality_submodules[VISION_MODALITY_NAME]
        self.assertIs(images.pg_collection, vision_pg)

    def test_provider_rejects_ambiguous_infra_and_direct_args(self):
        infra = SimpleNamespace(
            module_to_grid_map={},
            module_to_pg_collection={},
        )
        with self.assertRaises(ValueError):
            self.build_provider_model(mimo_infra=infra, module_to_grid_map={})
        with self.assertRaises(ValueError):
            self.build_provider_model(mimo_infra=infra, pg_collection=SimpleNamespace())
        with self.assertRaises(ValueError):
            self.build_provider_model(mimo_infra=infra, images_pg_collection=SimpleNamespace())

    def test_get_rope_index_delegates_to_rope_helper(self):
        model = self.build_provider_model()
        rope_mock = mock.MagicMock(return_value=("position_ids", "ids"))
        with mock.patch.object(grid_mimo, "get_rope_index", rope_mock):
            result = model.get_rope_index(
                input_ids="INPUT", image_grid_thw="IMG", video_grid_thw="VID", attention_mask="MASK"
            )
        self.assertEqual(result, ("position_ids", "ids"))
        rope_mock.assert_called_once_with(
            spatial_merge_size=self.language_config.spatial_merge_size,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            input_ids="INPUT",
            image_grid_thw="IMG",
            video_grid_thw="VID",
            attention_mask="MASK",
        )


class TestQwen35GridMimoFreeze(Qwen35GridMimoModelTestBase):
    """freeze(): Qwen35Model semantics for the locally present modules."""

    def test_freeze_language_only(self):
        model = self.build_provider_model()
        model.freeze(
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=False,
        )
        self.assertTrue(all(not p.requires_grad for p in model.language_model.parameters()))
        images = model.modality_submodules[VISION_MODALITY_NAME]
        self.assertTrue(all(p.requires_grad for p in images.parameters()))

    def test_freeze_vision_encoder(self):
        model = self.build_provider_model()
        model.freeze(
            freeze_language_model=False,
            freeze_vision_model=True,
            freeze_vision_projection=False,
        )
        self.assertTrue(all(p.requires_grad for p in model.language_model.parameters()))
        images = model.modality_submodules[VISION_MODALITY_NAME]
        self.assertTrue(all(not p.requires_grad for p in images.parameters()))
        # The projection lives inside the encoder; freezing the encoder covers it.
        encoder = images.encoders[VISION_ENCODER_NAME]
        self.assertTrue(all(not p.requires_grad for p in encoder.projection.parameters()))

    def test_freeze_vision_projection_only(self):
        model = self.build_provider_model()
        model.freeze(
            freeze_language_model=False,
            freeze_vision_model=False,
            freeze_vision_projection=True,
        )
        images = model.modality_submodules[VISION_MODALITY_NAME]
        encoder = images.encoders[VISION_ENCODER_NAME]
        self.assertTrue(all(not p.requires_grad for p in encoder.projection.parameters()))
        # Encoder backbone and language module stay trainable (the projection
        # is a submodule of the encoder, so use the encoder's own weight).
        self.assertTrue(encoder.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.language_model.parameters()))

    def test_freeze_all_colocated(self):
        model = self.build_provider_model()
        model.freeze(True, True, True)
        for module in (model.language_model, model.modality_submodules[VISION_MODALITY_NAME]):
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))

    def test_freeze_nothing_changes_nothing(self):
        model = self.build_provider_model()
        model.freeze(False, False, False)
        for module in (model.language_model, model.modality_submodules[VISION_MODALITY_NAME]):
            self.assertTrue(all(p.requires_grad for p in module.parameters()))

    def test_freeze_language_absent_on_encoder_rank(self):
        grids = {
            VISION_MODALITY_NAME: FakeGrid(0, WORLD_SIZE),
            LANGUAGE_MODULE_NAME: FakeGrid(WORLD_SIZE, WORLD_SIZE),
        }
        model = self.build_provider_model(module_to_grid_map=grids)
        self.assertIsNone(model.language_model)
        model.freeze(True, True, True)
        images = model.modality_submodules[VISION_MODALITY_NAME]
        self.assertTrue(all(not p.requires_grad for p in images.parameters()))

    def test_freeze_vision_absent_on_language_rank(self):
        grids = {
            VISION_MODALITY_NAME: FakeGrid(WORLD_SIZE, WORLD_SIZE),
            LANGUAGE_MODULE_NAME: FakeGrid(0, WORLD_SIZE),
        }
        model = self.build_provider_model(module_to_grid_map=grids)
        self.assertNotIn(VISION_MODALITY_NAME, model.modality_submodules)
        model.freeze(True, True, True)
        self.assertTrue(all(not p.requires_grad for p in model.language_model.parameters()))


class TestQwen35GridVisualSplitSizes(unittest.TestCase):
    """compute_grid_visual_split_sizes: per-sample patch counts from grid_thw."""

    def _grid_thw(self, rows):
        return torch.tensor(rows, dtype=torch.long)

    def test_nonuniform_sizes(self):
        sizes = grid_mimo.compute_grid_visual_split_sizes(
            self._grid_thw([[1, 48, 48], [1, 24, 24], [1, 32, 16]]), 848, 2
        )
        self.assertEqual(sizes, [576, 144, 128])

    def test_uniform_sizes_return_none(self):
        # Uniform counts: the bridge's uniform tensor_split fallback is exact.
        sizes = grid_mimo.compute_grid_visual_split_sizes(
            self._grid_thw([[1, 24, 24], [1, 24, 24]]), 288, 2
        )
        self.assertIsNone(sizes)

    def test_no_visual_data_returns_none(self):
        self.assertIsNone(grid_mimo.compute_grid_visual_split_sizes(None, 0, 2))
        self.assertIsNone(
            grid_mimo.compute_grid_visual_split_sizes(torch.zeros(0, 3, dtype=torch.long), 0, 2)
        )

    def test_merge_size_one(self):
        sizes = grid_mimo.compute_grid_visual_split_sizes(
            self._grid_thw([[1, 4, 4], [1, 2, 2]]), 20, 1
        )
        self.assertEqual(sizes, [16, 4])

    def test_video_frames_count_temporal_dim(self):
        # A 2-frame video contributes t*h*w / merge_unit tokens per image (the
        # grid path rejects videos, but the math must stay consistent).
        sizes = grid_mimo.compute_grid_visual_split_sizes(
            self._grid_thw([[2, 24, 24], [1, 24, 24]]), 432, 2
        )
        self.assertEqual(sizes, [288, 144])

    def test_sum_mismatch_fails_fast(self):
        with self.assertRaises(ValueError):
            grid_mimo.compute_grid_visual_split_sizes(
                self._grid_thw([[1, 48, 48], [1, 24, 24]]), 100, 2
            )

    def test_grid_not_divisible_by_merge_unit_fails_fast(self):
        with self.assertRaises(ValueError):
            grid_mimo.compute_grid_visual_split_sizes(self._grid_thw([[1, 49, 48]]), 1176, 2)

    def test_invalid_merge_size_fails_fast(self):
        with self.assertRaises(ValueError):
            grid_mimo.compute_grid_visual_split_sizes(self._grid_thw([[1, 4, 4]]), 16, 0)


class TestQwen35GridMimoSplitSizesAttach(Qwen35GridMimoModelTestBase):
    """_attach_modality_split_sizes override: grid-derived metadata, fail fast."""

    def test_attach_nonuniform_sizes_from_grid_thw(self):
        model = self.build_provider_model()
        model._current_grid_thw = torch.tensor([[1, 48, 48], [1, 24, 24]], dtype=torch.long)
        output = torch.zeros(720, 64)
        model._attach_modality_split_sizes(
            output, torch.zeros(2, 100, dtype=torch.long), VISION_MODALITY_NAME
        )
        self.assertEqual(output._mimo_bridge_split_sizes, [576, 144])

    def test_uniform_sizes_leave_no_metadata(self):
        model = self.build_provider_model()
        model._current_grid_thw = torch.tensor([[1, 24, 24], [1, 24, 24]], dtype=torch.long)
        output = torch.zeros(288, 64)
        model._attach_modality_split_sizes(
            output, torch.zeros(2, 50, dtype=torch.long), VISION_MODALITY_NAME
        )
        self.assertFalse(hasattr(output, "_mimo_bridge_split_sizes"))

    def test_sum_mismatch_raises_instead_of_silent_uniform_split(self):
        model = self.build_provider_model()
        model._current_grid_thw = torch.tensor([[1, 48, 48], [1, 24, 24]], dtype=torch.long)
        output = torch.zeros(100, 64)  # wrong size: silent drop would corrupt fan-out
        with self.assertRaises(ValueError):
            model._attach_modality_split_sizes(
                output, torch.zeros(2, 100, dtype=torch.long), VISION_MODALITY_NAME
            )

    def test_no_grid_thw_falls_back_to_super_token_counts(self):
        model = self.build_provider_model()
        model._current_grid_thw = None
        output = torch.zeros(1, 64)
        input_ids = torch.tensor([[IMAGE_TOKEN_ID, 1, 2], [3, 4, 5]], dtype=torch.long)
        model._attach_modality_split_sizes(output, input_ids, VISION_MODALITY_NAME)
        self.assertEqual(output._mimo_bridge_split_sizes, [1, 0])

    def test_encoder_rank_attach_with_fanout_guard_ok(self):
        grids = {
            VISION_MODALITY_NAME: FakeGrid(0, WORLD_SIZE),
            LANGUAGE_MODULE_NAME: FakeGrid(WORLD_SIZE, WORLD_SIZE),
        }
        model = self.build_provider_model(module_to_grid_map=grids)
        self.assertEqual(model.role.mode, ModuleLayout.NON_COLOCATED)
        model._current_grid_thw = torch.tensor(
            [[1, 48, 48], [1, 24, 24], [1, 32, 16]], dtype=torch.long
        )
        output = torch.zeros(848, 64)
        model._attach_modality_split_sizes(
            output, torch.zeros(3, 100, dtype=torch.long), VISION_MODALITY_NAME
        )
        self.assertEqual(output._mimo_bridge_split_sizes, [576, 144, 128])

    def test_fan_in_guard_rejects_encoder_dp_gt_language_dp(self):
        # NOTE: no "pp" dim on the stub grids - RankRole calls grid.get_pg
        # for "pp" during role derivation; the DP guard only needs dim_names.
        dims = ["tp", "dp"]
        grids = {
            VISION_MODALITY_NAME: FakeGrid(0, WORLD_SIZE, dims, [2, 2]),
            LANGUAGE_MODULE_NAME: FakeGrid(WORLD_SIZE, WORLD_SIZE, dims, [2, 1]),
        }
        model = self.build_provider_model(module_to_grid_map=grids)
        self.assertEqual(model.role.mode, ModuleLayout.NON_COLOCATED)
        model._current_grid_thw = torch.tensor([[1, 48, 48], [1, 24, 24]], dtype=torch.long)
        with self.assertRaises(ValueError):
            model._attach_modality_split_sizes(
                torch.zeros(720, 64), torch.zeros(2, 100, dtype=torch.long), VISION_MODALITY_NAME
            )


class TestQwen35GridBridgeSplitContract(unittest.TestCase):
    """Attached metadata must satisfy the real BridgeCommunicator fan-out contract."""

    @staticmethod
    def _split(sizes, num_splits):
        from megatron.core.pipeline_parallel.bridge_communicator import (
            BridgeCommunicator,
        )

        comm = object.__new__(BridgeCommunicator)
        comm.tensor_ndim = 2  # 2D encoder outputs fan in/out on dim 0
        tensor = torch.zeros(sum(sizes), 4)
        tensor._mimo_bridge_split_sizes = sizes
        splits = comm._split_tensor_at_batch_dim(tensor, num_splits)
        return [int(split.size(0)) for split in splits]

    def test_one_send_per_sample_sizes(self):
        self.assertEqual(self._split([576, 144, 128], 3), [576, 144, 128])

    def test_grouping_merges_samples_per_peer(self):
        self.assertEqual(self._split([576, 144, 128, 96], 2), [720, 224])

    def test_fanout_family7_dp2_to_dp6(self):
        # Family 7 (V TP1/DP2 -> L TP1/DP6): num_sends = 3 per encoder shard;
        # each shard holds 3 samples of a 6-sample micro-batch.  The metadata
        # length equals num_sends, so each peer receives exactly one sample's
        # patches (no uniform splitting of variable-resolution patches).
        sizes = [576, 144, 128]
        self.assertEqual(self._split(sizes, 3), sizes)

    def test_fanout_family1_dp1_to_dp6(self):
        # Family 1 (V TP2/DP1 -> L TP1/DP6): num_sends = 6; the 6-sample
        # metadata is grouped per peer (one sample each).
        sizes = [576, 144, 128, 96, 400, 64]
        self.assertEqual(self._split(sizes, 6), sizes)

    def test_misaligned_metadata_fails_fast_not_silent(self):
        # 3 per-sample sizes cannot serve 2 fan-out peers: the bridge raises
        # instead of silently falling back to a uniform split.
        with self.assertRaises(ValueError):
            self._split([576, 144, 128], 2)


class TestQwen35VisionRotaryShapeContract(unittest.TestCase):
    """Vision packed-seq rotary contract: query dim0 == sum(cu_seqlens).

    Reproduces the reported family-1 VTP2 failure signature: with sequence
    parallelism enabled on the TP2 vision encoder, the first column-parallel
    qkv layer all-gathers the full sequence along dim 0 (2x tokens), while the
    packed-seq params built from ``grid_thw`` still describe the full token
    dimension - ``torch.split(query, seqlens)`` then fails (query 6720 vs
    cu_seqlens sum 3360).  The grid SP policy
    (``compute_qwen35_grid_sequence_parallel``) keeps vision SP off, so the
    query dim matches the cu_seqlens sum and the real THD rotary path
    succeeds.
    """

    class _Cp1:
        """CP=1 stand-in for the vision module's context-parallel group."""

        @staticmethod
        def size():
            return 1

        @staticmethod
        def rank():
            return 0

    @staticmethod
    def _cu_seqlens(seqlens):
        cu = [0]
        for length in seqlens:
            cu.append(cu[-1] + length)
        return torch.tensor(cu, dtype=torch.int32)

    def test_thd_rotary_succeeds_with_full_dim_query(self):
        from flagscale.models.megatron.qwen3_vl.vision_rope_utils import (
            _apply_rotary_pos_emb_thd,
        )

        # The reported per-frame seqlens of a 6-image variable-resolution
        # micro-batch; the packed query must have exactly this many tokens.
        seqlens = [400, 400, 640, 640, 640, 640]
        cu_seqlens = self._cu_seqlens(seqlens)
        self.assertEqual(int(cu_seqlens[-1]), 3360)

        freqs = torch.randn(700, 1, 1, 8)
        query = torch.randn(3360, 2, 8)
        out = _apply_rotary_pos_emb_thd(query, cu_seqlens, freqs, cp_group=self._Cp1())
        self.assertEqual(tuple(out.shape), (3360, 2, 8))

    def test_sp_doubled_query_dim_fails_the_same_split(self):
        from flagscale.models.megatron.qwen3_vl.vision_rope_utils import (
            _apply_rotary_pos_emb_thd,
        )

        # SP-on-TP2 shape semantics: the qkv column-parallel all-gather doubles
        # dim 0 (6720 = 2 * 3360) while cu_seqlens still describe the full
        # token dimension - the exact failure mode the vision-SP disable
        # (compute_qwen35_grid_sequence_parallel) prevents.
        seqlens = [400, 400, 640, 640, 640, 640]
        cu_seqlens = self._cu_seqlens(seqlens)
        freqs = torch.randn(700, 1, 1, 8)
        sp_query = torch.randn(2 * int(cu_seqlens[-1]), 2, 8)
        with self.assertRaises(RuntimeError):
            _apply_rotary_pos_emb_thd(sp_query, cu_seqlens, freqs, cp_group=self._Cp1())


class TestQwen35LanguageRotaryShapeContract(unittest.TestCase):
    """Language absolute (mRoPE) rotary contract: query dim0 == freqs dim0.

    Reproduces the family-5/6 language-TP2 failure signature: with sequence
    parallelism enabled on the TP2 language module, the grid forward feeds
    the FULL sequence into the transformer stack (the Qwen3.5 grid embedding
    ``QwenVLLanguageModelEmbedding`` does not scatter to SP ranks), so the
    first column-parallel qkv all-gather doubles dim 0 (4096 = 2 * 2048)
    while the mRoPE freqs built from the full-length ``position_ids`` still
    describe S positions (2048) - ``apply_rotary_pos_emb_absolute`` then
    multiplies a 2S query against S freqs.  The grid SP policy
    (``compute_qwen35_grid_sequence_parallel``) keeps language SP off, so the
    real rotary path always sees matching dims.  No collectives are needed:
    the shape mismatch is visible in a single-rank rotary apply.
    """

    # S = 2048 tokens; query head dim 128 with a 64-wide rotary slice
    # (2 * kv_channels * rotary_percent), matching the Qwen3.5 4B mRoPE
    # setup.
    SEQLEN = 2048
    HEAD_DIM = 128
    ROTARY_DIM = 64

    @staticmethod
    def _config():
        return SimpleNamespace(
            apply_rope_fusion=False,
            apply_rotary_pos_emb_in_fp32=False,
            rotary_interleaved=False,
        )

    def test_absolute_rotary_succeeds_with_matching_dims(self):
        from flagscale.models.megatron.qwen35.rope import apply_rotary_pos_emb_absolute

        query = torch.randn(self.SEQLEN, 1, 1, self.HEAD_DIM)
        freqs = torch.randn(self.SEQLEN, 1, 1, self.ROTARY_DIM)
        out = apply_rotary_pos_emb_absolute(query, freqs, self._config())
        self.assertEqual(tuple(out.shape), (self.SEQLEN, 1, 1, self.HEAD_DIM))

    def test_sp_doubled_query_dim_fails_against_full_length_freqs(self):
        from flagscale.models.megatron.qwen35.rope import apply_rotary_pos_emb_absolute

        # SP-on-TP2 shape semantics for the grid language module: the qkv
        # column-parallel all-gather doubles dim 0 (2S) while the full-length
        # mRoPE freqs still describe S positions - the exact failure mode the
        # language-SP disable (compute_qwen35_grid_sequence_parallel)
        # prevents.
        sp_query = torch.randn(2 * self.SEQLEN, 1, 1, self.HEAD_DIM)
        freqs = torch.randn(self.SEQLEN, 1, 1, self.ROTARY_DIM)
        with self.assertRaises(RuntimeError):
            apply_rotary_pos_emb_absolute(sp_query, freqs, self._config())

    def test_sp_doubled_query_dim_fails_thd_path_too(self):
        from flagscale.models.megatron.qwen35.rope import apply_rotary_pos_emb_thd_absolute

        # The packed-seq (THD) variant takes cu_seqlens but still applies the
        # full-length freqs elementwise along dim 0: the 2S-vs-S mismatch
        # fails the same way (cu_seqlens describe the full sequence, not the
        # gathered 2x one).
        cu_seqlens = torch.tensor([0, self.SEQLEN // 2, self.SEQLEN], dtype=torch.int32)
        sp_query = torch.randn(2 * self.SEQLEN, 1, 1, self.HEAD_DIM)
        freqs = torch.randn(self.SEQLEN, 1, 1, self.ROTARY_DIM)
        with self.assertRaises(RuntimeError):
            apply_rotary_pos_emb_thd_absolute(sp_query, cu_seqlens, freqs, rotary_interleaved=False)


if __name__ == "__main__":
    unittest.main()

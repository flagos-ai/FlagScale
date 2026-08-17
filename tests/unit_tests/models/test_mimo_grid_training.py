# Copyright (c) 2026, BAAI. All rights reserved.

"""CPU unit tests for flagscale.models.mimo.bridge.training.

Covers the grid training-lifecycle helpers that are CPU-testable:

- ``setup_grid_mimo_ddp``: per-module DDP config resolution.  The images
  submodule (``Qwen35VisionSubmodules``) is a ``ModalitySubmodules`` and has
  no ``config`` attribute of its own; the vision transformer config threaded
  by ``build_qwen35_images_submodule_spec`` must be used (with a robust
  fallback to the model config).
- ``build_language_forward_kwargs``: the language-rank batch dict must contain
  exactly the keyword arguments accepted by ``Qwen35GridMIMOModel.forward`` -
  nulled leftover batch keys (``imgs`` / ``videos`` / ``image_thw_grids`` /
  ``video_thw_grids``) would make ``model(**data_batch)`` raise TypeError.
- ``reconfigure_grid_num_microbatches_calculator``: the parse-time
  num-microbatches calculator (YAML global DP) is updated to the grid-forced
  DP=1 before the batch contract / schedule consume it (48/6: 2 -> 8
  microbatches); the colocated path leaves the calculator untouched.

The real ``megatron.core.distributed.DistributedDataParallel`` constructor is
mocked out; only the config-resolution surface of the setup path is exercised.

Run (inside the container, from the FlagScale repo root):

    source /root/miniconda3/bin/activate flagscale
    PYTHONPATH=/workspace/multimodal/Megatron-LM-FL python -m pytest \
        tests/unit_tests/models/test_mimo_grid_training.py
"""

import inspect
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

_MEGATRON_REPO = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, "Megatron-LM-FL"))
if os.path.isdir(_MEGATRON_REPO) and _MEGATRON_REPO not in sys.path:
    sys.path.insert(0, _MEGATRON_REPO)

import torch
import torch.nn as nn

import flagscale.models.mimo.bridge.training as training
from flagscale.models.mimo.bridge.data import slice_batch_for_module_dp
from flagscale.models.mimo.bridge.parallelism import (
    LANGUAGE_MODULE_NAME,
)
from flagscale.models.mimo.bridge.providers.qwen35 import (
    Qwen35GridMIMOModel,
)
from flagscale.models.mimo.bridge.recipe.qwen35 import (
    build_qwen35_grid_config_from_args,
    qwen35_grid_data_contract,
)
from flagscale.models.mimo.bridge.training import (
    GRID_LANGUAGE_FORWARD_KEYS,
    build_language_forward_kwargs,
    build_vision_forward_kwargs,
    destroy_grid_training_states,
    get_logical_iteration_samples,
    reconfigure_grid_num_microbatches_calculator,
    setup_grid_mimo_ddp,
    validate_qwen35_grid_runtime_contract,
)


class FakeParamModule(nn.Module):
    """Stand-in for a wrapped grid submodule carrying a ``config``."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.randn(2, 2))

    def set_input_tensor(self, tensor):
        pass


class FakeDDP:
    """Mock for megatron.core.distributed.DistributedDataParallel."""

    instances = []

    def __init__(self, config, ddp_config, module, pg_collection):
        self.config = config
        self.ddp_config = ddp_config
        self.module = module
        self.pg_collection = pg_collection
        FakeDDP.instances.append(self)

    def set_input_tensor(self, tensor):
        pass


class FakeGrid:
    """Minimal grid stub (rank-range membership only)."""

    def __init__(self, size=8):
        self.size = size


def make_pg():
    return SimpleNamespace(dp=SimpleNamespace(size=lambda: 1))


def make_grid_state(pg):
    infra = SimpleNamespace(
        module_to_pg_collection={
            "images": pg,
            LANGUAGE_MODULE_NAME: pg,
        },
        module_to_grid_map={
            "images": FakeGrid(),
            LANGUAGE_MODULE_NAME: FakeGrid(),
        },
    )
    return SimpleNamespace(infra=infra)


class TestSetupGridMimoDdpConfig(unittest.TestCase):
    """setup_grid_mimo_ddp: per-module DDP must get the module's own config.

    Regression: the images submodule (``Qwen35VisionSubmodules``) has no
    ``config`` attribute of its own; before the fix ``module.config`` raised
    AttributeError on every encoder rank during DDP wrapping.
    """

    def setUp(self):
        FakeDDP.instances = []
        self.language_config = SimpleNamespace(name="language-config")
        self.vision_config = SimpleNamespace(name="vision-config")
        self.args = SimpleNamespace(mimo_layout="grid")

        self.language_module = FakeParamModule(config=self.language_config)
        self.images_submodule = FakeParamModule(config=self.vision_config)
        self.mimo_model = SimpleNamespace(
            language_model=self.language_module,
            modality_submodules={"images": self.images_submodule},
            config=self.language_config,
        )
        self.pg = make_pg()
        self.grid_state = make_grid_state(self.pg)
        self.mimo_model.mimo_grid_state = self.grid_state

        self.ddp_patcher = mock.patch("flagscale.models.mimo.bridge.training.DDP", FakeDDP)
        self.ddp_patcher.start()
        self.addCleanup(self.ddp_patcher.stop)
        self.ddp_config_patcher = mock.patch(
            "flagscale.models.mimo.bridge.training.build_mimo_ddp_config",
            return_value=SimpleNamespace(mock=True),
        )
        self.ddp_config_patcher.start()
        self.addCleanup(self.ddp_config_patcher.stop)

    def _run(self):
        return setup_grid_mimo_ddp([self.mimo_model], self.args, wrap_with_ddp=True)

    def test_images_submodule_gets_vision_config_language_gets_language_config(self):
        is_grid, grid_state = self._run()
        self.assertTrue(is_grid)
        self.assertIsNotNone(grid_state)
        by_module = {id(ddp.module): ddp for ddp in FakeDDP.instances}
        language_ddp = by_module[id(self.language_module)]
        images_ddp = by_module[id(self.images_submodule)]
        self.assertIs(language_ddp.config, self.language_config)
        self.assertIs(images_ddp.config, self.vision_config)

    def test_module_without_config_falls_back_to_model_config(self):
        # Robustness: a submodule type that does not carry a threaded config
        # must fall back to the model config (the language config) instead of
        # raising AttributeError.
        self.images_submodule.config = None
        is_grid, _ = self._run()
        self.assertTrue(is_grid)
        by_module = {id(ddp.module): ddp for ddp in FakeDDP.instances}
        self.assertIs(by_module[id(self.images_submodule)].config, self.language_config)

    def test_language_and_images_submodules_replaced_in_place(self):
        self._run()
        self.assertIsInstance(self.mimo_model.language_model, FakeDDP)
        self.assertIsInstance(self.mimo_model.modality_submodules["images"], FakeDDP)
        # Colocated-compatible aliases kept out of named_children().
        self.assertIs(self.mimo_model.language_ddp, self.mimo_model.language_model)
        self.assertIs(self.mimo_model.vision_ddp, self.mimo_model.modality_submodules["images"])
        self.assertEqual(list(self.mimo_model.module_to_ddp), ["images", LANGUAGE_MODULE_NAME])

    def test_non_grid_model_returns_false(self):
        self.args.mimo_layout = "colocated"
        is_grid, grid_state = self._run()
        self.assertFalse(is_grid)
        self.assertIsNone(grid_state)
        self.assertEqual(FakeDDP.instances, [])


class TestBuildLanguageForwardKwargs(unittest.TestCase):
    """build_language_forward_kwargs: exactly the forward-accepted keys."""

    def _batch(self, batch_size=4):
        return {
            "tokens": torch.arange(batch_size * 6, dtype=torch.long).reshape(batch_size, 6),
            "labels": torch.arange(batch_size * 6, dtype=torch.long).reshape(batch_size, 6),
            "loss_mask": torch.ones(batch_size, 6),
            "position_ids": torch.zeros(3, batch_size, 6, dtype=torch.long),
            "attention_mask": None,
            "imgs": torch.zeros(batch_size, 3, 16, 16),
            "videos": torch.zeros(batch_size, 3, 16, 16),
            "image_thw_grids": torch.ones(batch_size, 3, dtype=torch.long),
            "video_thw_grids": torch.ones(batch_size, 3, dtype=torch.long),
            "image_input_mask": torch.zeros(batch_size, 6, dtype=torch.bool),
            "video_input_mask": torch.zeros(batch_size, 6, dtype=torch.bool),
        }

    def _accepted_forward_keys(self):
        params = inspect.signature(Qwen35GridMIMOModel.forward).parameters
        return set(params) - {"self"}

    def test_keys_are_exactly_the_accepted_forward_kwargs(self):
        kwargs = build_language_forward_kwargs(
            self._batch(), dp_rank=0, dp_size=1, pp_rank=0, pp_size=1
        )
        self.assertEqual(set(kwargs), set(GRID_LANGUAGE_FORWARD_KEYS))
        self.assertLessEqual(set(kwargs), self._accepted_forward_keys())
        # The model's required args are present.
        for required in ("input_ids",):
            self.assertIn(required, kwargs)

    def test_no_leftover_nulled_vision_keys(self):
        kwargs = build_language_forward_kwargs(
            self._batch(), dp_rank=0, dp_size=1, pp_rank=0, pp_size=1
        )
        for leftover in ("imgs", "videos", "image_thw_grids", "video_thw_grids", "tokens"):
            self.assertNotIn(leftover, kwargs)

    def test_dp_slice_applied_to_tokens(self):
        batch = self._batch(batch_size=4)
        kwargs = build_language_forward_kwargs(batch, dp_rank=1, dp_size=2, pp_rank=0, pp_size=1)
        self.assertEqual(kwargs["input_ids"].size(0), 2)
        self.assertTrue(torch.equal(kwargs["input_ids"], batch["tokens"][2:4]))
        self.assertEqual(kwargs["image_input_mask"].size(0), 2)
        self.assertEqual(kwargs["video_input_mask"].size(0), 2)
        self.assertIsNone(kwargs["modality_inputs"])
        self.assertIsNone(kwargs["packing_kwargs"])

    def test_non_first_stage_nulls_input_ids_keeps_last_stage_fields(self):
        kwargs = build_language_forward_kwargs(
            self._batch(), dp_rank=0, dp_size=1, pp_rank=1, pp_size=2
        )
        self.assertIsNone(kwargs["input_ids"])
        self.assertIsNotNone(kwargs["labels"])
        self.assertIsNotNone(kwargs["loss_mask"])

    def test_non_last_stage_nulls_labels_and_loss_mask(self):
        kwargs = build_language_forward_kwargs(
            self._batch(), dp_rank=0, dp_size=1, pp_rank=0, pp_size=2
        )
        self.assertIsNotNone(kwargs["input_ids"])
        self.assertIsNone(kwargs["labels"])
        self.assertIsNone(kwargs["loss_mask"])

    def test_video_start_index_from_image_input_mask(self):
        batch = self._batch(batch_size=2)
        batch["image_input_mask"] = torch.zeros(2, 6, dtype=torch.bool)
        batch["image_input_mask"][0, 0] = True
        batch["image_input_mask"][1, 2] = True
        kwargs = build_language_forward_kwargs(batch, dp_rank=0, dp_size=1, pp_rank=0, pp_size=1)
        self.assertEqual(kwargs["video_start_index"], 2)
        self.assertEqual(kwargs["image_input_mask"].sum().item(), 2)

    def test_dp_sliced_modality_inputs_stay_none(self):
        kwargs = build_language_forward_kwargs(
            self._batch(batch_size=6), dp_rank=2, dp_size=3, pp_rank=0, pp_size=1
        )
        self.assertIsNone(kwargs["modality_inputs"])

    def test_language_kwargs_dp6_with_patch_packed_imgs(self):
        """Real-shape regression: mbs6 with patch-packed imgs (dim0 = total
        patches, 3080, not divisible by language DP 6) must succeed.

        The raw modality tensors are dropped BEFORE the sample-DP slice - the
        generic slicer must never see them (3080 % 6 != 0); only the
        sample-aligned keys are sliced (masks/tokens split to batch 1).
        """
        grid = torch.tensor(
            [[1, 32, 16]] * 5 + [[1, 26, 20]], dtype=torch.long
        )  # 6 images, products sum to 5*512 + 520 = 3080
        self.assertEqual(int(grid.prod(dim=-1).sum()), 3080)
        batch = {
            "tokens": torch.arange(6 * 8, dtype=torch.long).reshape(6, 8),
            "labels": torch.arange(6 * 8, dtype=torch.long).reshape(6, 8),
            "loss_mask": torch.ones(6, 8),
            "position_ids": torch.zeros(3, 6, 8, dtype=torch.long),
            "attention_mask": None,
            "imgs": torch.zeros(3080, 16),
            "videos": None,
            "image_thw_grids": grid,
            "video_thw_grids": None,
            "image_input_mask": torch.zeros(6, 8, dtype=torch.bool),
            "video_input_mask": torch.zeros(6, 8, dtype=torch.bool),
        }
        # Pre-fix failure mode: the generic sample-DP slicer on the raw batch
        # treats imgs dim0 (patches) as the sample batch and asserts % 6.
        with self.assertRaises(AssertionError):
            slice_batch_for_module_dp(batch, dp_rank=1, dp_size=6)

        kwargs = build_language_forward_kwargs(batch, dp_rank=1, dp_size=6, pp_rank=0, pp_size=1)
        # No raw modality keys survive (they are dropped before slicing).
        for leftover in (
            "imgs",
            "videos",
            "image_thw_grids",
            "video_thw_grids",
            "tokens",
        ):
            self.assertNotIn(leftover, kwargs)
        # Sample-aligned keys split to batch 1 (6 // 6).
        self.assertEqual(kwargs["input_ids"].size(0), 1)
        self.assertEqual(kwargs["labels"].size(0), 1)
        self.assertEqual(kwargs["loss_mask"].size(0), 1)
        self.assertEqual(kwargs["position_ids"].size(1), 1)
        self.assertEqual(kwargs["image_input_mask"].size(0), 1)
        self.assertEqual(kwargs["video_input_mask"].size(0), 1)
        self.assertIsNone(kwargs["modality_inputs"])


class TestBuildVisionForwardKwargs(unittest.TestCase):
    """build_vision_forward_kwargs: patch-packed joint slicing by vision DP."""

    def _batch(self, seq=8):
        grid = torch.tensor(
            [[1, 32, 16]] * 5 + [[1, 26, 20]], dtype=torch.long
        )  # 6 images, products sum to 3080
        imgs = torch.arange(3080 * 4, dtype=torch.float32).reshape(3080, 4)
        return {
            "tokens": torch.arange(6 * seq, dtype=torch.long).reshape(6, seq),
            "labels": torch.arange(6 * seq, dtype=torch.long).reshape(6, seq),
            "loss_mask": torch.ones(6, seq),
            "position_ids": torch.zeros(3, 6, seq, dtype=torch.long),
            "attention_mask": None,
            "imgs": imgs,
            "videos": None,
            "image_thw_grids": grid,
            "video_thw_grids": None,
            "image_input_mask": torch.ones(6, seq, dtype=torch.bool),
            "video_input_mask": torch.zeros(6, seq, dtype=torch.bool),
        }

    @staticmethod
    def _vision_inputs(kwargs):
        return kwargs["modality_inputs"]["images"]["qwen3_vit"]

    def test_dp1_keeps_all_images_and_patches(self):
        kwargs = build_vision_forward_kwargs(self._batch(), dp_rank=0, dp_size=1)
        vision = self._vision_inputs(kwargs)
        self.assertEqual(vision["grid_thw"].size(0), 6)
        self.assertEqual(vision["vision_data"].size(0), 3080)
        self.assertEqual(kwargs["input_ids"].size(0), 6)

    def test_family7_dp2_joint_slice_by_image_boundaries(self):
        """Family 7 (vision DP 2): imgs/grid are sliced JOINTLY along per-image
        patch boundaries, not independently (no mid-image patch splits)."""
        batch = self._batch()
        # Shard 0: first 3 images -> patches [0, 3*512) = 1536.
        kwargs0 = build_vision_forward_kwargs(batch, dp_rank=0, dp_size=2)
        vision0 = self._vision_inputs(kwargs0)
        self.assertEqual(vision0["grid_thw"].size(0), 3)
        self.assertEqual(vision0["vision_data"].size(0), 1536)
        torch.testing.assert_close(vision0["vision_data"], batch["imgs"][:1536])
        # Shard 1: last 3 images -> patches [1536, 3080) = 1544 (512+512+520).
        kwargs1 = build_vision_forward_kwargs(batch, dp_rank=1, dp_size=2)
        vision1 = self._vision_inputs(kwargs1)
        self.assertEqual(vision1["grid_thw"].size(0), 3)
        self.assertEqual(vision1["vision_data"].size(0), 1544)
        torch.testing.assert_close(vision1["vision_data"], batch["imgs"][1536:])
        # The two shards tile the full batch.
        torch.testing.assert_close(
            torch.cat([vision0["vision_data"], vision1["vision_data"]], dim=0),
            batch["imgs"],
        )
        # Sample-aligned keys slice by vision DP as usual.
        self.assertEqual(kwargs0["input_ids"].size(0), 3)
        self.assertEqual(kwargs1["input_ids"].size(0), 3)

    def test_video_input_fails_fast(self):
        batch = self._batch()
        batch["video_input_mask"] = torch.zeros(6, 8, dtype=torch.bool)
        batch["video_input_mask"][2, 0] = True
        with self.assertRaises(NotImplementedError):
            build_vision_forward_kwargs(batch, dp_rank=0, dp_size=1)

    def test_text_only_batch_uses_empty_modality_inputs(self):
        batch = self._batch()
        batch["imgs"] = torch.empty(0, 4)
        batch["image_thw_grids"] = torch.empty(0, 3, dtype=torch.long)
        batch["image_input_mask"] = torch.zeros(6, 8, dtype=torch.bool)
        kwargs = build_vision_forward_kwargs(batch, dp_rank=0, dp_size=1)
        self.assertIsNone(kwargs["modality_inputs"])

    def test_multi_image_sample_fails_fast(self):
        batch = self._batch()
        batch["image_input_mask"] = torch.zeros(6, 8, dtype=torch.bool)
        batch["image_input_mask"][:2, :2] = True
        with self.assertRaises(ValueError):
            build_vision_forward_kwargs(batch, dp_rank=0, dp_size=1)


class TestGridNumMicrobatchesCalculator(unittest.TestCase):
    """Grid-mode num-microbatches calculator reconfigure (real calculator).

    Uses the REAL ``megatron.core.num_microbatches_calculator`` in constant
    mode (no process-group initialization needed) to prove:

    - the parse-time calculator (initialized by ``set_global_variables`` with
      the YAML's global DP, e.g. 4 for TP2 on 8 ranks) reports 48/6 -> 2
      microbatches, which breaks the grid batch contract (2 * 6 != 48);
    - ``reconfigure_grid_num_microbatches_calculator`` updates it to the
      grid-forced DP=1 before validation/schedule: 48/6 -> 8 microbatches
      (8 * 6 == 48) and the contract passes;
    - the colocated (non-grid) path never touches the calculator.
    """

    def setUp(self):
        from megatron.core.num_microbatches_calculator import (
            destroy_num_microbatches_calculator,
        )

        destroy_num_microbatches_calculator()
        self.addCleanup(destroy_num_microbatches_calculator)

    def _init_parse_time(self):
        from megatron.core.num_microbatches_calculator import (
            init_num_microbatches_calculator,
        )

        init_num_microbatches_calculator(
            rank=0, global_batch_size=48, micro_batch_size=6, data_parallel_size=4
        )

    def _grid_args(self):
        return SimpleNamespace(
            rank=0,
            global_batch_size=48,
            micro_batch_size=6,
            data_parallel_size=1,  # forced by the grid entry point
            decrease_batch_size_if_needed=False,
            step_batch_size_schedule=None,
            seq_length=2048,
        )

    def test_colocated_parse_time_calculator_unchanged(self):
        # Colocated path: no reconfigure happens; the parse-time DP=4 calculator
        # keeps 48 / (6*4) = 2 microbatches and preserves gbs/mbs.
        self._init_parse_time()
        from megatron.core.num_microbatches_calculator import (
            get_current_global_batch_size,
            get_micro_batch_size,
            get_num_microbatches,
        )

        self.assertEqual(get_num_microbatches(), 2)
        self.assertEqual(get_micro_batch_size(), 6)
        self.assertEqual(get_current_global_batch_size(), 48)

    def test_grid_reconfigure_dp1_gives_8_microbatches(self):
        # Grid path: the parse-time DP=4 calculator is reconfigured to the
        # forced DP=1; 48/6 -> 8 microbatches, gbs/mbs preserved.
        self._init_parse_time()
        reconfigure_grid_num_microbatches_calculator(self._grid_args())
        from megatron.core.num_microbatches_calculator import (
            get_current_global_batch_size,
            get_micro_batch_size,
            get_num_microbatches,
        )

        self.assertEqual(get_num_microbatches(), 8)
        self.assertEqual(get_micro_batch_size(), 6)
        self.assertEqual(get_current_global_batch_size(), 48)
        self.assertEqual(get_num_microbatches() * get_micro_batch_size(), 48)

    def test_grid_contract_fails_at_parse_time_then_passes_after_reconfigure(self):
        # Reproduces the reported GPU failure: the grid batch contract sees
        # the parse-time microbatches (2) and rejects 2*6 != 48; after the
        # DP=1 reconfigure it sees 8 and passes (2+6 layout: language DP 6).
        self._init_parse_time()
        from megatron.core.num_microbatches_calculator import get_num_microbatches

        config = build_qwen35_grid_config_from_args(
            "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
            8,
            num_layers=32,
        )
        with self.assertRaises(ValueError):
            qwen35_grid_data_contract(
                config,
                micro_batch_size=6,
                global_batch_size=48,
                num_microbatches=get_num_microbatches(),
            )
        reconfigure_grid_num_microbatches_calculator(self._grid_args())
        self.assertEqual(get_num_microbatches(), 8)
        per_module_dp = qwen35_grid_data_contract(
            config,
            micro_batch_size=6,
            global_batch_size=48,
            num_microbatches=get_num_microbatches(),
        )
        self.assertEqual(per_module_dp, {"images": 1, "language": 6})

    def test_colocated_leaves_calculator_untouched(self):
        # The helper is only invoked by the grid entry point; without it the
        # calculator state never changes (colocated behavior preserved).
        self._init_parse_time()
        from megatron.core.num_microbatches_calculator import get_num_microbatches

        self.assertEqual(get_num_microbatches(), 2)
        self.assertEqual(get_num_microbatches(), 2)


class TestGridRuntimeContracts(unittest.TestCase):
    def _args(self, **overrides):
        values = dict(
            use_mimo=True,
            mimo_layout="grid",
            micro_batch_size=6,
            eval_micro_batch_size=6,
            eval_global_batch_size=48,
            eval_iters=0,
            cuda_graph_impl="none",
            context_parallel_size=1,
            expert_model_parallel_size=1,
            num_experts=None,
            virtual_pipeline_model_parallel_size=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_grid_iteration_samples_ignore_global_dp(self):
        self.assertEqual(get_logical_iteration_samples(self._args(), 8), 48)

    def test_colocated_uses_standard_sample_accounting(self):
        self.assertIsNone(get_logical_iteration_samples(self._args(mimo_layout="colocated"), 8))

    def test_full_iteration_cuda_graph_rejected(self):
        with self.assertRaises(ValueError):
            validate_qwen35_grid_runtime_contract(self._args(cuda_graph_impl="full_iteration"))

    def test_eval_microbatch_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            validate_qwen35_grid_runtime_contract(self._args(eval_iters=1, eval_micro_batch_size=4))

    def test_moe_and_vpp_rejected(self):
        for args in (
            self._args(num_experts=8),
            self._args(virtual_pipeline_model_parallel_size=2),
        ):
            with self.assertRaises(ValueError):
                validate_qwen35_grid_runtime_contract(args)


class TestGridStateLifecycle(unittest.TestCase):
    def test_destroy_all_is_idempotent(self):
        infra = mock.Mock()
        state = training.GridTrainingState(infra=infra)
        training._GRID_TRAINING_STATES.append(state)
        with mock.patch.object(
            training.BridgeCommunicator, "destroy_broadcast_pgs"
        ) as destroy_broadcast:
            destroy_grid_training_states()
            destroy_grid_training_states()
        infra.destroy.assert_called_once_with()
        self.assertEqual(destroy_broadcast.call_count, 2)


if __name__ == "__main__":
    unittest.main()

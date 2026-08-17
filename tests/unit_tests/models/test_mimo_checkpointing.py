# Copyright (c) 2026, BAAI. All rights reserved.

"""CPU unit tests for the MIMO module-namespaced sharded optimizer state.

Covers the checkpointing fix for the 8-GPU torch_dist exit-save failure:

- ``GridMimoOptimizer`` (``flagscale.models.mimo.bridge.training``): the
  per-module inner optimizers emit identical ``optimizer.distributed.dp_group_idx_*``
  shard keys, so the sharded state of the images and language modules collides
  in the global torch_dist checkpoint (duplicate ShardedObject keys and
  ShardedTensor global-shape mismatches at save-time validation).  The
  wrapper must namespace those keys per module on save and restore them
  before delegating to the inner optimizers on load.
- colocated ``ChainedOptimizer`` (``flagscale.models.mimo.colocated.mimo_optimizer``):
  the same collision exists for a chain of per-module optimizers; sharded keys
  get the MCore-style ``chained_<idx>.`` prefix on save, stripped on load.
- ``sync_grid_optimizer_param_group_lr``
  (``flagscale.models.mimo.bridge.training``): grid resume with
  ``--override-opt-param-scheduler`` must reset the checkpoint-restored
  per-param-group ``max_lr``/``min_lr`` (e.g. 1e-5/1e-6) to the configured
  override values (e.g. 0/0) across the nested module / chained optimizer
  structure, and must be a no-op when the override flag is off.

The inner optimizers are fakes mimicking the ``DistributedOptimizer``
sharded-state structure (ShardedObject for ``optimizer`` + ShardedTensor
``gbuf`` entries in ``param_state``); no real optimizer or CUDA is involved.

Run (inside the container, from the FlagScale repo root):

    source /root/miniconda3/bin/activate flagscale
    PYTHONPATH=/workspace/multimodal/Megatron-LM-FL python -m pytest \
        tests/unit_tests/models/test_mimo_checkpointing.py
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

_MEGATRON_REPO = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, "Megatron-LM-FL"))
if os.path.isdir(_MEGATRON_REPO) and _MEGATRON_REPO not in sys.path:
    sys.path.insert(0, _MEGATRON_REPO)

import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.models.mimo.optimizer import ModuleOptimizerInfo
from megatron.core.optimizer.optimizer import ChainedOptimizer as MegatronChainedOptimizer

from flagscale.models.mimo.bridge.training import (
    GridMimoOptimizer,
    _namespace_module_opt_keys,
    _unnamespace_module_opt_keys,
    sync_grid_optimizer_param_group_lr,
)
from flagscale.models.mimo.colocated.mimo_optimizer import ChainedOptimizer


def _pg_mock(tp_rank=0, pp_rank=0, dp_rank=0):
    """Minimal ProcessGroupCollection stand-in with ``rank()`` callables."""
    return SimpleNamespace(
        tp=SimpleNamespace(rank=lambda: tp_rank),
        pp=SimpleNamespace(rank=lambda: pp_rank),
        dp=SimpleNamespace(rank=lambda: dp_rank),
    )


class _FakeDistOptimizer:
    """Mimics the ``DistributedOptimizer.sharded_state_dict`` output shape.

    Produces the exact key families that collide across MIMO modules in the
    wild: a ShardedObject ``optimizer.distributed.dp_group_idx_<mp_rank>.optimizer``
    and ShardedTensors ``...gbuf_idx_0.dtype_0.bucket_idx_0.<state>`` whose
    global shape is the module's whole grad buffer (different per module).
    """

    def __init__(self, module_name, mp_rank, pg_collection, gbuf_world_size, gbuf_seed):
        self.module_name = module_name
        self.mp_rank = mp_rank
        self.pg_collection = pg_collection
        self.gbuf_world_size = gbuf_world_size
        self.gbuf_seed = gbuf_seed
        self.loaded = None
        self.loaded_counts = 0

    @property
    def is_stub_optimizer(self):
        return False

    def _local_gbuf(self, slice_len):
        return torch.arange(slice_len, dtype=torch.float32) + self.gbuf_seed

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, **kwargs):
        # Mirrors DistributedOptimizer.sharded_state_dict: state_dict items
        # (here: the ``optimizer`` param-group metadata) become ShardedObjects
        # keyed by the module-local MP rank; ``param_state`` becomes
        # ShardedTensors of the module's own grad buffer.
        dp_rank = self.pg_collection.dp.rank()
        opt_obj = ShardedObject(
            f"optimizer.distributed.dp_group_idx_{self.mp_rank}.optimizer",
            {"param_groups": [{"lr": 1e-4, "params": []}], "step": 2 + self.gbuf_seed},
            (1,),
            (0,),
            replica_id=(0, 0, dp_rank),
        )
        gbuf_tensor = ShardedTensor(
            f"optimizer.distributed.dp_group_idx_{self.mp_rank}."
            f"gbuf_idx_0.dtype_0.bucket_idx_0.exp_avg",
            self._local_gbuf(self.gbuf_world_size),
            torch.float32,
            (self.gbuf_world_size,),
            (self.gbuf_world_size,),
            (0,),
            axis_fragmentations=None,
            flattened_range=None,
            allow_shape_mismatch=False,
            replica_id=(0, 0, 0),
        )
        return {
            "optimizer": opt_obj,
            "param_state": {0: {"exp_avg": gbuf_tensor}},
            "param_state_sharding_type": "dp_reshardable",
        }

    def load_state_dict(self, module_sd):
        self.loaded = module_sd
        self.loaded_counts += 1


class TestNamespaceHelpers(unittest.TestCase):
    """Direct tests of the key-renaming helpers."""

    def test_namespacing_only_touches_opt_dist_keys(self):
        opt_obj = ShardedObject(
            "optimizer.distributed.dp_group_idx_0.optimizer", {"x": 1}, (1,), (0,)
        )
        gbuf = ShardedTensor(
            "optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_0.bucket_idx_0.exp_avg",
            torch.zeros(4),
            torch.float32,
            (4,),
            (4,),
            (0,),
            axis_fragmentations=None,
        )
        # Already module-scoped (MCore extraction) or model-space keys must be
        # left untouched.
        mimo_pg = ShardedObject("optimizer.mimo.images.param_groups", [{"lr": 1e-4}], (1,), (0,))
        model_space = ShardedTensor(
            "optimizer.state.exp_avg.modality_submodules.images.module.encoder.layers.0.w",
            torch.zeros(2),
            torch.float32,
            (2,),
            (2,),
            (0,),
            axis_fragmentations=None,
        )
        nested = {
            "optimizer": opt_obj,
            "param_state": {0: {"exp_avg": gbuf}, "1": [mimo_pg, model_space]},
        }

        _namespace_module_opt_keys(nested, "images")

        self.assertEqual(opt_obj.key, "optimizer.distributed.images.dp_group_idx_0.optimizer")
        self.assertEqual(
            gbuf.key,
            "optimizer.distributed.images.dp_group_idx_0.gbuf_idx_0.dtype_0.bucket_idx_0.exp_avg",
        )
        # Untouched key families.
        self.assertEqual(mimo_pg.key, "optimizer.mimo.images.param_groups")
        self.assertEqual(
            model_space.key,
            "optimizer.state.exp_avg.modality_submodules.images.module.encoder.layers.0.w",
        )

        _unnamespace_module_opt_keys(nested, "images")
        self.assertEqual(opt_obj.key, "optimizer.distributed.dp_group_idx_0.optimizer")
        self.assertEqual(
            gbuf.key,
            "optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_0.bucket_idx_0.exp_avg",
        )

    def test_namespacing_is_per_module(self):
        sd = {
            "optimizer": ShardedObject(
                "optimizer.distributed.dp_group_idx_0.optimizer", {"x": 1}, (1,), (0,)
            )
        }
        _namespace_module_opt_keys(sd, "language")
        self.assertEqual(
            sd["optimizer"].key, "optimizer.distributed.language.dp_group_idx_0.optimizer"
        )
        _namespace_module_opt_keys(sd, "language")  # idempotent on the same prefix
        self.assertEqual(
            sd["optimizer"].key, "optimizer.distributed.language.dp_group_idx_0.optimizer"
        )

    def test_plain_data_untouched(self):
        sd = {"optimizer": {"param_groups": [{"lr": 1e-4}]}, "param_state": {}}
        _namespace_module_opt_keys(sd, "images")
        _unnamespace_module_opt_keys(sd, "images")
        self.assertEqual(sd["optimizer"]["param_groups"][0]["lr"], 1e-4)


class TestGridMimoOptimizerShardedState(unittest.TestCase):
    """The wrapper must produce globally unique shard keys per module."""

    def _build_optimizer(self):
        images_fake = _FakeDistOptimizer(
            "images",
            mp_rank=0,
            pg_collection=_pg_mock(dp_rank=0),
            gbuf_world_size=64,
            gbuf_seed=1000,
        )
        language_fake = _FakeDistOptimizer(
            "language",
            mp_rank=0,
            pg_collection=_pg_mock(dp_rank=0),
            gbuf_world_size=36,
            gbuf_seed=2000,
        )
        module_infos = {
            "images": ModuleOptimizerInfo(
                optimizer=images_fake,
                grid=None,
                pg_collection=_pg_mock(dp_rank=0),
                is_active=True,
            ),
            "language": ModuleOptimizerInfo(
                optimizer=language_fake,
                grid=None,
                pg_collection=_pg_mock(dp_rank=0),
                is_active=True,
            ),
        }
        config = SimpleNamespace(log_num_zeros_in_grad=False, clip_grad=1.0)
        return GridMimoOptimizer(module_infos, config), images_fake, language_fake

    @staticmethod
    def _collect_shard_keys(sharded_state):
        keys = []

        def _collect(x):
            from megatron.core.dist_checkpointing.mapping import ShardedBase

            if isinstance(x, ShardedBase):
                keys.append(x.key)
            return x

        from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace

        dict_list_map_inplace(_collect, sharded_state)
        return keys

    def test_sharded_state_dict_keys_are_module_namespaced_and_unique(self):
        opt, images_fake, language_fake = self._build_optimizer()
        sharded_state = opt.sharded_state_dict({}, is_loading=False)

        # Both modules present, each nested under its own name.
        self.assertEqual(set(sharded_state.keys()), {"images", "language"})

        keys = self._collect_shard_keys(sharded_state)

        def _module_keys(module):
            # Namespaced distributed-optimizer keys and the MCore-extracted
            # optimizer.mimo.* keys both carry the module name.
            return [
                k
                for k in keys
                if f"optimizer.distributed.{module}." in k or f"optimizer.mimo.{module}." in k
            ]

        images_keys = _module_keys("images")
        language_keys = _module_keys("language")
        self.assertTrue(images_keys)
        self.assertTrue(language_keys)
        # No raw (un-namespaced) distributed-optimizer keys survive.
        for k in keys:
            self.assertFalse(k.startswith("optimizer.distributed.dp_group_idx_"))
        # No key collision between the two modules.
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(set(keys), set(images_keys) | set(language_keys))

        # The module-namespaced keys keep the module's own global gbuf shape.
        images_gbuf = [k for k in images_keys if "gbuf_idx_0" in k]
        language_gbuf = [k for k in language_keys if "gbuf_idx_0" in k]
        self.assertEqual(len(images_gbuf), 1)
        self.assertEqual(len(language_gbuf), 1)

    def test_load_state_dict_routes_each_module_state_to_its_optimizer(self):
        opt, images_fake, language_fake = self._build_optimizer()
        # Post-load state dict: plain tensors, no ShardedBase left (what the
        # torch_dist loader returns).  Each module has its own tensor values.
        loaded_opt = {
            "images": {
                "optimizer": {"param_groups": [{"lr": 1e-4}], "step": 42},
                "param_state": {0: {"exp_avg": torch.full((64,), 1.0)}},
                "param_state_sharding_type": "dp_reshardable",
            },
            "language": {
                "optimizer": {"param_groups": [{"lr": 1e-4}], "step": 43},
                "param_state": {0: {"exp_avg": torch.full((36,), 2.0)}},
                "param_state_sharding_type": "dp_reshardable",
            },
        }
        opt.load_state_dict(loaded_opt)

        self.assertEqual(images_fake.loaded_counts, 1)
        self.assertEqual(language_fake.loaded_counts, 1)
        # Routing: the images optimizer saw only the images state (shape 64),
        # the language optimizer only the language state (shape 36).
        self.assertEqual(images_fake.loaded["param_state"][0]["exp_avg"].shape, (64,))
        self.assertEqual(language_fake.loaded["param_state"][0]["exp_avg"].shape, (36,))
        self.assertEqual(images_fake.loaded["optimizer"]["step"], 42)
        self.assertEqual(language_fake.loaded["optimizer"]["step"], 43)
        # Un-namespacing restores the exact pre-fix key form for any remaining
        # ShardedBase (here none; the delegated dicts are plain data).
        self.assertNotIn("optimizer", loaded_opt["images"].get("_mimo_placeholder", {}))

    def test_load_state_dict_tolerates_missing_and_inactive_modules(self):
        opt, images_fake, language_fake = self._build_optimizer()
        # A module absent from the loaded dict must be skipped (MCore behavior).
        opt.load_state_dict({"images": {"param_state": {}, "optimizer": {}}})
        self.assertEqual(images_fake.loaded_counts, 1)
        self.assertEqual(language_fake.loaded_counts, 0)

    def test_inactive_module_produces_empty_sharded_state(self):
        images_fake = _FakeDistOptimizer(
            "images",
            mp_rank=0,
            pg_collection=_pg_mock(dp_rank=0),
            gbuf_world_size=64,
            gbuf_seed=1000,
        )
        module_infos = {
            "images": ModuleOptimizerInfo(
                optimizer=images_fake, grid=None, pg_collection=_pg_mock(), is_active=True
            ),
            "language": ModuleOptimizerInfo(
                optimizer=None, grid=None, pg_collection=None, is_active=False
            ),
        }
        opt = GridMimoOptimizer(
            module_infos, SimpleNamespace(log_num_zeros_in_grad=False, clip_grad=1.0)
        )
        sharded_state = opt.sharded_state_dict({})
        self.assertEqual(sharded_state["language"], {})
        self.assertTrue(self._collect_shard_keys(sharded_state["images"]))


class TestColocatedChainedOptimizerShardedPrefixes(unittest.TestCase):
    """Colocated ChainedOptimizer: per-optimizer shard-key prefixes."""

    def test_sharded_state_dict_prefixes_and_load_strips(self):
        images_fake = _FakeDistOptimizer(
            "images",
            mp_rank=0,
            pg_collection=_pg_mock(dp_rank=0),
            gbuf_world_size=64,
            gbuf_seed=1000,
        )
        language_fake = _FakeDistOptimizer(
            "language",
            mp_rank=0,
            pg_collection=_pg_mock(dp_rank=0),
            gbuf_world_size=36,
            gbuf_seed=2000,
        )
        chained = ChainedOptimizer([images_fake, language_fake])

        sharded_state_dicts = chained.sharded_state_dict({})
        self.assertEqual(len(sharded_state_dicts), 2)
        images_keys = self._collect_keys(sharded_state_dicts[0])
        language_keys = self._collect_keys(sharded_state_dicts[1])
        for k in images_keys:
            self.assertTrue(k.startswith("chained_0."), k)
        for k in language_keys:
            self.assertTrue(k.startswith("chained_1."), k)
        # No collision across the chain.
        self.assertEqual(
            len(set(images_keys) | set(language_keys)), len(images_keys) + len(language_keys)
        )

        # Simulated post-load dicts: ShardedTensor keys still carry the prefix.
        loaded = [
            {
                "optimizer": {"param_groups": []},
                "param_state": {
                    0: {
                        "exp_avg": ShardedTensor(
                            "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_0"
                            ".dtype_0.bucket_idx_0.exp_avg",
                            torch.full((64,), 1.0),
                            torch.float32,
                            (64,),
                            (64,),
                            (0,),
                            axis_fragmentations=None,
                        )
                    }
                },
                "param_state_sharding_type": "dp_reshardable",
            },
            {
                "optimizer": {"param_groups": []},
                "param_state": {
                    0: {
                        "exp_avg": ShardedTensor(
                            "chained_1.optimizer.distributed.dp_group_idx_0.gbuf_idx_0"
                            ".dtype_0.bucket_idx_0.exp_avg",
                            torch.full((36,), 2.0),
                            torch.float32,
                            (36,),
                            (36,),
                            (0,),
                            axis_fragmentations=None,
                        )
                    }
                },
                "param_state_sharding_type": "dp_reshardable",
            },
        ]
        chained.load_state_dict(loaded)

        self.assertEqual(images_fake.loaded_counts, 1)
        self.assertEqual(language_fake.loaded_counts, 1)
        # Prefix stripped before delegating: un-prefixed DistributedOptimizer key.
        self.assertEqual(
            images_fake.loaded["param_state"][0]["exp_avg"].key,
            "optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_0.bucket_idx_0.exp_avg",
        )
        self.assertEqual(
            language_fake.loaded["param_state"][0]["exp_avg"].key,
            "optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_0.bucket_idx_0.exp_avg",
        )

    def test_plain_torch_state_dicts_pass_through_unchanged(self):
        images_fake = _FakeDistOptimizer(
            "images", mp_rank=0, pg_collection=_pg_mock(), gbuf_world_size=64, gbuf_seed=1
        )
        chained = ChainedOptimizer([images_fake])
        # Non-sharded (legacy torch format) state dicts have no ShardedBase;
        # load must be a no-op pass-through.
        chained.load_state_dict([{"optimizer": {"param_groups": [{"lr": 1e-4}]}}])
        self.assertEqual(images_fake.loaded["optimizer"]["param_groups"][0]["lr"], 1e-4)

    @staticmethod
    def _collect_keys(sharded_state):
        from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
        from megatron.core.dist_checkpointing.mapping import ShardedBase

        keys = []

        def _collect(x):
            if isinstance(x, ShardedBase):
                keys.append(x.key)
            return x

        dict_list_map_inplace(_collect, sharded_state)
        return keys


class TestQwen3VisionModelReplicatedParams(unittest.TestCase):
    """Module-local TP replica coordinate for the replicated vision params.

    ``Qwen3VisionModel``'s direct params (``patch_embed.proj.*``,
    ``pos_embed.weight``) are replicated across the vision TP group.  Their
    sharded metadata must carry the module-local TP rank in the replica_id
    (the ``(0, tp_rank, dp_rank)`` MCore convention); otherwise two TP ranks
    both claim ``replica_id=(0, 0, 0)`` and torch_dist save-time validation
    reports an access count of 2 for the unsharded global tensors.  The
    module-local TP group comes from the ``pg_collection`` threaded into the
    constructor (grid mode), never from the global parallel state.
    """

    def _vision_config(self):
        from flagscale.models.megatron.qwen35.transformer_config import (
            Qwen35TransformerConfig,
        )

        cfg = Qwen35TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=256,
            kv_channels=16,
            num_query_groups=4,
            seq_length=64,
            layernorm_epsilon=1e-6,
            attention_backend="flash",
            use_cpu_initialization=True,
        )
        cfg.patch_size = 16
        cfg.temporal_patch_size = 1
        cfg.in_channels = 3
        cfg.spatial_merge_size = 2
        cfg.num_position_embeddings = 64
        cfg.add_bias_linear = True
        cfg.add_qkv_bias = True
        cfg.gated_linear_unit = False
        cfg.normalization = "LayerNorm"
        cfg.hidden_dropout = 0.0
        cfg.attention_dropout = 0.0
        return cfg

    @staticmethod
    def _pg_mock(tp_rank, dp_rank=0):
        return SimpleNamespace(
            tp=SimpleNamespace(rank=lambda: tp_rank, size=lambda: 2),
            pp=SimpleNamespace(rank=lambda: 0, size=lambda: 1),
            dp=SimpleNamespace(rank=lambda: dp_rank, size=lambda: 1),
            dp_cp=SimpleNamespace(rank=lambda: dp_rank, size=lambda: 1),
        )

    def _mock_pg_helpers(self):
        """Let ``get_pg_rank``/``get_pg_size`` consult SimpleNamespace mocks.

        The real helpers short-circuit to 0/1 when ``torch.distributed`` is
        not initialized, so without this the replica_id computation would not
        see the mock group ranks.  Real process groups are exercised by the
        gloo checkpoint smoke instead.
        """
        import megatron.core.utils as mcore_utils

        real_rank = mcore_utils.get_pg_rank
        real_size = mcore_utils.get_pg_size

        def _rank(group=None):
            if isinstance(group, SimpleNamespace):
                return group.rank()
            return real_rank(group)

        def _size(group=None):
            if isinstance(group, SimpleNamespace):
                return group.size()
            return real_size(group)

        return (
            mock.patch.object(mcore_utils, "get_pg_rank", side_effect=_rank),
            mock.patch.object(mcore_utils, "get_pg_size", side_effect=_size),
        )

    def _build_model(self, pg_collection):
        import torch.nn as nn

        from flagscale.models.megatron.qwen3_vl.vision_model import Qwen3VisionModel

        class _FakeContainer(nn.Module):
            """Stand-in for the heavy decoder block / projection containers."""

            def __init__(self, *args, **kwargs):
                super().__init__()

        with (
            mock.patch(
                "flagscale.models.megatron.qwen3_vl.vision_model.VisionTransformerBlock",
                _FakeContainer,
            ),
            mock.patch(
                "flagscale.models.megatron.qwen3_vl.vision_model.MultimodalProjector",
                _FakeContainer,
            ),
        ):
            return Qwen3VisionModel(
                transformer_config=self._vision_config(),
                transformer_layer_spec=None,
                projection_config=self._vision_config(),
                projection_layer_spec=None,
                projection_type="mlp",
                pre_process=True,
                post_process=True,
                pg_collection=pg_collection,
            )

    def test_tp_group_wired_from_pg_collection(self):
        pg = self._pg_mock(tp_rank=1, dp_rank=0)
        model = self._build_model(pg)
        # MegatronModule.sharded_state_dict keys on ``self.tp_group``; it must
        # be the module-local vision TP group, not the global parallel state.
        self.assertIs(model.tp_group, pg.tp)
        self.assertIs(model.pg_collection, pg)

    def test_no_tp_group_without_pg_collection(self):
        # Non-grid path: no attribute -> MegatronModule falls back to
        # the global parallel state, exactly as before the fix.
        model = self._build_model(None)
        self.assertFalse(hasattr(model, "tp_group"))

    def test_replicated_params_carry_module_local_tp_replica_id(self):
        pg = self._pg_mock(tp_rank=1, dp_rank=0)
        model = self._build_model(pg)
        rank_mock, size_mock = self._mock_pg_helpers()
        with rank_mock, size_mock:
            sd = model.sharded_state_dict(
                prefix="modality_submodules.images.module.encoders.qwen3_vit.",
                metadata={"dp_cp_group": pg.dp_cp},
            )
        for key in (
            "modality_submodules.images.module.encoders.qwen3_vit.patch_embed.proj.weight",
            "modality_submodules.images.module.encoders.qwen3_vit.patch_embed.proj.bias",
            "modality_submodules.images.module.encoders.qwen3_vit.pos_embed.weight",
        ):
            sh = sd[key]
            self.assertEqual(sh.replica_id, (0, 1, 0), key)
            # Replicated: full global tensor, no sharding offsets.
            self.assertEqual(sh.global_offset, (0,) * len(sh.global_shape), key)

    def test_tp0_rank_replica_is_main(self):
        pg = self._pg_mock(tp_rank=0, dp_rank=0)
        model = self._build_model(pg)
        rank_mock, size_mock = self._mock_pg_helpers()
        with rank_mock, size_mock:
            sd = model.sharded_state_dict(
                prefix="modality_submodules.images.module.encoders.qwen3_vit.",
                metadata={"dp_cp_group": pg.dp_cp},
            )
        sh = sd["modality_submodules.images.module.encoders.qwen3_vit.pos_embed.weight"]
        self.assertEqual(sh.replica_id, (0, 0, 0))

    def test_module_local_tp_rank_beats_global_parallel_state(self):
        # The replica_id must come from the module's TP group even when the
        # global parallel-state TP rank would say something else (grid mode
        # forces global TP=1, i.e. rank 0 on every rank).
        pg = self._pg_mock(tp_rank=1, dp_rank=0)
        model = self._build_model(pg)
        rank_mock, size_mock = self._mock_pg_helpers()
        with (
            rank_mock,
            size_mock,
            mock.patch(
                "megatron.core.parallel_state.get_tensor_model_parallel_rank",
                return_value=0,
            ),
        ):
            sd = model.sharded_state_dict(
                prefix="modality_submodules.images.module.encoders.qwen3_vit.",
                metadata={"dp_cp_group": pg.dp_cp},
            )
        sh = sd["modality_submodules.images.module.encoders.qwen3_vit.pos_embed.weight"]
        self.assertEqual(sh.replica_id, (0, 1, 0))


class TestSyncGridOptimizerParamGroupLr(unittest.TestCase):
    """Grid resume with --override-opt-param-scheduler re-syncs param-group LR.

    The optimizer checkpoint load restores every param group's
    ``max_lr``/``min_lr`` from the saved state, and
    ``OptimizerParamScheduler.get_lr`` prefers the param-group values over the
    scheduler's configured fields, so a configured lr/min_lr of 0 would be
    silently ignored after resume.  The sync helper must reset max_lr/min_lr
    on every active inner optimizer param group to the configured override
    values (touching no other hyperparameter) and must be a no-op when the
    override flag is off.
    """

    @staticmethod
    def _make_group(max_lr=1e-5, min_lr=1e-6, lr=1e-6):
        """Param group exactly as the optimizer checkpoint restores it."""
        return {
            "params": [],
            "lr": lr,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "weight_decay": 0.1,
            "wd_mult": 1.0,
            "default_config": True,
        }

    @staticmethod
    def _make_module_optimizer(n_groups=1):
        """DistributedOptimizer-like fake exposing Megatron ``param_groups``."""
        return SimpleNamespace(
            param_groups=[TestSyncGridOptimizerParamGroupLr._make_group() for _ in range(n_groups)]
        )

    @staticmethod
    def _make_grid_optimizer(module_optimizers):
        """Real ``GridMimoOptimizer`` over fake per-module optimizers."""
        module_infos = {
            name: ModuleOptimizerInfo(
                optimizer=opt, grid=None, pg_collection=None, is_active=opt is not None
            )
            for name, opt in module_optimizers.items()
        }
        return GridMimoOptimizer(
            module_infos, SimpleNamespace(log_num_zeros_in_grad=False, clip_grad=1.0)
        )

    def test_resets_max_lr_and_min_lr_on_all_module_param_groups(self):
        images_opt = self._make_module_optimizer(n_groups=2)
        language_opt = self._make_module_optimizer(n_groups=1)
        opt = self._make_grid_optimizer({"images": images_opt, "language": language_opt})
        args = SimpleNamespace(override_opt_param_scheduler=True, lr=0.0, min_lr=0.0)

        changed = sync_grid_optimizer_param_group_lr(opt, args)
        self.assertTrue(changed)

        groups = images_opt.param_groups + language_opt.param_groups
        self.assertEqual(len(groups), 3)
        for group in groups:
            self.assertEqual(group["max_lr"], 0.0)
            self.assertEqual(group["min_lr"], 0.0)
            # Unrelated hyperparameters stay exactly as the checkpoint
            # restored them.
            self.assertEqual(group["lr"], 1e-6)
            self.assertEqual(group["weight_decay"], 0.1)
            self.assertEqual(group["wd_mult"], 1.0)
            self.assertTrue(group["default_config"])

    def test_chained_module_optimizer_inner_groups_reset(self):
        # MoE-style module optimizer: a Megatron ChainedOptimizer with one
        # inner optimizer per parameter partition (dense / experts).
        dense_opt = self._make_module_optimizer()
        expert_opt = self._make_module_optimizer()
        chained = MegatronChainedOptimizer([dense_opt, expert_opt])
        images_opt = self._make_module_optimizer()
        opt = self._make_grid_optimizer({"images": images_opt, "language": chained})
        args = SimpleNamespace(override_opt_param_scheduler=True, lr=0.0, min_lr=0.0)

        sync_grid_optimizer_param_group_lr(opt, args)

        for inner in (dense_opt, expert_opt, images_opt):
            group = inner.param_groups[0]
            self.assertEqual(group["max_lr"], 0.0)
            self.assertEqual(group["min_lr"], 0.0)

    def test_plain_optimizer_reset_is_generic(self):
        # The walk also handles a bare Megatron optimizer (no module nesting).
        plain = self._make_module_optimizer()
        args = SimpleNamespace(override_opt_param_scheduler=True, lr=1e-7, min_lr=1e-8)

        self.assertTrue(sync_grid_optimizer_param_group_lr(plain, args))
        self.assertEqual(plain.param_groups[0]["max_lr"], 1e-7)
        self.assertEqual(plain.param_groups[0]["min_lr"], 1e-8)

    def test_inactive_modules_skipped(self):
        images_opt = self._make_module_optimizer()
        opt = self._make_grid_optimizer({"images": images_opt, "language": None})
        args = SimpleNamespace(override_opt_param_scheduler=True, lr=0.0, min_lr=0.0)

        self.assertTrue(sync_grid_optimizer_param_group_lr(opt, args))
        self.assertEqual(images_opt.param_groups[0]["max_lr"], 0.0)
        self.assertEqual(images_opt.param_groups[0]["min_lr"], 0.0)

    def test_none_optimizer_noop(self):
        args = SimpleNamespace(override_opt_param_scheduler=True, lr=0.0, min_lr=0.0)
        self.assertFalse(sync_grid_optimizer_param_group_lr(None, args))

    def test_noop_when_override_opt_param_scheduler_false(self):
        images_opt = self._make_module_optimizer()
        language_opt = self._make_module_optimizer()
        opt = self._make_grid_optimizer({"images": images_opt, "language": language_opt})
        args = SimpleNamespace(override_opt_param_scheduler=False, lr=0.0, min_lr=0.0)

        self.assertFalse(sync_grid_optimizer_param_group_lr(opt, args))
        for group in images_opt.param_groups + language_opt.param_groups:
            # Checkpoint-restored values stay untouched.
            self.assertEqual(group["max_lr"], 1e-5)
            self.assertEqual(group["min_lr"], 1e-6)


if __name__ == "__main__":
    unittest.main()

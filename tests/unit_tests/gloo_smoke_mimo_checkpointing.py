# Copyright (c) 2026, BAAI. All rights reserved.

"""Gloo distributed checkpoint smoke for the module-namespaced MIMO optimizer.

Distributed smoke (CPU / gloo backend, no GPUs) that exercises the *real*
torch_dist save/load machinery (``megatron.core.dist_checkpointing``) on the
combined sharded optimizer state of the images and language modules of the
non-colocated grid MIMO layout.  This is the exact failure mode of the 8-GPU
exit-save: both modules' inner ``DistributedOptimizer``s emit identical
``optimizer.distributed.dp_group_idx_*`` shard keys, so the un-namespaced
state fails save-time sharding validation with duplicate ShardedObject keys
and ShardedTensor global-shape mismatches (e.g. 752394752 vs 51800320).

Steps per world size (2 and 8; family 1 on world 8: images [0, 2) TP2/DP1,
language [2, 8) TP1/PP1/DP6):

1. Build the real grid infra (``build_mimo_infra``) and fake per-module
   optimizers mimicking the ``DistributedOptimizer`` sharded-state structure
   (ShardedObject ``optimizer`` + ``gbuf`` ShardedTensors whose global shape
   is the module's own grad buffer).
2. Pre-fix sanity: the *un-namespaced* MCore ``MimoOptimizer`` state must fail
   ``validate_sharding_integrity`` (reproduces the reported failure).
3. The namespaced ``GridMimoOptimizer`` state must pass the same validation,
   save a real torch_dist checkpoint, load it back, and route each module's
   state to its own optimizer (per-module values and DP replica slices).

Run (from the FlagScale repo root):

    torchrun --nproc-per-node 2 --master-port 29557 \\
        tests/unit_tests/gloo_smoke_mimo_checkpointing.py --world-size 2
    torchrun --nproc-per-node 8 --master-port 29558 \\
        tests/unit_tests/gloo_smoke_mimo_checkpointing.py --world-size 8

or use ``tests/run_mimo_checkpointing_gloo_smoke.sh``.
"""

import argparse
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.core.dist_checkpointing import (
    load as dcp_load,
    save as dcp_save,
)
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.validation import (
    determine_global_metadata,
    validate_sharding_integrity,
)
from megatron.core.models.mimo.optimizer import (
    MimoOptimizer,
    ModuleOptimizerInfo,
    _get_replica_id,
)

from flagscale.models.mimo.bridge.infra import build_mimo_infra
from flagscale.models.mimo.bridge.parallelism import (
    ModuleParallelismConfig,
)
from flagscale.models.mimo.bridge.training import GridMimoOptimizer

#: Per-module global grad-buffer sizes (mirrors the failing run: different
#: sizes per module under identical un-namespaced shard keys).
MODULE_GBUF_WORLD = {"images": 64, "language": 36}
#: Per-module value seeds (make every module's state distinguishable).
MODULE_SEED = {"images": 1000, "language": 2000}


def _build_config(world_size: int):
    if world_size == 2:
        return {
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1),
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=1,
            ),
        }
    if world_size == 8:
        # Family 1: V TP2/DP1 on [0, 2), L TP1/PP1/DP6 on [2, 8).
        return {
            "images": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                data_parallel_size=6,
                rank_offset=2,
            ),
        }
    raise ValueError(f"unsupported smoke world size {world_size}")


class _FakeDistOptimizer:
    """Mimics ``DistributedOptimizer.sharded_state_dict`` output for one module.

    Emits the exact key families that collide across MIMO modules: a
    ShardedObject ``optimizer.distributed.dp_group_idx_<mp_rank>.optimizer``
    (replica per DP rank) and ``gbuf`` ShardedTensors whose global shape is
    the module's whole grad buffer, DP-sliced per rank (replica (0, 0, 0),
    matching ``sharded_param_state_dp_reshardable``).
    """

    def __init__(self, module_name, mp_rank, pg_collection):
        self.module_name = module_name
        self.mp_rank = mp_rank
        self.pg_collection = pg_collection
        self.gbuf_world_size = MODULE_GBUF_WORLD[module_name]
        self.seed = MODULE_SEED[module_name]
        self.dp_rank = pg_collection.dp.rank()
        self.dp_size = pg_collection.dp.size()
        assert self.gbuf_world_size % self.dp_size == 0
        self.gbuf_local_size = self.gbuf_world_size // self.dp_size
        self.loaded = None

    @property
    def is_stub_optimizer(self):
        return False

    def sharded_state_dict(self, model_sharded_state_dict=None, is_loading=False, **kwargs):
        _get_replica_id(self.pg_collection)
        opt_obj = ShardedObject(
            f"optimizer.distributed.dp_group_idx_{self.mp_rank}.optimizer",
            {"param_groups": [{"lr": 1e-4, "params": []}], "step": self.seed},
            (1,),
            (0,),
            replica_id=(0, 0, self.dp_rank),
        )
        local_start = self.dp_rank * self.gbuf_local_size
        local = torch.arange(self.gbuf_local_size, dtype=torch.float32) + (self.seed + local_start)
        gbuf_tensor = ShardedTensor(
            f"optimizer.distributed.dp_group_idx_{self.mp_rank}."
            f"gbuf_idx_0.dtype_0.bucket_idx_0.exp_avg",
            local,
            torch.float32,
            (self.gbuf_local_size,),
            (self.gbuf_world_size,),
            (local_start,),
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

    def expected_local(self):
        local_start = self.dp_rank * self.gbuf_local_size
        return torch.arange(self.gbuf_local_size, dtype=torch.float32) + (self.seed + local_start)


def _build_module_infos(infra, optimizer_cls=None):
    """Per-module ModuleOptimizerInfo with fakes on active ranks."""
    module_infos = {}
    fakes = {}
    for module_name, pg_collection in infra.module_to_pg_collection.items():
        grid = infra.module_to_grid_map[module_name]
        if pg_collection is None:
            module_infos[module_name] = ModuleOptimizerInfo(
                optimizer=None, grid=grid, pg_collection=None, is_active=False
            )
            continue
        mp_rank = pg_collection.tp.rank() * pg_collection.pp.size() + pg_collection.pp.rank()
        fake = _FakeDistOptimizer(module_name, mp_rank, pg_collection)
        module_infos[module_name] = ModuleOptimizerInfo(
            optimizer=fake, grid=grid, pg_collection=pg_collection, is_active=True
        )
        fakes[module_name] = fake
    return module_infos, fakes


def _config():
    return SimpleNamespace(log_num_zeros_in_grad=False, clip_grad=1.0)


def _pre_fix_validation_must_fail(infra, rank):
    """Reproduce the reported failure on the un-namespaced MCore state."""
    module_infos, _ = _build_module_infos(infra)
    plain = MimoOptimizer(module_infos, _config())
    sharded_state = plain.sharded_state_dict({})
    _, global_metadata = determine_global_metadata(sharded_state)
    if rank == 0:
        from megatron.core.dist_checkpointing.core import CheckpointingException

        try:
            validate_sharding_integrity(global_metadata)
        except (CheckpointingException, AssertionError) as e:
            print(f"rank {rank}: pre-fix validation failed as expected: {type(e).__name__}")
        else:
            raise AssertionError(
                "un-namespaced MIMO sharded state passed validation; "
                "the reported collision is not reproduced"
            )


def _verify_loaded(loaded_opt, fakes, rank):
    """Per-module routing + values of the loaded optimizer state."""
    for module_name, fake in fakes.items():
        module_sd = loaded_opt[module_name]
        # ShardedObject payload was replicated to every rank by the loader.
        assert module_sd["optimizer"]["step"] == MODULE_SEED[module_name], (
            module_name,
            module_sd["optimizer"],
        )
        # DP-local slice of the module's own gbuf.
        loaded_gbuf = module_sd["param_state"][0]["exp_avg"]
        assert isinstance(loaded_gbuf, torch.Tensor), type(loaded_gbuf)
        assert loaded_gbuf.shape == (fake.gbuf_local_size,), (
            module_name,
            loaded_gbuf.shape,
            fake.gbuf_local_size,
        )
        assert torch.equal(loaded_gbuf, fake.expected_local()), (
            module_name,
            rank,
            loaded_gbuf,
            fake.expected_local(),
        )


def _build_vision_config():
    """Tiny CPU-friendly Qwen3.5 vision encoder config."""
    from flagscale.models.megatron.qwen35.transformer_config import Qwen35TransformerConfig

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


def _build_real_vision_model(pg_collection):
    """Build the REAL ``Qwen3VisionModel`` (patch_embed + pos_embed + TE layers)."""
    from flagscale.models.megatron.qwen3_vl.layer_specs import (
        get_mlp_module_spec,
        get_qwen3vl_vision_model_spec,
    )
    from flagscale.models.megatron.qwen3_vl.vision_model import Qwen3VisionModel
    from flagscale.models.megatron.qwen35.transformer_config import Qwen35TransformerConfig

    cfg = _build_vision_config()
    proj_cfg = Qwen35TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=256,
        seq_length=64,
        kv_channels=16,
        num_query_groups=4,
    )
    model = Qwen3VisionModel(
        transformer_config=cfg,
        transformer_layer_spec=get_qwen3vl_vision_model_spec(),
        projection_config=proj_cfg,
        projection_layer_spec=get_mlp_module_spec(add_norm=False).submodules,
        projection_type="mlp",
        pre_process=True,
        post_process=True,
        pg_collection=pg_collection,
    )
    return model


def _vision_sharded_state(model, images_pg):
    """Sharded state of the images encoder under its module-local dp_cp group."""
    from megatron.core.models.mimo.submodules.base import ModalitySubmodules

    prefix = "model.modality_submodules.images.module.encoders.qwen3_vit."
    if isinstance(model, ModalitySubmodules):
        return model.sharded_state_dict(
            prefix="model.modality_submodules.images.module.",
            metadata={"dp_cp_group": images_pg.dp_cp},
        )
    return model.sharded_state_dict(prefix=prefix, metadata={"dp_cp_group": images_pg.dp_cp})


def _vision_sharding_phase(infra, rank, world_size, ckpt_dir):
    """Real TP2 vision shards through validation and save/load.

    Reproduces the second reported failure: with vision TP2, the *replicated*
    vision params (``patch_embed.proj.weight/bias``, ``pos_embed.weight``)
    were all tagged ``replica_id=(0, 0, 0)`` (global parallel state is TP=1 in
    grid mode), so save-time validation reported an access count of 2 for the
    unsharded global tensors.  After the fix the module-local vision TP rank
    is encoded in the replica_id, while the true TP-sharded layer params stay
    fragments (different offsets, not replicas).
    """
    from megatron.core.dist_checkpointing.mapping import (
        ShardedBase,
    )

    images_pg = infra.module_to_pg_collection["images"]
    model = _build_real_vision_model(images_pg) if images_pg is not None else None

    # Model shards are only present on the images ranks; other ranks
    # contribute an empty model state (validation only inspects the sharded
    # part, and common-state differences are warnings in MCore).
    model_sd = _vision_sharded_state(model, images_pg) if model is not None else {}

    if world_size == 8 and images_pg is not None:
        # Replicated params: module-local TP coordinate in replica_id.
        for key in (
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "pos_embed.weight",
        ):
            sh = model_sd[f"model.modality_submodules.images.module.encoders.qwen3_vit.{key}"]
            assert isinstance(sh, ShardedBase), (key, type(sh))
            assert sh.replica_id[1] == images_pg.tp.rank(), (key, sh.replica_id)
            assert sh.replica_id[2] == images_pg.dp.rank(), (key, sh.replica_id)
        # True TP-sharded layer params: replica_id (0,0,0) but fragments
        # (per-rank offsets) - NOT replicas.
        qkv_key = (
            "model.modality_submodules.images.module.encoders.qwen3_vit.decoder.layers.0"
            ".self_attention.linear_qkv.weight"
        )
        qkv = model_sd[qkv_key]
        assert qkv.replica_id == (0, 0, 0), qkv.replica_id
        assert qkv.global_offset[0] == images_pg.tp.rank() * (qkv.global_shape[0] // 2), (
            qkv.global_offset,
            qkv.global_shape,
        )

    # Full save with sharding-integrity validation (the failing step).
    sharded_state = {"args": None, "iteration": 1, "model": model_sd}
    dcp_save(sharded_state, ckpt_dir, async_strategy="mcore")

    # Load back with the same request; every images rank must receive the full
    # replicated tensors and its own TP slice of the fragment tensors.  Ranks
    # without the vision module have no model state in the checkpoint (their
    # common skeleton is not persisted), so they skip the value checks.
    load_request = {"args": None, "iteration": 1, "model": model_sd}
    loaded = dcp_load(load_request, ckpt_dir, strict="assume_ok_unexpected")
    loaded_model = loaded.get("model", {})

    if model is not None:
        for key in (
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "pos_embed.weight",
        ):
            full_key = f"model.modality_submodules.images.module.encoders.qwen3_vit.{key}"
            loaded_ten = loaded_model[full_key]
            model_ten = dict(model.state_dict())[key]  # full local copy
            assert isinstance(loaded_ten, torch.Tensor), (full_key, type(loaded_ten))
            assert torch.equal(loaded_ten, model_ten), (full_key, loaded_ten.shape)
        # TP fragment: local slice matches the local shard.
        qkv_key = (
            "model.modality_submodules.images.module.encoders.qwen3_vit.decoder.layers.0"
            ".self_attention.linear_qkv.weight"
        )
        loaded_qkv = loaded_model[qkv_key]
        model_qkv = dict(model.state_dict())["decoder.layers.0.self_attention.linear_qkv.weight"]
        assert torch.equal(loaded_qkv, model_qkv), (qkv_key, loaded_qkv.shape, model_qkv.shape)

    print(f"rank {rank}: world {world_size} vision TP sharding validation PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--keep-ckpt-dir", action="store_true", help="keep the temp checkpoint dir")
    args = parser.parse_args()
    world_size = args.world_size

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == world_size, (
        f"torchrun world ({dist.get_world_size()}) != --world-size ({world_size})"
    )

    infra = build_mimo_infra(_build_config(world_size))

    # 1. The un-namespaced MCore state must reproduce the reported failure.
    _pre_fix_validation_must_fail(infra, rank)

    # 2. Namespaced state: unique per-module keys + real save/load round trip.
    module_infos, fakes = _build_module_infos(infra)
    opt = GridMimoOptimizer(module_infos, _config())

    ckpt_dir = None
    if rank == 0:
        ckpt_dir = tempfile.mkdtemp(prefix="mimo_ckpt_gloo_")
    object_list = [ckpt_dir]
    dist.broadcast_object_list(object_list, src=0)
    ckpt_dir = object_list[0]
    assert ckpt_dir is not None

    try:
        # Save with full sharding-integrity validation (the failing step).
        sharded_state = {"args": None, "iteration": 1, "optimizer": opt.sharded_state_dict({})}
        dcp_save(sharded_state, ckpt_dir, async_strategy="mcore")

        # Load back with the same (is_loading=True) namespaced request.
        load_request = {
            "args": None,
            "iteration": 1,
            "optimizer": opt.sharded_state_dict({}, is_loading=True),
        }
        loaded = dcp_load(
            load_request,
            ckpt_dir,
            strict="assume_ok_unexpected",
        )

        _verify_loaded(loaded["optimizer"], fakes, rank)

        # 3. load_state_dict routes each module's state to its own optimizer.
        opt.load_state_dict(loaded["optimizer"])
        for module_name, fake in fakes.items():
            assert fake.loaded is not None, module_name
            assert fake.loaded["optimizer"]["step"] == MODULE_SEED[module_name]
            assert torch.equal(fake.loaded["param_state"][0]["exp_avg"], fake.expected_local())
            # Routing: the module optimizer must NOT see the other module's state.
            other_seeds = set(MODULE_SEED.values()) - {MODULE_SEED[module_name]}
            assert fake.loaded["optimizer"]["step"] not in other_seeds, module_name

        # 4. Real TP2 vision model shards: replicated params carry the
        #    module-local TP replica coordinate, TP-sharded params stay
        #    fragments; full save/load with validation.
        _vision_sharding_phase(infra, rank, world_size, ckpt_dir)

        print(f"rank {rank}: world {world_size} MIMO checkpoint gloo smoke PASSED")
    finally:
        infra.destroy()
        dist.destroy_process_group()
        if not args.keep_ckpt_dir and rank == 0 and ckpt_dir:
            shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

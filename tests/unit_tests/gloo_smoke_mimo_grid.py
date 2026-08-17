# Copyright (c) 2026, BAAI. All rights reserved.

"""Gloo process-group smoke for the non-colocated grid MIMO infra.

Distributed smoke that exercises the *real* collective code path of
``flagscale.models.mimo.bridge.infra`` (deterministic group creation in
global module order on every world rank, nullable per-module
``ProcessGroupCollection``) plus the schedule-facing
``MultiModuleProcessGroupCollection``.  Runs on CPU with the gloo backend; no
model is built and no CUDA tensors are used (the MCore bridge communicator
itself requires nccl + CUDA and is therefore out of scope for this smoke -
it is covered by the 8-GPU gates).

Layouts exercised:

- world 2: images on [0, 1) (TP1/DP1), language on [1, 2) (TP1/DP1).
- world 8: the supported family 1 - images on [0, 2) (TP2/DP1), language on
  [2, 8) (TP1/PP1/DP6).

Run (from the FlagScale repo root):

    torchrun --nproc-per-node 2 --master-port 29555 \\
        tests/unit_tests/gloo_smoke_mimo_grid.py --world-size 2
    torchrun --nproc-per-node 8 --master-port 29555 \\
        tests/unit_tests/gloo_smoke_mimo_grid.py --world-size 8

or use ``tests/run_mimo_grid_gloo_smoke.sh``.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flagscale.models.mimo.bridge.infra import build_mimo_infra
from flagscale.models.mimo.bridge.parallelism import (
    ModuleParallelismConfig,
)
from flagscale.models.mimo.bridge.runtime import (
    build_pg_collection_for_schedule,
    get_active_module_pg,
    validate_no_stub_ranks,
)


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


def _group_ranks(group):
    """Global ranks of a live process group (no ``.ranks`` attr on real PGs)."""
    return sorted(dist.get_global_rank(group, i) for i in range(group.size()))


def _sanity_collective(infra, module_name):
    """All-reduce over a module's DP/TP/PP groups (membership sanity)."""
    pg = infra.module_to_pg_collection[module_name]
    if pg is None:
        return
    for group in (pg.dp, pg.tp, pg.pp):
        t = torch.ones(1, dtype=torch.float32)
        dist.all_reduce(t, group=group)
        expected = float(group.size())
        assert torch.allclose(t, torch.full_like(t, expected)), (
            f"all-reduce over {module_name} group of size {group.size()} returned {t.item()}"
        )


def _assert_no_int_fields(pg_collection):
    """Every PG field must be a real process group or None.

    Regression: ``dist.new_group`` returns ``GroupMember.NON_GROUP_MEMBER``
    (-100, an int) to non-member ranks; the nullable contract must map it to
    None on every field (embd/pos_embd/gloo/expert included), otherwise
    ``get_pg_size`` crashes with AttributeError: 'int' object has no
    attribute 'size' during ``finalize_model_grads``.
    """
    for name, value in vars(pg_collection).items():
        if isinstance(value, list):
            for item in value:
                assert not isinstance(item, int), f"PG field {name} contains int {item!r}"
        else:
            assert not isinstance(value, int), f"PG field {name} is int {value!r}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    world_size = args.world_size

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == world_size, (
        f"torchrun world ({dist.get_world_size()}) != --world-size ({world_size})"
    )

    configs = _build_config(world_size)
    # Collective on every rank: grids + nullable per-module PG collections.
    infra = build_mimo_infra(configs)
    validate_no_stub_ranks(infra.module_to_grid_map, world_size)

    # Membership: images on [0, images_max), language on the rest.
    images_max = 2 if world_size == 8 else 1
    images_expected = rank < images_max
    language_expected = rank >= images_max
    assert infra.module_to_grid_map["images"].is_current_rank_in_grid() == images_expected
    assert infra.module_to_grid_map["language"].is_current_rank_in_grid() == language_expected
    assert (infra.module_to_pg_collection["images"] is not None) == images_expected
    assert (infra.module_to_pg_collection["language"] is not None) == language_expected
    active = [name for name, pg in infra.module_to_pg_collection.items() if pg is not None]
    assert len(active) == 1, f"expected exactly one active module, got {active}"
    assert infra.current_module_names() == active

    # Group memberships must match the layout.
    images_pg = infra.module_to_pg_collection["images"]
    if images_pg is not None:
        if world_size == 8:
            assert _group_ranks(images_pg.tp) == [0, 1]
            assert _group_ranks(images_pg.dp) == [rank]
            assert _group_ranks(images_pg.pp) == [rank]
            assert _group_ranks(images_pg.tp_dp_cp) == [0, 1]
        else:
            assert _group_ranks(images_pg.tp) == [0]
    language_pg = infra.module_to_pg_collection["language"]
    if language_pg is not None and world_size == 8:
        assert _group_ranks(language_pg.dp) == list(range(2, 8))
        assert _group_ranks(language_pg.tp) == [rank]
        assert _group_ranks(language_pg.pp) == [rank]

    # Real collectives over the module groups.
    _sanity_collective(infra, "images")
    _sanity_collective(infra, "language")

    # No PG field may be the NON_GROUP_MEMBER int sentinel (real gloo
    # dist.new_group returns it to non-members): every field must be a real
    # group or None, embd/pos_embd/gloo included.
    for module_name, pg in infra.module_to_pg_collection.items():
        if pg is not None:
            _assert_no_int_fields(pg)

    # Schedule-facing collection: exactly one active module per rank.
    multimodule_pg = build_pg_collection_for_schedule(infra.module_to_pg_collection)
    assert multimodule_pg.language_model_module_name == ("language" if language_expected else None)
    assert multimodule_pg.has_language_model() == language_expected
    active_name, local_pg = get_active_module_pg(infra.module_to_pg_collection)
    assert (active_name == "images") == images_expected
    assert (active_name == "language") == language_expected
    assert local_pg is not None

    if rank == 0:
        print(f"world {world_size}: grid gloo smoke PASSED")
    infra.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

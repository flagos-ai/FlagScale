# Copyright (c) 2025, BAAI. All rights reserved.

"""Process group utilities for colocated MIMO deployment."""

import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection

from .mimo_config import ModuleParallelismConfig


def _validate_module_parallelism(
    module_name: str,
    cfg: ModuleParallelismConfig,
    world_size: int,
):
    """Ensure module parallelism sizes multiply to world size and CP is 1."""
    assert cfg.context_parallel_size == 1, (
        f"{module_name}: context parallelism must be 1 for colocated MIMO."
    )
    product = (
        cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size * cfg.data_parallel_size
    )
    assert product == world_size, (
        f"{module_name}: TP*PP*DP = {product} != world_size {world_size} "
        f"(TP={cfg.tensor_model_parallel_size}, PP={cfg.pipeline_model_parallel_size}, "
        f"DP={cfg.data_parallel_size})"
    )


def _compute_rank_groups(
    world_size: int,
    tp_size: int,
    pp_size: int,
    dp_size: int,
) -> dict[str, list[list[int]]]:
    """Compute colocated rank groups using tp-dp-pp ordering.

    Returns a dict mapping group name to a list of rank lists.
    """
    tp_groups: list[list[int]] = []
    dp_groups: list[list[int]] = []
    pp_groups: list[list[int]] = []
    mp_groups: list[list[int]] = []

    # TP groups: fixed (dp, pp), vary tp.
    for pp_rank in range(pp_size):
        for dp_rank in range(dp_size):
            ranks = [
                pp_rank * tp_size * dp_size + dp_rank * tp_size + tp_rank
                for tp_rank in range(tp_size)
            ]
            tp_groups.append(ranks)

    # DP groups: fixed (tp, pp), vary dp.
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            ranks = [
                pp_rank * tp_size * dp_size + dp_rank * tp_size + tp_rank
                for dp_rank in range(dp_size)
            ]
            dp_groups.append(ranks)

    # PP groups: fixed (tp, dp), vary pp.
    for dp_rank in range(dp_size):
        for tp_rank in range(tp_size):
            ranks = [
                pp_rank * tp_size * dp_size + dp_rank * tp_size + tp_rank
                for pp_rank in range(pp_size)
            ]
            pp_groups.append(ranks)

    # MP (tp-pp) groups: fixed dp, vary (tp, pp).
    for dp_rank in range(dp_size):
        ranks = []
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                ranks.append(pp_rank * tp_size * dp_size + dp_rank * tp_size + tp_rank)
        mp_groups.append(ranks)

    # TP+DP groups: fixed pp, vary (tp, dp).
    tp_dp_groups: list[list[int]] = []
    for pp_rank in range(pp_size):
        tp_dp_groups.append([pp_rank * tp_size * dp_size + i for i in range(tp_size * dp_size)])

    return {
        "tp": tp_groups,
        "dp": dp_groups,
        "pp": pp_groups,
        "mp": mp_groups,
        "tp_dp": tp_dp_groups,
    }


def _create_module_pg_collection(
    module_name: str,
    cfg: ModuleParallelismConfig,
    world_size: int,
    build_embedding_groups: bool = False,
) -> tuple[ProcessGroupCollection, torch.distributed.ProcessGroup, torch.distributed.ProcessGroup]:
    """Create all process groups for a single module and return the collection.

    The second and third return values are the singleton group standing in
    for the embedding and position-embedding groups; callers may discard them.
    """
    rank = dist.get_rank()
    groups = _compute_rank_groups(
        world_size,
        cfg.tensor_model_parallel_size,
        cfg.pipeline_model_parallel_size,
        cfg.data_parallel_size,
    )

    pg_collection = ProcessGroupCollection()

    # Create groups in a fixed order across all ranks.
    for ranks in groups["tp"]:
        group = dist.new_group(ranks)
        if rank in ranks:
            pg_collection.tp = group

    for ranks in groups["pp"]:
        group = dist.new_group(ranks)
        if rank in ranks:
            pg_collection.pp = group

    for ranks in groups["dp"]:
        group = dist.new_group(ranks)
        gloo_group = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            pg_collection.dp = group
            pg_collection.dp_gloo = gloo_group

    for ranks in groups["mp"]:
        group = dist.new_group(ranks)
        if rank in ranks:
            pg_collection.mp = group

    # TP+DP groups (fixed pp), matching Megatron's
    # ``get_tensor_and_data_parallel_group`` semantics (with CP=1 the
    # with/without-CP variants coincide).
    for ranks in groups["tp_dp"]:
        group = dist.new_group(ranks)
        if rank in ranks:
            pg_collection.tp_dp_cp = group

    # Singleton groups for all unused dimensions (CP=1, EP=1).  Every rank
    # must participate in the same set of ``new_group`` calls, so we create
    # one singleton group per global rank in a deterministic order and keep
    # the group that contains the current rank.
    singleton_group = None
    for r in range(world_size):
        group = dist.new_group(ranks=[r])
        if r == rank:
            singleton_group = group
    assert singleton_group is not None, "failed to create singleton group for current rank"

    if build_embedding_groups and cfg.pipeline_model_parallel_size > 1:
        # Tied embedding/output weights live on the first and last PP ranks;
        # the init-time weight sync needs a real group over each PP column's
        # endpoint pair.  Middle stages belong to no endpoint pair and keep
        # the singleton (their all-reduce is a no-op), matching vanilla.
        embd_group = singleton_group
        for ranks in groups["pp"]:
            endpoints = [ranks[0], ranks[-1]]
            group = dist.new_group(endpoints)
            if rank in endpoints:
                embd_group = group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = embd_group
    else:
        pg_collection.embd = singleton_group
        pg_collection.pos_embd = singleton_group
    pg_collection.cp = singleton_group
    pg_collection.tp_cp = singleton_group
    pg_collection.hcp = [singleton_group]

    # Expert groups.  EP subdivides the module's DP domain following
    # Megatron's expert RankGenerator ('tp-ep-dp-pp') semantics: with
    # dp = ep * edp, the DP index d decomposes as d = edp_idx * ep + ep_idx.
    # The vision module is dense (ep == 1) and keeps singleton fallbacks.
    ep_size = cfg.expert_model_parallel_size
    if ep_size > 1:
        tp_size = cfg.tensor_model_parallel_size
        pp_size = cfg.pipeline_model_parallel_size
        dp_size = cfg.data_parallel_size
        assert dp_size % ep_size == 0, f"{module_name}: EP ({ep_size}) must divide DP ({dp_size})."
        edp_size = dp_size // ep_size

        def _rank(pp_idx: int, d_idx: int, tp_idx: int) -> int:
            return pp_idx * tp_size * dp_size + d_idx * tp_size + tp_idx

        # EP groups: fixed (pp, tp, edp), vary ep — consecutive DP blocks.
        for pp_idx in range(pp_size):
            for tp_idx in range(tp_size):
                for edp_idx in range(edp_size):
                    ranks = [_rank(pp_idx, edp_idx * ep_size + e, tp_idx) for e in range(ep_size)]
                    group = dist.new_group(ranks)
                    if rank in ranks:
                        pg_collection.ep = group

        # Expert DP groups: fixed (pp, tp, ep), vary edp — strided by ep.
        for pp_idx in range(pp_size):
            for tp_idx in range(tp_size):
                for e in range(ep_size):
                    ranks = [
                        _rank(pp_idx, edp_idx * ep_size + e, tp_idx) for edp_idx in range(edp_size)
                    ]
                    group = dist.new_group(ranks)
                    gloo_group = dist.new_group(ranks, backend="gloo")
                    if rank in ranks:
                        pg_collection.expt_dp = group
                        pg_collection.intra_expt_dp = group
                        pg_collection.expt_dp_gloo = gloo_group

        # TPxEP groups: fixed (pp, edp), vary (tp, ep).
        for pp_idx in range(pp_size):
            for edp_idx in range(edp_size):
                ranks = [
                    _rank(pp_idx, edp_idx * ep_size + e, tp_idx)
                    for tp_idx in range(tp_size)
                    for e in range(ep_size)
                ]
                group = dist.new_group(ranks)
                if rank in ranks:
                    pg_collection.tp_ep = group

        # TPxEPxPP groups: fixed edp, vary (tp, ep, pp).
        for edp_idx in range(edp_size):
            ranks = [
                _rank(pp_idx, edp_idx * ep_size + e, tp_idx)
                for pp_idx in range(pp_size)
                for tp_idx in range(tp_size)
                for e in range(ep_size)
            ]
            group = dist.new_group(ranks)
            if rank in ranks:
                pg_collection.tp_ep_pp = group

        # Expert TP groups coincide with the module's TP groups (fixed d =
        # fixed (edp, ep)), so the existing TP group is the right object.
        pg_collection.expt_tp = pg_collection.tp
    else:
        pg_collection.ep = singleton_group
        pg_collection.expt_tp = singleton_group
        pg_collection.tp_ep = singleton_group
        pg_collection.tp_ep_pp = singleton_group
        pg_collection.expt_dp = singleton_group
        pg_collection.intra_expt_dp = singleton_group

    # Data-parallel groups with CP=1 alias the DP group.
    pg_collection.dp_cp = pg_collection.dp
    pg_collection.intra_dp_cp = pg_collection.dp

    # Distributed-optimizer groups: map model-parallel to intra and
    # data-parallel to inter, matching Megatron-LM-FL expectations.
    pg_collection.intra_dist_opt = pg_collection.mp
    pg_collection.inter_dist_opt = pg_collection.dp

    return pg_collection, singleton_group, singleton_group


def _validate_colocated_rank_mapping(
    vision_pg: ProcessGroupCollection,
    language_pg: ProcessGroupCollection,
    world_size: int,
):
    """Verify that language TP-first ranks map one-to-one to vision DP ranks.

    ``mimo_bridge.get_source_vision_rank`` assumes that the first rank of each
    language TP group is also a valid vision rank, and that cycling through
    these first ranks covers all vision DP replicas.  This function checks the
    assumption and fails early if the rank layout is unexpected.
    """
    language_tp_first_ranks = set()
    for ranks in _compute_rank_groups(
        world_size,
        dist.get_world_size(language_pg.tp),
        dist.get_world_size(language_pg.pp),
        dist.get_world_size(language_pg.dp),
    )["tp"]:
        language_tp_first_ranks.add(ranks[0])

    vision_dp_groups = _compute_rank_groups(
        world_size,
        dist.get_world_size(vision_pg.tp),
        dist.get_world_size(vision_pg.pp),
        dist.get_world_size(vision_pg.dp),
    )["dp"]

    for rank in sorted(language_tp_first_ranks):
        matches = [g for g in vision_dp_groups if rank in g]
        assert len(matches) == 1, (
            f"language TP-first rank {rank} matches {len(matches)} vision DP groups; "
            f"expected exactly 1. Rank layout is incompatible with colocated MIMO."
        )

    all_vision_ranks = set()
    for g in vision_dp_groups:
        all_vision_ranks.update(g)
    assert language_tp_first_ranks.issubset(all_vision_ranks), (
        "language TP-first ranks are not a subset of vision ranks"
    )


def build_colocated_pg_collections(
    vision_parallelism: ModuleParallelismConfig,
    language_parallelism: ModuleParallelismConfig,
    world_size: int,
) -> dict[str, ProcessGroupCollection]:
    """Build ProcessGroupCollection objects for colocated vision and language modules.

    Args:
        vision_parallelism: Parallelism config for the vision module.
        language_parallelism: Parallelism config for the language module.
        world_size: Total number of ranks.

    Returns:
        Dict mapping module names to ProcessGroupCollection.
    """
    _validate_module_parallelism("vision", vision_parallelism, world_size)
    _validate_module_parallelism("language", language_parallelism, world_size)

    # Create vision groups first, then language groups, to keep a deterministic
    # global creation order across all ranks.  Only the language module needs
    # real embedding groups (tied embedding/output weights across PP stages);
    # the vision module is never pipelined and keeps singletons.
    vision_pg, _, _ = _create_module_pg_collection("vision", vision_parallelism, world_size)
    language_pg, _, _ = _create_module_pg_collection(
        "language", language_parallelism, world_size, build_embedding_groups=True
    )

    _validate_colocated_rank_mapping(vision_pg, language_pg, world_size)

    return {"vision": vision_pg, "language": language_pg}

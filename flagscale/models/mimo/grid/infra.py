# Copyright (c) 2026, BAAI. All rights reserved.

"""Unified MIMO grid / process-group infrastructure.

Colocated (and, in the future, non-colocated) multi-module training needs one
coherent answer to "which ranks run which module, and which process groups
does each module use".  This module builds that answer on top of
Megatron-LM-FL's ``HyperCommGrid``, mirroring the grid/optimizer builder
convention from ``megatron.core.models.mimo``: the grid owns process-group
creation via ``create_pg``, and a ``ProcessGroupCollection`` is materialized
from a pre-created grid.

Design invariants
-----------------
* Every rank builds from the *same* ordered module list (the insertion order
  of ``module_configs``), so the build is deterministic.
* Every world rank participates in every collective call, for every module,
  including modules whose rank range it does not belong to.  Membership only
  decides what a rank *stores*, never what it calls.  This keeps
  ``dist.new_subgroups_by_enumeration`` / ``dist.new_group`` hang-free for
  grids with a non-zero ``rank_offset`` (disjoint rank ranges).
* Ranks outside a module's grid range get ``None`` as that module's
  ``ProcessGroupCollection``.
* Grids may overlap (colocated) or be disjoint (non-colocated modules).

Grid layout
-----------
A module grid has dim_names ``["tp", "cp", "dp", "ep", "pp"]``; grid rank
``r`` maps to indices
``r = rank_offset + tp + cp*TP + dp*TP*CP + ep*TP*CP*DP + pp*TP*CP*DP*EP``.
With CP = EP = 1 this reproduces the colocated layout
``rank = pp*TP*DP + dp*TP + tp`` of ``hetero_pg_utils``, so TP/DP/PP group
memberships are identical.  EP is modeled as an independent grid dimension
(Megatron-LM-FL convention) while the colocated builder subdivides DP as
``dp = ep * edp``, so ``expt_dp`` memberships differ when EP > 1 (with EP = 1
both coincide); reconciliation is left to a follow-up.
"""

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from megatron.core import tensor_parallel
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection

from .parallelism import ModuleParallelismConfig

# Dimension order of a module grid: the first dim varies fastest, the last
# (pp) slowest, matching the colocated rank layout.
MODULE_GRID_DIM_NAMES: tuple[str, ...] = ("tp", "cp", "dp", "ep", "pp")

#: Sentinel returned by ``torch.distributed.new_group`` on non-member ranks
#: (``GroupMember.NON_GROUP_MEMBER``, an int); this module's nullable contract
#: maps it to ``None`` wherever a non-member must not hold a group.
_NON_GROUP_MEMBER = dist.GroupMember.NON_GROUP_MEMBER


def _is_member_process_group(pg) -> bool:
    """True when ``pg`` is an actual process group of the current rank.

    ``dist.new_group`` returns ``GroupMember.NON_GROUP_MEMBER`` (an int
    sentinel) on non-member ranks instead of ``None``.  Every field of a
    ``ProcessGroupCollection`` - including the directly-created gloo and
    endpoint groups - must be a real process group or ``None``, never the
    sentinel: MCore's ``get_pg_size`` would call ``group.size()`` on it
    (AttributeError: 'int' object has no attribute 'size') during gradient
    finalization.
    """
    return pg is not None and pg is not _NON_GROUP_MEMBER


# (ProcessGroupCollection field, grid dims) pairs.  The list order IS the
# deterministic creation order every world rank must follow, for every module.
# It covers the group set required by ``_get_pg_collection_for_optimizer``
# (dp, dp-cp, tp, pp, tp-pp, tp-ep-pp, dp-ep, all dims) plus the groups
# consumed by ``flagscale.models.mimo.colocated.parallel_state_ctx``.
_GRID_PG_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dp", ("dp",)),
    ("dp_cp", ("dp", "cp")),
    ("tp", ("tp",)),
    ("pp", ("pp",)),
    ("mp", ("tp", "pp")),
    ("tp_dp_cp", ("tp", "dp")),
    ("cp", ("cp",)),
    ("tp_cp", ("tp", "cp")),
    ("ep", ("ep",)),
    ("tp_ep", ("tp", "ep")),
    ("tp_ep_pp", ("tp", "ep", "pp")),
    ("expt_dp", ("dp", "ep")),
    ("intra_dist_opt", ("tp", "cp", "ep", "pp", "dp")),
)


@dataclass
class ModuleGridConfig:
    """Per-module input for grid construction.

    ``rank_offset`` is the global rank where this module's grid starts.  When
    left ``None``, it is read from ``parallelism.rank_offset`` if present,
    else ``0``.
    """

    module_name: str
    parallelism: ModuleParallelismConfig
    rank_offset: int | None = None

    def resolved_rank_offset(self) -> int:
        """Return the effective rank offset for this module's grid."""
        if self.rank_offset is not None:
            return self.rank_offset
        return int(getattr(self.parallelism, "rank_offset", 0) or 0)


@dataclass
class MIMOInfra:
    """The unified MIMO rank/membership/PG infrastructure.

    ``module_to_grid_map`` maps module names to their ``HyperCommGrid`` in
    global module order.  ``module_to_pg_collection`` maps the same module
    names to the process-group collection created from the grid, or ``None``
    for modules whose grid range does not contain the current rank.
    """

    module_to_grid_map: dict[str, HyperCommGrid]
    module_to_pg_collection: dict[str, ProcessGroupCollection | None]

    def current_module_names(self) -> list[str]:
        """Modules whose grid contains the current rank, in global module order."""
        return [
            name for name, grid in self.module_to_grid_map.items() if grid.is_current_rank_in_grid()
        ]

    def destroy(self) -> None:
        """Destroy every process group created by this infra's grids.

        Covers grid-owned groups (``HyperCommGrid.destroy``) and the groups
        this infra creates directly via ``dist.new_group`` (gloo DP /
        expert-DP and tied-embedding endpoint groups), which the grid does
        not track and which would otherwise leak.
        """
        for grid in self.module_to_grid_map.values():
            grid.destroy()
            direct_pgs = getattr(grid, "_mimo_direct_pgs", None)
            if direct_pgs:
                for pg in direct_pgs:
                    dist.destroy_process_group(pg)
                direct_pgs.clear()


def _normalize_module_configs(
    module_configs: Mapping[str, ModuleParallelismConfig | ModuleGridConfig]
    | Iterable[ModuleGridConfig],
) -> list[ModuleGridConfig]:
    """Normalize user input to an ordered list of ``ModuleGridConfig``.

    Accepts a ``Mapping`` (module name -> parallelism config or
    ``ModuleGridConfig``) or an iterable of ``ModuleGridConfig``.  ``Mapping``
    (not just ``dict``) is required: read-only mappings such as
    ``MIMOParallelismConfig.module_parallelisms`` (a ``MappingProxyType``)
    must work.  The canonical module order is the deterministic input order -
    the global module order every world rank must agree on.

    Raises on empty input or duplicate module names; when a mapping value is
    a ``ModuleGridConfig``, its ``module_name`` must match the mapping key.
    """
    if isinstance(module_configs, Mapping):
        items: list[tuple[str, ModuleParallelismConfig | ModuleGridConfig]] = list(
            module_configs.items()
        )
    else:
        items = [(cfg.module_name, cfg) for cfg in module_configs]

    if not items:
        raise ValueError("module_configs must contain at least one module")

    configs: list[ModuleGridConfig] = []
    seen: set = set()
    for name, cfg in items:
        if name in seen:
            raise ValueError(f"duplicate module name in module_configs: {name!r}")
        seen.add(name)
        if isinstance(cfg, ModuleGridConfig):
            if cfg.module_name != name:
                raise ValueError(
                    f"ModuleGridConfig.module_name {cfg.module_name!r} does not match "
                    f"dict key {name!r}"
                )
            configs.append(cfg)
        else:
            configs.append(ModuleGridConfig(module_name=name, parallelism=cfg))

    for cfg in configs:
        _validate_parallelism(cfg.module_name, cfg.parallelism)
    return configs


def _validate_parallelism(module_name: str, cfg: ModuleParallelismConfig) -> None:
    """Duck-type validation of the parallelism config (works for future sibling module types too)."""
    for attr in (
        "tensor_model_parallel_size",
        "context_parallel_size",
        "data_parallel_size",
        "expert_model_parallel_size",
        "pipeline_model_parallel_size",
    ):
        assert hasattr(cfg, attr), (
            f"{module_name}: parallelism config is missing attribute {attr!r}; expected a "
            f"ModuleParallelismConfig-like object"
        )
    dims = {
        "tp": cfg.tensor_model_parallel_size,
        "cp": cfg.context_parallel_size,
        "dp": cfg.data_parallel_size,
        "ep": cfg.expert_model_parallel_size,
        "pp": cfg.pipeline_model_parallel_size,
    }
    for dim, size in dims.items():
        assert isinstance(size, int) and size >= 1, (
            f"{module_name}: {dim} must be a positive int, got {size!r}"
        )


def build_module_grids(
    module_configs: Mapping[str, ModuleParallelismConfig | ModuleGridConfig]
    | Iterable[ModuleGridConfig],
) -> dict[str, HyperCommGrid]:
    """Construct one ``HyperCommGrid`` per module, in global module order.

    Pure object construction: no collectives are issued, so this is safe to
    call on every rank before ``dist`` group creation.  Grids with disjoint
    rank ranges are expressed via ``rank_offset``.

    Returns:
        Ordered dict mapping module names to ``HyperCommGrid``.
    """
    configs = _normalize_module_configs(module_configs)
    grids: dict[str, HyperCommGrid] = {}
    for cfg in configs:
        p = cfg.parallelism
        shape = [
            p.tensor_model_parallel_size,
            p.context_parallel_size,
            p.data_parallel_size,
            p.expert_model_parallel_size,
            p.pipeline_model_parallel_size,
        ]
        grids[cfg.module_name] = HyperCommGrid(
            shape, list(MODULE_GRID_DIM_NAMES), rank_offset=cfg.resolved_rank_offset()
        )
    return grids


def _create_pp_endpoint_groups(
    grid: HyperCommGrid,
    direct_pgs: list[dist.ProcessGroup] | None = None,
) -> list[tuple[dist.ProcessGroup | None, dist.ProcessGroup | None]]:
    """Create the tied-embedding endpoint groups (first/last PP ranks).

    Collective: every world rank must call this for every module grid, in the
    same order.  Returns one ``(embd, pos_embd)`` pair per PP stage, where
    each entry is ``None`` for stages the current rank is not a member of:

    - ``embd``: the first/last-stage endpoint pair (tied embedding/output
      weights live on both).  With PP == 1 both endpoints are the same rank,
      so it is the per-rank singleton group instead.
    - ``pos_embd``: the first-stage singleton (position embeddings only exist
      on the first stage); with PP == 1 this is the same singleton as
      ``embd``.

    ``direct_pgs`` (optional): list to record the created groups into, so
    ``MIMOInfra.destroy`` can release them - they are created directly via
    ``dist.new_group`` and are NOT tracked by the grid.  Only real groups are
    recorded: non-member ranks receive the ``NON_GROUP_MEMBER`` int sentinel,
    which the nullable contract maps to ``None`` (see
    :func:`_is_member_process_group`).  Every world rank issues the same
    ``new_group`` calls for every stage, including ranks outside the grid and
    middle PP stages; membership only decides what is stored.
    """
    pp_size = grid.shape[grid.dim_names.index("pp")]
    rank = dist.get_rank()
    pairs: list[tuple[dist.ProcessGroup | None, dist.ProcessGroup | None]] = []
    for stage_ranks in grid.get_rank_enum("pp"):
        first, last = stage_ranks[0], stage_ranks[-1]
        # Position embeddings exist on the first stage only (singleton group).
        pos_embd_group = dist.new_group([first])
        if direct_pgs is not None and _is_member_process_group(pos_embd_group):
            direct_pgs.append(pos_embd_group)
        # Tied embedding/output weights span the first and last stages; with
        # PP == 1 the singleton doubles as the embedding group.
        embd_group = pos_embd_group if pp_size <= 1 else dist.new_group([first, last])
        if (
            embd_group is not pos_embd_group
            and _is_member_process_group(embd_group)
            and direct_pgs is not None
        ):
            direct_pgs.append(embd_group)
        # Never store the NON_GROUP_MEMBER sentinel: non-members must hold
        # None (the per-rank singleton columns of PP == 1 grids give member
        # ranks real groups here).
        embd = (
            embd_group
            if (pp_size <= 1 or rank in (first, last)) and _is_member_process_group(embd_group)
            else None
        )
        pos_embd = (
            pos_embd_group if rank == first and _is_member_process_group(pos_embd_group) else None
        )
        pairs.append((embd, pos_embd))
    return pairs


def create_module_pg_collection(grid: HyperCommGrid) -> ProcessGroupCollection | None:
    """Create the process groups of one module grid and return the collection.

    Collective: every world rank must call this, in the same global module
    order, including ranks outside the grid's range.  Ranks outside the grid
    range return ``None``; member ranks return a fully populated
    ``ProcessGroupCollection``.  The created group set is ``_GRID_PG_SPECS``
    plus gloo DP / expert-DP groups and the tied-embedding endpoint groups,
    mirroring ``hetero_pg_utils._create_module_pg_collection``.
    """
    for _, dims in _GRID_PG_SPECS:
        grid.create_pg(list(dims))

    # Gloo DP groups (used by gradient reduction / weight-tie sync paths).
    # Every world rank enters these collectives; each rank only receives the
    # group that contains it (None for the rest).
    dp_ranks_enum = grid.get_rank_enum("dp")
    dp_gloo_groups = [dist.new_group(ranks, backend="gloo") for ranks in dp_ranks_enum]
    ep_size = grid.shape[grid.dim_names.index("ep")]
    expt_dp_gloo_groups: list[dist.ProcessGroup | None] = []
    if ep_size > 1:
        expt_dp_gloo_groups = [
            dist.new_group(ranks, backend="gloo") for ranks in grid.get_rank_enum(["dp", "ep"])
        ]

    # Tied-embedding endpoint groups and first-stage position-embedding
    # groups.  Must run on every world rank before the membership early
    # return below.  They are created directly via dist.new_group and are
    # invisible to HyperCommGrid; record real groups (sentinels filtered out)
    # on the grid so ``MIMOInfra.destroy`` releases them too.
    direct_pgs: list[dist.ProcessGroup] = []
    for group in dp_gloo_groups + expt_dp_gloo_groups:
        if _is_member_process_group(group):
            direct_pgs.append(group)
    pp_endpoint_groups = _create_pp_endpoint_groups(grid, direct_pgs)

    if not grid.is_current_rank_in_grid():
        return None

    pg = ProcessGroupCollection()
    for field, dims in _GRID_PG_SPECS:
        setattr(pg, field, grid.get_pg(list(dims)))

    # Aliases matching the colocated builder (CP=1): the distributed
    # optimizer maps model-parallel to intra and data-parallel to inter;
    # expert TP spans TP x EP (grid convention, cf. mcore expert groups).
    pg.intra_dp_cp = pg.dp_cp
    pg.inter_dist_opt = pg.dp
    pg.expt_tp = pg.tp_ep
    pg.intra_expt_dp = pg.expt_dp

    # Gloo groups: for member ranks exactly one entry per enumeration is a
    # real group; the other entries are the NON_GROUP_MEMBER int sentinel
    # (not None) and are filtered out - a real group or None, never an int.
    pg.dp_gloo = next((g for g in dp_gloo_groups if _is_member_process_group(g)), None)
    pg.expt_dp_gloo = next((g for g in expt_dp_gloo_groups if _is_member_process_group(g)), None)
    if pg.expt_dp_gloo is None:
        pg.expt_dp_gloo = pg.expt_dp

    # Embedding groups: tied weights live on first/last PP ranks; position
    # embeddings on the first stage only.  With PP == 1 both are the
    # per-rank singleton group: the input- and output-side contributions
    # already accumulate locally, and a TP x DP x CP-wide group would
    # elementwise all-reduce *different vocab shards* on every TP rank (and
    # re-sum already-averaged DP replicas) - silent embedding-gradient
    # corruption.  Ranks that are not endpoints of their PP column get None;
    # mcore's embedding-grad all-reduce skips None/singleton groups
    # (``get_pg_size`` returns 1 for both).
    pg.embd = next(
        (pair[0] for pair in pp_endpoint_groups if _is_member_process_group(pair[0])),
        None,
    )
    pg.pos_embd = next(
        (pair[1] for pair in pp_endpoint_groups if _is_member_process_group(pair[1])),
        None,
    )

    # Context-parallel aliases (CP=1).
    pg.hcp = [pg.cp]

    # Deduplicate the directly-created groups by identity (PP==1 aliases
    # pos_embd/embd) and record them on the grid - not the collection - so
    # the record survives on ranks that return ``None`` above.
    seen: set = set()
    unique_direct: list[dist.ProcessGroup] = []
    for group in direct_pgs:
        if id(group) not in seen:
            seen.add(id(group))
            unique_direct.append(group)
    if unique_direct:
        grid._mimo_direct_pgs = unique_direct

    return pg


def build_module_pg_collections(
    module_to_grid_map: dict[str, HyperCommGrid],
) -> dict[str, ProcessGroupCollection | None]:
    """Create all modules' process groups in global module order.

    Collective: every world rank must call with the same ordered grid map.
    Returns an ordered dict mapping module names to collections (``None``
    for modules whose grid range does not contain the current rank).
    """
    collections: dict[str, ProcessGroupCollection | None] = {}
    for module_name, grid in module_to_grid_map.items():
        collections[module_name] = create_module_pg_collection(grid)
    return collections


def build_mimo_infra(
    module_configs: Mapping[str, ModuleParallelismConfig | ModuleGridConfig]
    | Iterable[ModuleGridConfig],
) -> MIMOInfra:
    """One-shot builder: grids + process-group collections + membership.

    Collective: every world rank must call with the same ordered configs.
    Returns the ``MIMOInfra`` for the current rank (non-member modules carry
    ``None`` collections).
    """
    grids = build_module_grids(module_configs)
    collections = build_module_pg_collections(grids)
    return MIMOInfra(module_to_grid_map=grids, module_to_pg_collection=collections)


def grids_are_colocated(module_to_grid_map: dict[str, HyperCommGrid]) -> bool:
    """True if all module grids span exactly the same rank range.

    Mirrors ``RankRole._all_grids_colocated``: equal ``rank_offset`` and
    equal size.  An empty map is vacuously colocated.
    """
    grids = list(module_to_grid_map.values())
    if not grids:
        return True
    first = grids[0]
    return all(g.rank_offset == first.rank_offset and g.size == first.size for g in grids[1:])


def set_per_module_random_seed(args, infra: MIMOInfra) -> None:
    """Re-seed Python/NumPy/torch/MCore RNG by the rank's module TP/PP ranks.

    In grid mode the global parallel state is initialized with TP=1/PP=1, so
    the standard seed path gives every rank the same seed; TP-sharded module
    weights would then be initialized differently across a module's TP group.
    Mirror the Megatron-Bridge ``_set_per_module_random_seeds``: seed by the
    module's own PP rank (different stages get different seeds) and fork the
    MCore CUDA RNG tracker with the module's TP/EP/ETP ranks.
    """
    seed = args.seed
    tp_rank = ep_rank = etp_rank = 0
    pp_rank = 0
    for module_name, grid in infra.module_to_grid_map.items():
        if not grid.is_current_rank_in_grid():
            continue
        pg_collection = infra.module_to_pg_collection.get(module_name)
        if pg_collection is None:
            continue
        current_rank = torch.distributed.get_rank()
        tp_rank = torch.distributed.get_group_rank(pg_collection.tp, current_rank)
        pp_rank = torch.distributed.get_group_rank(pg_collection.pp, current_rank)
        if getattr(pg_collection, "ep", None) is not None:
            ep_rank = torch.distributed.get_group_rank(pg_collection.ep, current_rank)
        if getattr(pg_collection, "expt_tp", None) is not None:
            etp_rank = torch.distributed.get_group_rank(pg_collection.expt_tp, current_rank)
        break

    pp_seed = seed + (100 * pp_rank)
    random.seed(pp_seed)
    np.random.seed(pp_seed)
    torch.manual_seed(pp_seed)
    if torch.cuda.device_count() > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(
            pp_seed, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=etp_rank
        )

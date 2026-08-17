# Copyright (c) 2026, BAAI. All rights reserved.

"""Unit tests for flagscale.models.mimo.bridge.infra.

Uses the REAL ``HyperCommGrid`` (deterministic rank enumeration) with the
``torch.distributed`` surface mocked out: no process group initialization is
required.  ``WORLD_SIZE`` is injected via the environment (the mock path
HyperCommGrid itself supports), and ``new_group`` /
``new_subgroups_by_enumeration`` / ``get_rank`` are replaced with a recorder
that returns a fake group for member ranks and ``None`` otherwise.
"""

import os
import sys
import unittest
from unittest import mock

import torch.distributed as dist

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# The training backend repo (sibling of FlagScale) must be importable; the
# conda env ships an editable megatron_core that points at a stale checkout,
# so the current repo is prepended explicitly.
_MEGATRON_REPO = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, "Megatron-LM-FL"))
if os.path.isdir(_MEGATRON_REPO) and _MEGATRON_REPO not in sys.path:
    sys.path.insert(0, _MEGATRON_REPO)

# The real package import chain is used (``mimo_optimizer`` imports only from
# ``megatron.core``, so the package-level ``__init__`` imports cleanly).
# Importing the real package instead of stubbing ``flagscale.models.mimo`` in
# sys.modules matters for combined collection: a leftover stub would shadow
# the real package for other test modules imported later in the same process
# (the qwen35 grid model tests import ``flagscale.models.mimo`` transitively).
from flagscale.models.mimo.bridge.infra import (
    _GRID_PG_SPECS,
    MODULE_GRID_DIM_NAMES,
    MIMOInfra,
    ModuleGridConfig,
    build_mimo_infra,
    build_module_grids,
    build_module_pg_collections,
    create_module_pg_collection,
    grids_are_colocated,
)
from flagscale.models.mimo.bridge.parallelism import ModuleParallelismConfig
from flagscale.models.mimo.colocated.mimo_config import (
    ModuleParallelismConfig as ColocatedModuleParallelismConfig,
)

WORLD_SIZE = 8


class FakeProcessGroup:
    """Stand-in for a torch.distributed.ProcessGroup carrying its member ranks."""

    def __init__(self, ranks):
        self.ranks = list(ranks)

    def __repr__(self):
        return f"FakeProcessGroup({self.ranks})"

    def __eq__(self, other):
        return isinstance(other, FakeProcessGroup) and self.ranks == other.ranks

    def __hash__(self):
        return hash(tuple(self.ranks))


class DistRecorder:
    """Deterministic mock of the torch.distributed surface used by the build.

    ``rank`` selects the simulated current rank.  All collective calls are
    recorded so tests can assert determinism (identical call sequences across
    ranks) and membership (None for non-member ranks).
    """

    def __init__(self):
        self.rank = 0
        self.new_group_calls = []  # (ranks, kwargs)
        self.subgroup_calls = []  # (rank_enum, kwargs)
        self.destroyed = []

    def get_rank(self):
        return self.rank

    def new_group(self, ranks, **kwargs):
        self.new_group_calls.append((list(ranks), kwargs))
        if self.rank in ranks:
            return FakeProcessGroup(ranks)
        # REAL torch semantics: non-member ranks receive the
        # GroupMember.NON_GROUP_MEMBER int sentinel (-100), NOT None.
        return dist.GroupMember.NON_GROUP_MEMBER

    def new_subgroups_by_enumeration(self, rank_enum, **kwargs):
        self.subgroup_calls.append(([list(r) for r in rank_enum], kwargs))
        for group_ranks in rank_enum:
            if self.rank in group_ranks:
                return FakeProcessGroup(group_ranks), None
        return None, None

    def destroy_process_group(self, pg):
        self.destroyed.append(pg)


class MimoGridUtilsTestBase(unittest.TestCase):
    """Shared harness: mock dist + WORLD_SIZE, reset recorder per test."""

    def setUp(self):
        self.recorder = DistRecorder()
        patchers = [
            mock.patch("torch.distributed.get_rank", side_effect=self.recorder.get_rank),
            mock.patch("torch.distributed.new_group", side_effect=self.recorder.new_group),
            mock.patch(
                "torch.distributed.new_subgroups_by_enumeration",
                side_effect=self.recorder.new_subgroups_by_enumeration,
            ),
            mock.patch(
                "torch.distributed.destroy_process_group",
                side_effect=self.recorder.destroy_process_group,
            ),
            mock.patch.dict(os.environ, {"WORLD_SIZE": str(WORLD_SIZE)}),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def as_rank(self, rank):
        self.recorder.rank = rank


class TestBuildModuleGrids(MimoGridUtilsTestBase):
    """Grid construction: shapes, offsets, and layout vs the colocated one."""

    def _colocated_cfgs(self):
        return {
            "vision": ModuleParallelismConfig(
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=2,
                data_parallel_size=2,
            ),
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=2,
                data_parallel_size=2,
            ),
        }

    def test_shapes_and_global_module_order(self):
        grids = build_module_grids(self._colocated_cfgs())
        self.assertEqual(list(grids.keys()), ["vision", "language"])
        for grid in grids.values():
            self.assertEqual(grid.shape, [2, 1, 2, 1, 2])
            self.assertEqual(grid.dim_names, list(MODULE_GRID_DIM_NAMES))
            self.assertEqual(grid.rank_offset, 0)
            self.assertEqual(grid.size, 8)

    def test_layout_matches_colocated_rank_groups(self):
        # Colocated hetero_pg_utils._compute_rank_groups with tp2 dp2 pp2:
        # tp groups fixed (dp, pp), dp groups fixed (tp, pp),
        # pp groups fixed (tp, dp).
        grids = build_module_grids(self._colocated_cfgs())
        grid = grids["vision"]
        self.assertEqual(grid.get_rank_enum("tp"), [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(grid.get_rank_enum("dp"), [[0, 2], [1, 3], [4, 6], [5, 7]])
        self.assertEqual(grid.get_rank_enum("pp"), [[0, 4], [1, 5], [2, 6], [3, 7]])
        self.assertEqual(grid.get_rank_enum(["tp", "pp"]), [[0, 1, 4, 5], [2, 3, 6, 7]])
        self.assertEqual(grid.get_rank_enum(["tp", "dp"]), [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(
            grid.get_rank_enum(["tp", "cp", "ep", "pp", "dp"]), [[0, 1, 2, 3, 4, 5, 6, 7]]
        )

    def test_rank_offset_shifts_enumerations(self):
        grid = build_module_grids(
            {
                "language": ModuleGridConfig(
                    "language",
                    ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
                    rank_offset=4,
                )
            }
        )["language"]
        self.assertEqual(grid.size, 4)
        self.assertEqual(grid.rank_offset, 4)
        self.assertEqual(grid.get_rank_enum("tp"), [[4, 5], [6, 7]])
        self.assertEqual(grid.get_rank_enum("dp"), [[4, 6], [5, 7]])
        self.as_rank(3)
        self.assertFalse(grid.is_current_rank_in_grid())
        self.as_rank(5)
        self.assertTrue(grid.is_current_rank_in_grid())

    def test_iterable_and_dict_forms_agree(self):
        cfgs = [
            ModuleGridConfig(
                "vision",
                ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=4),
            ),
            ModuleGridConfig(
                "language",
                ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=4),
            ),
        ]
        from_iterable = build_module_grids(cfgs)
        from_dict = build_module_grids({c.module_name: c.parallelism for c in cfgs})
        self.assertEqual(list(from_iterable.keys()), list(from_dict.keys()))
        for name in from_iterable:
            self.assertEqual(from_iterable[name].shape, from_dict[name].shape)
            self.assertEqual(from_iterable[name].rank_offset, from_dict[name].rank_offset)

    def test_rank_offset_resolution_precedence(self):
        # ModuleGridConfig.rank_offset wins over parallelism.rank_offset.
        cfg_with_attr = mock.Mock()
        cfg_with_attr.tensor_model_parallel_size = 2
        cfg_with_attr.context_parallel_size = 1
        cfg_with_attr.data_parallel_size = 4
        cfg_with_attr.expert_model_parallel_size = 1
        cfg_with_attr.pipeline_model_parallel_size = 1
        cfg_with_attr.rank_offset = 2
        self.assertEqual(ModuleGridConfig("m", cfg_with_attr).resolved_rank_offset(), 2)
        self.assertEqual(
            ModuleGridConfig("m", cfg_with_attr, rank_offset=6).resolved_rank_offset(), 6
        )
        # No rank_offset anywhere -> 0.
        plain = ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=4)
        self.assertEqual(ModuleGridConfig("m", plain).resolved_rank_offset(), 0)


class TestMappingModuleConfigs(MimoGridUtilsTestBase):
    """Mapping input form (incl. read-only MappingProxyType) for the builders.

    Regression: ``model_provider`` calls ``build_mimo_infra(mimo_config.module_parallelisms)``
    where ``module_parallelisms`` is a ``MappingProxyType``; the old
    ``isinstance(..., dict)`` branch treated it as an iterable of configs and
    iterated the module-name *strings* ('str' object has no attribute
    'module_name').
    """

    def _mapping_proxy_configs(self, language_first=False):
        from types import MappingProxyType

        entries = [
            (
                "images",
                ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
            ),
            (
                "language",
                ModuleParallelismConfig(
                    tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=2
                ),
            ),
        ]
        if language_first:
            entries.reverse()
        return MappingProxyType(dict(entries))

    def test_mapping_proxy_accepted_by_build_mimo_infra(self):
        self.as_rank(0)
        infra = build_mimo_infra(self._mapping_proxy_configs())
        self.assertEqual(list(infra.module_to_grid_map.keys()), ["images", "language"])
        self.assertEqual(infra.module_to_grid_map["images"].shape, [2, 1, 1, 1, 1])
        self.assertEqual(infra.module_to_grid_map["language"].shape, [1, 1, 6, 1, 1])
        self.assertEqual(infra.module_to_grid_map["language"].rank_offset, 2)
        # Rank 0 is an images member only (language spans [2, 8)).
        self.assertIsNotNone(infra.module_to_pg_collection["images"])
        self.assertIsNone(infra.module_to_pg_collection["language"])
        self.assertEqual(infra.current_module_names(), ["images"])

    def test_mapping_proxy_accepted_by_build_module_grids(self):
        grids = build_module_grids(self._mapping_proxy_configs())
        self.assertEqual(list(grids.keys()), ["images", "language"])

    def test_mapping_canonical_order_is_deterministic_input_order(self):
        self.as_rank(0)
        infra = build_mimo_infra(self._mapping_proxy_configs(language_first=True))
        self.assertEqual(list(infra.module_to_grid_map.keys()), ["language", "images"])
        # Same content in a different (deterministic) order yields the same grids.
        other = build_mimo_infra(self._mapping_proxy_configs())
        for name in ("images", "language"):
            self.assertEqual(
                infra.module_to_grid_map[name].shape,
                other.module_to_grid_map[name].shape,
            )
            self.assertEqual(
                infra.module_to_grid_map[name].rank_offset,
                other.module_to_grid_map[name].rank_offset,
            )

    def test_mapping_key_mismatch_with_modulegridconfig_value_rejected(self):
        from types import MappingProxyType

        with self.assertRaises(ValueError):
            build_module_grids(
                MappingProxyType(
                    {
                        "vision": ModuleGridConfig(
                            "language", ModuleParallelismConfig(data_parallel_size=8)
                        )
                    }
                )
            )

    def test_mapping_and_list_forms_agree(self):
        cfgs = [
            ModuleGridConfig(
                "images",
                ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
            ),
            ModuleGridConfig(
                "language",
                ModuleParallelismConfig(
                    tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=2
                ),
            ),
        ]
        from_list = build_module_grids(cfgs)
        from_mapping = build_module_grids({cfg.module_name: cfg.parallelism for cfg in cfgs})
        self.assertEqual(list(from_list.keys()), list(from_mapping.keys()))
        for name in from_list:
            self.assertEqual(from_list[name].shape, from_mapping[name].shape)
            self.assertEqual(from_list[name].rank_offset, from_mapping[name].rank_offset)


class TestValidation(MimoGridUtilsTestBase):
    def test_empty_configs_rejected(self):
        with self.assertRaises(ValueError):
            build_module_grids({})

    def test_duplicate_module_names_rejected(self):
        with self.assertRaises(ValueError):
            build_module_grids(
                [
                    ModuleGridConfig("vision", ModuleParallelismConfig()),
                    ModuleGridConfig("vision", ModuleParallelismConfig()),
                ]
            )

    def test_dict_key_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            build_module_grids(
                {
                    "vision": ModuleGridConfig(
                        "language", ModuleParallelismConfig(data_parallel_size=8)
                    )
                }
            )

    def test_missing_parallelism_attr_rejected(self):
        with self.assertRaises(AssertionError):
            build_module_grids({"vision": object()})

    def test_non_positive_dim_rejected(self):
        # Sibling config type validates at construction.
        with self.assertRaises(ValueError):
            build_module_grids({"vision": ModuleParallelismConfig(tensor_model_parallel_size=0)})
        # Defensive path: a lax (colocated-style) config type is caught by the
        # builder's own validation.
        with self.assertRaises(AssertionError):
            build_module_grids(
                {"vision": ColocatedModuleParallelismConfig(tensor_model_parallel_size=0)}
            )

    def test_oversized_grid_rejected(self):
        # tp8 dp2 -> 16 ranks > WORLD_SIZE 8.
        with self.assertRaises(RuntimeError):
            build_module_grids(
                {
                    "vision": ModuleParallelismConfig(
                        tensor_model_parallel_size=8, data_parallel_size=2
                    )
                }
            )


class TestCreateModulePgCollection(MimoGridUtilsTestBase):
    """PG creation for one module: memberships, aliases, nullable non-members."""

    def _grid(self, tp=2, dp=2, pp=2, offset=0):
        return build_module_grids(
            {
                "m": ModuleGridConfig(
                    "m",
                    ModuleParallelismConfig(
                        tensor_model_parallel_size=tp,
                        pipeline_model_parallel_size=pp,
                        data_parallel_size=dp,
                    ),
                    rank_offset=offset,
                )
            }
        )["m"]

    def test_member_rank_gets_full_collection(self):
        self.as_rank(2)
        pg = create_module_pg_collection(self._grid())
        self.assertIsNotNone(pg)
        # Rank 2 layout (tp2 dp2 pp2, offset 0): rank = pp*4 + dp*2 + tp
        # -> pp0 dp1 tp0.
        self.assertEqual(pg.tp.ranks, [2, 3])
        self.assertEqual(pg.dp.ranks, [0, 2])
        self.assertEqual(pg.pp.ranks, [2, 6])
        self.assertEqual(pg.mp.ranks, [2, 3, 6, 7])
        self.assertEqual(pg.tp_dp_cp.ranks, [0, 1, 2, 3])
        self.assertEqual(pg.tp_cp.ranks, [2, 3])
        self.assertEqual(pg.tp_ep.ranks, [2, 3])
        self.assertEqual(pg.tp_ep_pp.ranks, [2, 3, 6, 7])
        self.assertEqual(pg.expt_dp.ranks, [0, 2])
        self.assertEqual(pg.intra_dist_opt.ranks, list(range(8)))
        # CP/EP singletons (CP=EP=1).
        self.assertEqual(pg.cp.ranks, [2])
        self.assertEqual(pg.ep.ranks, [2])
        # Aliases (CP=1).
        self.assertIs(pg.intra_dp_cp, pg.dp_cp)
        self.assertIs(pg.inter_dist_opt, pg.dp)
        self.assertIs(pg.expt_tp, pg.tp_ep)
        self.assertIs(pg.intra_expt_dp, pg.expt_dp)
        self.assertEqual(pg.dp_cp.ranks, [0, 2])
        # Gloo DP group for the current rank.
        self.assertEqual(pg.dp_gloo.ranks, [0, 2])
        # EP=1 -> expert gloo falls back to the nccl expert-DP group.
        self.assertIs(pg.expt_dp_gloo, pg.expt_dp)
        # PP=2 -> embd is the first/last-stage endpoint pair; pos_embd is the
        # first-stage singleton (rank 2 is pp0 of the [2, 6] column).
        self.assertEqual(pg.embd.ranks, [2, 6])
        self.assertEqual(pg.pos_embd.ranks, [2])
        self.assertIsNot(pg.pos_embd, pg.embd)
        self.assertEqual(pg.hcp, [pg.cp])

    def test_pp1_embd_and_pos_embd_are_rank_singletons(self):
        # PP=1: the whole module is one stage, so embd/pos_embd must be the
        # per-rank singleton group (a TP x DP x CP-wide group would all-reduce
        # different vocab shards across TP ranks - gradient corruption).
        self.as_rank(2)
        pg = create_module_pg_collection(self._grid(pp=1))
        self.assertEqual(pg.embd.ranks, [2])
        self.assertIs(pg.pos_embd, pg.embd)
        self.assertIsNot(pg.embd, pg.tp_dp_cp)

    def test_pp_gt_1_non_endpoint_stage_gets_none_embedding_groups(self):
        # tp2 dp2 pp3 -> 12-rank grid; rank = pp*4 + dp*2 + tp.
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "12"}):
            self.as_rank(2)  # pp0 dp1 tp0: first stage of the [2, 6, 10] column
            pg = create_module_pg_collection(self._grid(tp=2, dp=2, pp=3))
            self.assertEqual(pg.embd.ranks, [2, 10])
            self.assertEqual(pg.pos_embd.ranks, [2])

            self.as_rank(6)  # pp1: middle stage, no embedding groups
            pg = create_module_pg_collection(self._grid(tp=2, dp=2, pp=3))
            self.assertIsNone(pg.embd)
            self.assertIsNone(pg.pos_embd)

            self.as_rank(10)  # pp2: last stage, embd only (no pos_embd)
            pg = create_module_pg_collection(self._grid(tp=2, dp=2, pp=3))
            self.assertEqual(pg.embd.ranks, [2, 10])
            self.assertIsNone(pg.pos_embd)

    def test_non_member_rank_returns_none(self):
        # Grid over ranks [4, 8): tp2 dp2 pp1.
        self.as_rank(3)  # outside the grid range
        self.assertIsNone(create_module_pg_collection(self._grid(pp=1, offset=4)))
        self.as_rank(4)  # first grid rank
        self.assertIsNotNone(create_module_pg_collection(self._grid(pp=1, offset=4)))
        self.as_rank(7)  # last grid rank
        self.assertIsNotNone(create_module_pg_collection(self._grid(pp=1, offset=4)))

    def test_collective_sequence_identical_across_ranks(self):
        """Non-member ranks must issue the exact same collective calls.

        Uses a fresh grid per simulated rank: HyperCommGrid refuses to
        re-create a group key, which is exactly why real deployments build
        once per world rank with the same deterministic sequence.
        """
        # Grid over ranks [0, 4): rank 1 is a member, rank 7 is not.
        self.as_rank(1)
        create_module_pg_collection(self._grid(pp=1))
        member_calls = (
            [tuple(dims) for dims, _ in self.recorder.subgroup_calls],
            [tuple(ranks) for ranks, _ in self.recorder.new_group_calls],
        )
        self.recorder.subgroup_calls.clear()
        self.recorder.new_group_calls.clear()
        self.as_rank(7)
        create_module_pg_collection(self._grid(pp=1))
        non_member_calls = (
            [tuple(dims) for dims, _ in self.recorder.subgroup_calls],
            [tuple(ranks) for ranks, _ in self.recorder.new_group_calls],
        )
        self.assertEqual(member_calls, non_member_calls)

    def test_create_pg_order_matches_spec_and_optimizer_reference(self):
        from megatron.core.hyper_comm_grid import HyperCommGrid

        recorded_dims = []
        original_create_pg = HyperCommGrid.create_pg

        def spy_create_pg(grid_self, dims, **kwargs):
            recorded_dims.append(list(dims))
            return original_create_pg(grid_self, dims, **kwargs)

        with mock.patch.object(
            HyperCommGrid, "create_pg", autospec=True, side_effect=spy_create_pg
        ):
            self.as_rank(0)
            create_module_pg_collection(self._grid())
        self.assertEqual(recorded_dims, [list(dims) for _, dims in _GRID_PG_SPECS])
        # The dims required by
        # megatron.core.models.mimo.optimizer._get_pg_collection_for_optimizer
        # must all be pre-created:
        required = [
            ("dp",),
            ("dp", "cp"),
            ("tp",),
            ("pp",),
            ("tp", "pp"),
            ("tp", "ep", "pp"),
            ("dp", "ep"),
            ("tp", "cp", "ep", "pp", "dp"),
        ]
        for dims in required:
            self.assertIn(list(dims), recorded_dims)


class TestNoSentinelLeaks(MimoGridUtilsTestBase):
    """No PG field may hold the NON_GROUP_MEMBER int sentinel.

    Regression: ``dist.new_group`` returns ``GroupMember.NON_GROUP_MEMBER``
    (-100, an int) on non-member ranks; the collection assembly used
    ``is not None`` filters and ``next(...)`` selections that picked the
    sentinel up, so ``pg.embd`` / ``pg.pos_embd`` / ``pg.dp_gloo`` /
    ``pg.expt_dp_gloo`` (and the recorded direct groups) could be ints -
    ``get_pg_size`` then failed with AttributeError: 'int' object has no
    attribute 'size' during ``finalize_model_grads``.
    """

    def _grid(self, tp=2, dp=2, pp=2, offset=0):
        return build_module_grids(
            {
                "m": ModuleGridConfig(
                    "m",
                    ModuleParallelismConfig(
                        tensor_model_parallel_size=tp,
                        pipeline_model_parallel_size=pp,
                        data_parallel_size=dp,
                    ),
                    rank_offset=offset,
                )
            }
        )["m"]

    def _assert_no_int_fields(self, pg):
        for name, value in vars(pg).items():
            if isinstance(value, list):
                for item in value:
                    self.assertFalse(
                        isinstance(item, int), f"PG field {name} contains int {item!r}"
                    )
            else:
                self.assertFalse(isinstance(value, int), f"PG field {name} is int {value!r}")

    def test_pp1_member_rank_embd_is_real_singleton_not_sentinel(self):
        # The reported failure: a pp==1 member rank that is not the grid's
        # first rank used to get embd = -100 (the sentinel returned for the
        # earlier singleton columns) -> get_pg_size(embd) crashed in
        # finalize_model_grads.
        self.as_rank(2)
        pg = create_module_pg_collection(self._grid(pp=1))
        self.assertIsInstance(pg.embd, FakeProcessGroup)
        self.assertEqual(pg.embd.ranks, [2])
        self.assertIs(pg.pos_embd, pg.embd)
        self.assertIsInstance(pg.dp_gloo, FakeProcessGroup)
        self.assertIsInstance(pg.expt_dp_gloo, FakeProcessGroup)
        self._assert_no_int_fields(pg)

    def test_no_int_fields_pp1_grid_all_ranks(self):
        # tp2 dp2 pp1 grid spans [0, 4): members get a full collection with
        # no int fields, non-members get None.
        for rank in range(8):
            self.as_rank(rank)
            pg = create_module_pg_collection(self._grid(pp=1))
            if rank < 4:
                self._assert_no_int_fields(pg)
            else:
                self.assertIsNone(pg)

    def test_no_int_fields_pp2_grid_all_ranks(self):
        for rank in range(8):
            self.as_rank(rank)
            pg = create_module_pg_collection(self._grid(pp=2))
            self._assert_no_int_fields(pg)

    def test_no_int_fields_offset_grid_member_and_nonmember(self):
        self.as_rank(9)  # outside the grid range [4, 8): collection is None
        self.assertIsNone(create_module_pg_collection(self._grid(pp=1, offset=4)))
        self.as_rank(5)  # member rank of the offset grid
        pg = create_module_pg_collection(self._grid(pp=1, offset=4))
        self._assert_no_int_fields(pg)

    def test_direct_groups_contain_only_real_groups(self):
        self.as_rank(2)
        grid = self._grid(pp=2)
        create_module_pg_collection(grid)
        direct = getattr(grid, "_mimo_direct_pgs", [])
        self.assertTrue(direct)
        for group in direct:
            self.assertIsInstance(group, FakeProcessGroup)
            self.assertFalse(isinstance(group, int))


class TestBuildModulePgCollections(MimoGridUtilsTestBase):
    """Multi-module build: global module order + nullable per-module collections."""

    def _disjoint_infra_input(self):
        # vision on ranks [0, 4), language on ranks [4, 8): disjoint ranges.
        grids = build_module_grids(
            {
                "vision": ModuleGridConfig(
                    "vision",
                    ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
                    rank_offset=0,
                ),
                "language": ModuleGridConfig(
                    "language",
                    ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
                    rank_offset=4,
                ),
            }
        )
        return grids

    def test_nullable_collections_per_module(self):
        self.as_rank(2)  # vision member, language non-member
        collections = build_module_pg_collections(self._disjoint_infra_input())
        self.assertEqual(list(collections.keys()), ["vision", "language"])
        self.assertIsNotNone(collections["vision"])
        self.assertIsNone(collections["language"])
        self.assertEqual(collections["vision"].dp.ranks, [0, 2])

        self.as_rank(5)  # language member, vision non-member
        collections = build_module_pg_collections(self._disjoint_infra_input())
        self.assertIsNone(collections["vision"])
        self.assertIsNotNone(collections["language"])
        self.assertEqual(collections["language"].tp.ranks, [4, 5])
        self.assertEqual(collections["language"].dp.ranks, [5, 7])

    def test_modules_processed_in_global_module_order_on_every_rank(self):
        from megatron.core.hyper_comm_grid import HyperCommGrid

        recorded_dims = []
        original_create_pg = HyperCommGrid.create_pg

        def spy_create_pg(grid_self, dims, **kwargs):
            recorded_dims.append(list(dims))
            return original_create_pg(grid_self, dims, **kwargs)

        with mock.patch.object(
            HyperCommGrid, "create_pg", autospec=True, side_effect=spy_create_pg
        ):
            self.as_rank(2)
            build_module_pg_collections(self._disjoint_infra_input())
            order_rank2 = list(recorded_dims)
            recorded_dims.clear()
            self.as_rank(7)
            build_module_pg_collections(self._disjoint_infra_input())
            order_rank7 = list(recorded_dims)
        # Same call sequence: vision grid first, then language grid.
        self.assertEqual(order_rank2, order_rank7)
        self.assertEqual(order_rank2, [list(dims) for _, dims in _GRID_PG_SPECS] * 2)


class TestBuildMimoInfra(MimoGridUtilsTestBase):
    def test_colocated_infra_and_current_modules(self):
        self.as_rank(3)
        infra = build_mimo_infra(
            {
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=2, data_parallel_size=4
                ),
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2, data_parallel_size=4
                ),
            }
        )
        self.assertIsInstance(infra, MIMOInfra)
        self.assertEqual(list(infra.module_to_grid_map.keys()), ["vision", "language"])
        self.assertIsNotNone(infra.module_to_pg_collection["vision"])
        self.assertIsNotNone(infra.module_to_pg_collection["language"])
        self.assertEqual(infra.current_module_names(), ["vision", "language"])
        self.assertTrue(grids_are_colocated(infra.module_to_grid_map))

    def test_disjoint_infra_current_modules(self):
        infra = build_mimo_infra(
            {
                "vision": ModuleGridConfig(
                    "vision",
                    ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
                    rank_offset=0,
                ),
                "language": ModuleGridConfig(
                    "language",
                    ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
                    rank_offset=4,
                ),
            }
        )
        self.assertFalse(grids_are_colocated(infra.module_to_grid_map))
        self.as_rank(1)
        self.assertEqual(infra.current_module_names(), ["vision"])
        self.as_rank(6)
        self.assertEqual(infra.current_module_names(), ["language"])

    def test_destroy_tears_down_all_grid_groups(self):
        self.as_rank(0)
        infra = build_mimo_infra(
            {
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=2, data_parallel_size=4
                ),
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2, data_parallel_size=4
                ),
            }
        )
        infra.destroy()
        # Every grid destroyed every created (non-None) group, including the
        # groups created directly via dist.new_group (gloo DP groups and the
        # tied-embedding endpoint groups) that HyperCommGrid does not track.
        n_pgs = sum(
            1
            for dims, _ in self.recorder.subgroup_calls
            for group_ranks in dims
            if 0 in group_ranks
        )
        n_direct = len([ranks for ranks, _ in self.recorder.new_group_calls if 0 in ranks])
        self.assertEqual(len(self.recorder.destroyed), n_pgs + n_direct)

    def test_destroy_releases_directly_created_groups(self):
        # PP=2: the first-stage singleton [first] and the endpoint pair
        # [first, last] groups are created via dist.new_group directly (not
        # tracked by HyperCommGrid); infra.destroy() must release them too.
        self.as_rank(0)
        infra = build_mimo_infra(
            {
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=2,
                )
            }
        )
        grid = infra.module_to_grid_map["language"]
        direct_before = list(getattr(grid, "_mimo_direct_pgs", []))
        self.assertTrue(direct_before)
        expected_ranks = {tuple(sorted(pg.ranks)) for pg in direct_before}
        # Rank 0 (pp0 dp0 tp0 of the tp2 dp2 pp2 grid over [0, 8)):
        # gloo DP group [0, 2], endpoint pair [0, 4], first-stage singleton [0].
        self.assertEqual(expected_ranks, {(0, 2), (0, 4), (0,)})

        infra.destroy()
        destroyed_ids = {id(pg) for pg in self.recorder.destroyed}
        self.assertTrue(all(id(pg) in destroyed_ids for pg in direct_before))
        # The record is cleared; a second destroy is a no-op (no double
        # destroys of the same group).
        self.assertEqual(getattr(grid, "_mimo_direct_pgs", []), [])
        destroyed_after_first = len(self.recorder.destroyed)
        infra.destroy()
        self.assertEqual(len(self.recorder.destroyed), destroyed_after_first)
        self.assertTrue(all(id(pg) in destroyed_ids for pg in direct_before))

    def test_direct_group_recording_does_not_change_creation_order(self):
        # The record is a pure side effect: the dist.new_group call sequence
        # (gloo DP groups, then endpoint groups) is untouched.
        self.as_rank(0)
        grids = build_module_grids(
            {
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=2,
                )
            }
        )
        build_module_pg_collections(grids)
        calls = [tuple(ranks) for ranks, _ in self.recorder.new_group_calls]
        # gloo DP groups first (dp enumeration), then endpoint groups (pp
        # enumeration: singleton then pair per pp column).
        self.assertEqual(calls[:4], [(0, 2), (1, 3), (4, 6), (5, 7)])
        self.assertEqual(calls[4:], [(0,), (0, 4), (1,), (1, 5), (2,), (2, 6), (3,), (3, 7)])

    def test_grids_are_colocated_empty_and_mixed(self):
        self.assertTrue(grids_are_colocated({}))
        grids = build_module_grids(
            {
                "a": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=4),
                "b": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=4),
            }
        )
        self.assertTrue(grids_are_colocated(grids))
        grids["b"].rank_offset = 4  # disjoint now
        self.assertFalse(grids_are_colocated(grids))


if __name__ == "__main__":
    unittest.main()

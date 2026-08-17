# Copyright (c) 2025, BAAI. All rights reserved.

"""Unit tests for non-colocated MIMO parallel utilities.

The tests run on CPU: ``torch.distributed`` is mocked via ``patch``, and
``HyperCommGrid`` is replaced with lightweight mocks carrying only the
attributes used by the utilities (``rank_offset``, ``size``, ``shape``,
``dim_names``).
"""

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from flagscale.models.mimo.bridge import runtime as mpu

MODULE = "flagscale.models.mimo.bridge.runtime"


class TestGetDpSizeFromGrid:
    """Test cases for _get_dp_size_from_grid()."""

    def test_dp_size_from_shape_metadata(self):
        """DP size is read from grid.shape / grid.dim_names, no PGs required."""
        mock_grid = MagicMock()
        mock_grid.shape = [4, 2, 8]
        mock_grid.dim_names = ["tp", "dp", "pp"]

        assert mpu._get_dp_size_from_grid(mock_grid) == 2

    def test_dp_first_dimension(self):
        """DP as the leading dimension."""
        mock_grid = MagicMock()
        mock_grid.shape = [8]
        mock_grid.dim_names = ["dp"]

        assert mpu._get_dp_size_from_grid(mock_grid) == 8


class TestIsCurrentRankInGrid:
    """Test cases for is_current_rank_in_grid()."""

    @patch(f"{MODULE}.dist")
    def test_rank_in_grid(self, mock_dist):
        """Test rank within grid range returns True."""
        mock_dist.get_rank.return_value = 2
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4

        assert mpu.is_current_rank_in_grid(mock_grid) is True

    @patch(f"{MODULE}.dist")
    def test_rank_not_in_grid(self, mock_dist):
        """Test rank outside grid range returns False."""
        mock_dist.get_rank.return_value = 5
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4

        assert mpu.is_current_rank_in_grid(mock_grid) is False

    @patch(f"{MODULE}.dist")
    def test_rank_at_grid_boundary(self, mock_dist):
        """Test rank at grid boundaries (start inclusive, end exclusive)."""
        mock_grid = MagicMock()
        mock_grid.rank_offset = 4
        mock_grid.size = 4

        # At start boundary (inclusive)
        mock_dist.get_rank.return_value = 4
        assert mpu.is_current_rank_in_grid(mock_grid) is True

        # At end boundary (exclusive)
        mock_dist.get_rank.return_value = 8
        assert mpu.is_current_rank_in_grid(mock_grid) is False

    @patch(f"{MODULE}.dist")
    def test_rank_before_grid(self, mock_dist):
        """Test rank before grid range returns False."""
        mock_dist.get_rank.return_value = 2
        mock_grid = MagicMock()
        mock_grid.rank_offset = 4
        mock_grid.size = 4

        assert mpu.is_current_rank_in_grid(mock_grid) is False


class TestGetActiveModulePg:
    """Test cases for get_active_module_pg() with nullable collections."""

    def test_single_active_module(self):
        """Exactly one non-None collection is returned with its module name."""
        mock_pg = MagicMock()

        name, pg = mpu.get_active_module_pg({"encoder": None, "language": mock_pg})

        assert name == "language"
        assert pg is mock_pg

    def test_no_active_module_raises(self):
        """All-None collections must fail fast (stub rank)."""
        with pytest.raises(AssertionError, match="exactly one active"):
            mpu.get_active_module_pg({"encoder": None, "language": None})

    def test_multiple_active_modules_raise(self):
        """Colocated ranks (multiple active PGs) are not supported here."""
        mock_pg1 = MagicMock()
        mock_pg2 = MagicMock()

        with pytest.raises(AssertionError, match="exactly one active"):
            mpu.get_active_module_pg({"encoder": mock_pg1, "language": mock_pg2})


class _FakeMimoModel:
    """Stand-in for MimoModel used to exercise unwrap logic without mcore."""


class TestUnwrapMimoModel:
    """Test cases for unwrap_mimo_model()."""

    def test_unwraps_bare_model(self):
        """A bare MimoModel passes through unchanged."""
        with patch.object(mpu, "MimoModel", _FakeMimoModel):
            fake = _FakeMimoModel()
            assert mpu.unwrap_mimo_model(fake) is fake

    def test_unwraps_nested_wrappers(self):
        """Float16Module/DDP-style wrappers (with .module) are unwrapped."""
        with patch.object(mpu, "MimoModel", _FakeMimoModel):
            fake = _FakeMimoModel()
            wrapper = SimpleNamespace(module=SimpleNamespace(module=fake))
            assert mpu.unwrap_mimo_model(wrapper) is fake

    def test_raises_for_non_mimo_model(self):
        """A model that cannot be unwrapped to MimoModel raises RuntimeError."""
        with (
            patch.object(mpu, "MimoModel", _FakeMimoModel),
            pytest.raises(RuntimeError, match="Failed to unwrap model to MimoModel"),
        ):
            mpu.unwrap_mimo_model(object())

    def test_raises_when_chain_never_reaches_mimo_model(self):
        """Wrapper chains that never terminate at a MimoModel raise RuntimeError."""
        with patch.object(mpu, "MimoModel", _FakeMimoModel):
            chain = SimpleNamespace(module=SimpleNamespace(module=object()))
            with pytest.raises(RuntimeError, match="Failed to unwrap model to MimoModel"):
                mpu.unwrap_mimo_model(chain)


class TestGetModuleToGridTuple:
    """Test cases for get_module_to_grid_tuple()."""

    @patch(f"{MODULE}.is_current_rank_in_grid")
    def test_builds_tuples_for_participating_modules(self, mock_in_grid):
        """Language and modality modules on this rank map to (module, grid)."""
        llm_grid = MagicMock(name="llm_grid")
        vision_grid = MagicMock(name="vision_grid")
        mock_in_grid.side_effect = lambda grid: grid in (llm_grid, vision_grid)

        fake_lm = MagicMock(name="language_model")
        fake_vision = MagicMock(name="vision_model")
        fake_model = SimpleNamespace(
            language_model=fake_lm,
            modality_submodules={"vision": fake_vision},
        )

        with patch.object(mpu, "unwrap_mimo_model", return_value=fake_model):
            result = mpu.get_module_to_grid_tuple(
                SimpleNamespace(module=fake_model),  # wrapped, unwrapped by helper
                {"vision": vision_grid, "language": llm_grid},
            )

        assert (fake_vision, vision_grid) in result
        assert (fake_lm, llm_grid) in result

    @patch(f"{MODULE}.is_current_rank_in_grid")
    def test_skips_non_participating_modules(self, mock_in_grid):
        """Modules whose grid does not include this rank are skipped."""
        mock_in_grid.return_value = False

        fake_model = SimpleNamespace(
            language_model=MagicMock(),
            modality_submodules={"vision": MagicMock()},
        )

        with patch.object(mpu, "unwrap_mimo_model", return_value=fake_model):
            result = mpu.get_module_to_grid_tuple(
                fake_model,
                {"vision": MagicMock(), "language": MagicMock()},
            )

        assert result == []

    @patch(f"{MODULE}.is_current_rank_in_grid")
    def test_skips_unknown_modules_with_warning(self, mock_in_grid, caplog):
        """Module names absent from the model are skipped with a warning."""
        mock_in_grid.return_value = True

        fake_model = SimpleNamespace(
            language_model=MagicMock(),
            modality_submodules={},
        )

        with (
            patch.object(mpu, "unwrap_mimo_model", return_value=fake_model),
            caplog.at_level("WARNING", logger=mpu.logger.name),
        ):
            result = mpu.get_module_to_grid_tuple(
                fake_model,
                {"audio": MagicMock(), "language": MagicMock()},
            )

        assert len(result) == 1  # only the language module
        assert "not found in MimoModel" in caplog.text


class TestValidateNoStubRanks:
    """Test cases for validate_no_stub_ranks()."""

    def test_all_ranks_participate(self):
        """Test validation passes when all ranks participate."""
        mock_grid1 = MagicMock()
        mock_grid1.rank_offset = 0
        mock_grid1.size = 4

        mock_grid2 = MagicMock()
        mock_grid2.rank_offset = 4
        mock_grid2.size = 4

        module_to_grid_map = {
            "encoder": mock_grid1,
            "language": mock_grid2,
        }

        # Should not raise
        mpu.validate_no_stub_ranks(module_to_grid_map, world_size=8)

    def test_stub_ranks_detected(self):
        """Test validation fails when stub ranks exist."""
        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4

        module_to_grid_map = {"language": mock_grid}

        with pytest.raises(ValueError, match="do not participate in any module"):
            mpu.validate_no_stub_ranks(module_to_grid_map, world_size=8)

    def test_overlapping_grids(self):
        """Test validation with overlapping grids (colocated case)."""
        mock_grid1 = MagicMock()
        mock_grid1.rank_offset = 0
        mock_grid1.size = 4

        mock_grid2 = MagicMock()
        mock_grid2.rank_offset = 0
        mock_grid2.size = 4

        module_to_grid_map = {
            "encoder": mock_grid1,
            "language": mock_grid2,
        }

        # Should not raise (all 4 ranks participate)
        mpu.validate_no_stub_ranks(module_to_grid_map, world_size=4)


class TestValidateDataLoaderContract:
    """Test cases for validate_data_loader_contract()."""

    def _dp_grid(self, dp_size):
        mock_grid = MagicMock()
        mock_grid.shape = [dp_size]
        mock_grid.dim_names = ["dp"]
        return mock_grid

    def test_valid_configuration(self):
        """Test validation passes for valid configuration."""
        module_to_grid_map = {"language": self._dp_grid(dp_size=2)}

        # global_batch=8, dp=2, microbatches=2, global micro_batch_size=4.
        # Each module-local DP rank sees 4 / 2 = 2 samples per microbatch.
        mpu.validate_data_loader_contract(
            module_to_grid_map=module_to_grid_map,
            global_batch_size=8,
            micro_batch_size=4,
            num_microbatches=2,
        )

    def test_batch_not_divisible_by_dp(self):
        """Test validation fails when batch not divisible by DP size."""
        module_to_grid_map = {"language": self._dp_grid(dp_size=3)}

        with pytest.raises(ValueError, match="not divisible"):
            mpu.validate_data_loader_contract(
                module_to_grid_map=module_to_grid_map,
                global_batch_size=8,
                micro_batch_size=4,
                num_microbatches=2,
            )

    def test_microbatch_count_mismatch(self):
        """Test validation fails when accumulation does not match global batch."""
        module_to_grid_map = {"language": self._dp_grid(dp_size=2)}

        with pytest.raises(ValueError, match="Microbatch mismatch"):
            mpu.validate_data_loader_contract(
                module_to_grid_map=module_to_grid_map,
                global_batch_size=16,
                micro_batch_size=4,
                num_microbatches=2,
            )


class TestBuildPgCollectionForSchedule:
    """Test cases for build_pg_collection_for_schedule()."""

    def test_uses_multimodule_collection_with_language_key(self):
        """MultiModuleProcessGroupCollection is built with the language key set."""
        mock_pg1 = MagicMock()
        mock_pg2 = MagicMock()

        result = mpu.build_pg_collection_for_schedule({"encoder": mock_pg1, "language": mock_pg2})

        # In the v0.18.2 env MultiModuleProcessGroupCollection is importable.
        assert result is not None
        assert result.module_pgs == {"encoder": mock_pg1, "language": mock_pg2}
        assert result.language_model_module_name == "language"

    def test_filters_none_pg_collections(self):
        """None pg_collections are filtered out before building the collection."""
        mock_pg = MagicMock()

        result = mpu.build_pg_collection_for_schedule(
            {"encoder": None, "language": mock_pg}  # Non-participating module
        )

        assert result is not None
        assert result.module_pgs == {"language": mock_pg}
        assert result.language_model_module_name == "language"

    def test_constructor_failure_propagates(self):
        """A genuine MultiModuleProcessGroupCollection failure must not be swallowed.

        The old code caught (ImportError, ValueError, TypeError) and silently
        downgraded to a plain list of collections, which the schedule consumes
        with different semantics.  Configuration errors must propagate.
        """
        mock_pg1 = MagicMock()
        mock_pg2 = MagicMock()

        with patch(f"{MODULE}.MultiModuleProcessGroupCollection") as mock_multimodule_cls:
            mock_multimodule_cls.side_effect = TypeError("construction failed")
            with pytest.raises(TypeError, match="construction failed"):
                mpu.build_pg_collection_for_schedule({"encoder": mock_pg1, "language": mock_pg2})

    def test_all_none_collections_fail_fast(self):
        """An all-None map (stub rank) must raise instead of returning [].

        The empty map is a configuration error (every rank must participate in
        at least one module); ``validate_no_stub_ranks`` rejects it at setup
        time, and this helper must not paper over it at schedule time.
        """
        with pytest.raises(ValueError, match="module_pgs dict cannot be empty"):
            mpu.build_pg_collection_for_schedule({"encoder": None, "language": None})


class TestMultimoduleNoSync:
    """Test cases for multimodule_no_sync context manager."""

    @patch(f"{MODULE}.is_current_rank_in_grid")
    def test_enters_and_exits_contexts(self, mock_in_grid):
        """Test that no_sync contexts are properly entered and exited."""
        mock_in_grid.return_value = True

        mock_module = MagicMock()
        mock_context = MagicMock()
        mock_module.no_sync.return_value = mock_context

        mock_grid = MagicMock()

        module_to_grid_tuple = [(mock_module, mock_grid)]

        with mpu.multimodule_no_sync(module_to_grid_tuple=module_to_grid_tuple):
            pass

        # Verify context was entered and exited
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()

    @patch(f"{MODULE}.is_current_rank_in_grid")
    def test_skips_non_participating_modules(self, mock_in_grid):
        """Test that non-participating modules are skipped."""
        mock_in_grid.return_value = False  # Not participating

        mock_module = MagicMock()
        mock_grid = MagicMock()

        module_to_grid_tuple = [(mock_module, mock_grid)]

        with mpu.multimodule_no_sync(module_to_grid_tuple=module_to_grid_tuple):
            pass

        # no_sync should not be called
        mock_module.no_sync.assert_not_called()

    @patch(f"{MODULE}.is_current_rank_in_grid")
    def test_fails_fast_on_raw_module_without_no_sync(self, mock_in_grid):
        """A raw (non-DDP) module must raise instead of silently skipping.

        ``get_module_to_grid_tuple`` returns raw unwrapped modules, so the
        DDP-only ``no_sync`` may be absent; failing fast beats silently
        running without gradient-sync control.
        """
        mock_in_grid.return_value = True

        raw_module = SimpleNamespace()  # no no_sync (raw unwrapped module)
        mock_grid = MagicMock()

        with (
            pytest.raises(AttributeError, match="no_sync"),
            mpu.multimodule_no_sync(module_to_grid_tuple=[(raw_module, mock_grid)]),
        ):
            pass


class TestZeroGradBufferForMultimodule:
    """Test cases for zero_grad_buffer_for_multimodule()."""

    @patch(f"{MODULE}.is_current_rank_in_grid")
    def test_zeros_grad_buffers(self, mock_in_grid):
        """Test gradient buffers are zeroed for participating modules."""
        mock_in_grid.return_value = True

        mock_module = MagicMock()
        mock_grid = MagicMock()

        module_to_grid_tuple = [(mock_module, mock_grid)]

        mpu.zero_grad_buffer_for_multimodule(module_to_grid_tuple)

        mock_module.zero_grad_buffer.assert_called_once()

    @patch(f"{MODULE}.is_current_rank_in_grid")
    def test_skips_non_participating(self, mock_in_grid):
        """Test non-participating modules are skipped."""
        mock_in_grid.return_value = False

        mock_module = MagicMock()
        mock_grid = MagicMock()

        module_to_grid_tuple = [(mock_module, mock_grid)]

        mpu.zero_grad_buffer_for_multimodule(module_to_grid_tuple)

        mock_module.zero_grad_buffer.assert_not_called()

    @patch(f"{MODULE}.is_current_rank_in_grid")
    def test_fails_fast_when_zero_grad_buffer_missing(self, mock_in_grid):
        """A raw (non-DDP) module must raise instead of silently skipping.

        A silently skipped ``zero_grad_buffer`` means gradient buffers are
        never reset and gradients accumulate across optimizer steps.
        """
        mock_in_grid.return_value = True

        raw_module = SimpleNamespace()  # no zero_grad_buffer (raw unwrapped module)
        mock_grid = MagicMock()

        with pytest.raises(AttributeError, match="zero_grad_buffer"):
            mpu.zero_grad_buffer_for_multimodule([(raw_module, mock_grid)])


def _build_two_module_setup(llm_dp, encoder_dp, *, llm_rank_offset=4, llm_size=4):
    """Build an encoder + language MIMO setup for finalize_model_grads tests.

    The same grid objects are shared between ``module_to_grid_map`` and
    ``module_to_grid_tuple`` because the function matches modules to grids by
    identity (``mg is grid``).
    """
    llm_key = mpu.MIMO_LANGUAGE_MODULE_KEY
    encoder_key = "encoder"

    llm_grid = MagicMock(name="llm_grid")
    llm_grid.rank_offset = llm_rank_offset
    llm_grid.size = llm_size
    encoder_grid = MagicMock(name="encoder_grid")

    llm_module = MagicMock(name="llm_module")
    encoder_module = MagicMock(name="encoder_module")
    llm_pg = MagicMock(name="llm_pg")
    encoder_pg = MagicMock(name="encoder_pg")

    module_to_grid_map = {encoder_key: encoder_grid, llm_key: llm_grid}
    pg_collections = {encoder_key: encoder_pg, llm_key: llm_pg}

    # Encoder listed first to verify per-module routing is independent of order.
    module_to_grid_tuple = [(encoder_module, encoder_grid), (llm_module, llm_grid)]

    dp_by_grid = {id(llm_grid): llm_dp, id(encoder_grid): encoder_dp}

    return types.SimpleNamespace(
        module_to_grid_map=module_to_grid_map,
        pg_collections=pg_collections,
        module_to_grid_tuple=module_to_grid_tuple,
        llm_grid=llm_grid,
        encoder_grid=encoder_grid,
        llm_module=llm_module,
        encoder_module=encoder_module,
        llm_pg=llm_pg,
        encoder_pg=encoder_pg,
        dp_by_grid=dp_by_grid,
    )


class TestFinalizeModelGradsMultimodule:
    """Test cases for finalize_model_grads_multimodule().

    The function selects its gradient-normalization branch on
    ``num_tokens is not None``, which is exactly ``calculate_per_token_loss``:
    Megatron-Core's schedule passes ``total_num_tokens if
    config.calculate_per_token_loss else None`` to ``finalize_model_grads_func``.
    These tests pin that equivalence so the branch keying stays correct.
    """

    @patch(f"{MODULE}.dist")
    @patch(f"{MODULE}.is_current_rank_in_grid")
    @patch(f"{MODULE}._get_dp_size_from_grid")
    @patch(f"{MODULE}._finalize_model_grads")
    def test_per_token_loss_path(self, mock_finalize, mock_dp, mock_in_grid, mock_dist):
        """num_tokens is not None (calculate_per_token_loss=True) path.

        Only the LLM receives num_tokens (so MCore can PP-broadcast + DP-all-reduce
        the per-rank counts); encoder grads are normalized manually by 1/total with
        no DP compensation factor.
        """
        s = _build_two_module_setup(llm_dp=2, encoder_dp=4)
        mock_in_grid.return_value = True
        mock_dp.side_effect = lambda grid: s.dp_by_grid[id(grid)]

        num_tokens = torch.tensor(100)

        mpu.finalize_model_grads_multimodule(
            [MagicMock()],  # model arg is ignored
            num_tokens,
            force_all_reduce=True,
            module_to_grid_map=s.module_to_grid_map,
            pg_collections=s.pg_collections,
            module_to_grid_tuple=s.module_to_grid_tuple,
        )

        # Phase 1: LLM finalized with num_tokens, encoder with num_tokens=None.
        finalize_by_module = {call.args[0][0]: call.kwargs for call in mock_finalize.call_args_list}
        assert finalize_by_module[s.llm_module]["num_tokens"] is num_tokens
        assert finalize_by_module[s.llm_module]["pg_collection"] is s.llm_pg
        assert finalize_by_module[s.llm_module]["force_all_reduce"] is True
        assert finalize_by_module[s.encoder_module]["num_tokens"] is None
        assert finalize_by_module[s.encoder_module]["pg_collection"] is s.encoder_pg
        assert finalize_by_module[s.encoder_module]["force_all_reduce"] is True

        # Phase 2: broadcast the global total from the LLM's last rank (4 + 4 - 1).
        mock_dist.broadcast.assert_called_once()
        assert mock_dist.broadcast.call_args.args[0] is num_tokens
        assert mock_dist.broadcast.call_args.kwargs["src"] == 7

        # Phase 3: encoder scaled by 1/total only; LLM already normalized by DDP.
        s.encoder_module.scale_gradients.assert_called_once_with(1.0 / 100)
        s.llm_module.scale_gradients.assert_not_called()

    @patch(f"{MODULE}.dist")
    @patch(f"{MODULE}.is_current_rank_in_grid")
    @patch(f"{MODULE}._get_dp_size_from_grid")
    @patch(f"{MODULE}._finalize_model_grads")
    def test_non_per_token_loss_path_applies_dp_compensation(
        self, mock_finalize, mock_dp, mock_in_grid, mock_dist
    ):
        """num_tokens is None (calculate_per_token_loss=False) path.

        Every module is finalized with num_tokens=None, no broadcast happens, and
        modules whose DP differs from the LLM's are rescaled by module_dp/llm_dp.
        """
        s = _build_two_module_setup(llm_dp=2, encoder_dp=4)
        mock_in_grid.return_value = True
        mock_dp.side_effect = lambda grid: s.dp_by_grid[id(grid)]

        mpu.finalize_model_grads_multimodule(
            [MagicMock()],
            None,
            force_all_reduce=True,
            module_to_grid_map=s.module_to_grid_map,
            pg_collections=s.pg_collections,
            module_to_grid_tuple=s.module_to_grid_tuple,
        )

        # All modules finalized without num_tokens (DDP does a plain mean).
        for call in mock_finalize.call_args_list:
            assert call.kwargs["num_tokens"] is None
            assert call.kwargs["force_all_reduce"] is True
        assert mock_dist.broadcast.call_count == 0

        # encoder_dp (4) != llm_dp (2) -> scale by 4/2; LLM matches llm_dp -> no scale.
        s.encoder_module.scale_gradients.assert_called_once_with(2.0)
        s.llm_module.scale_gradients.assert_not_called()

    @patch(f"{MODULE}.dist")
    @patch(f"{MODULE}.is_current_rank_in_grid")
    @patch(f"{MODULE}._get_dp_size_from_grid")
    @patch(f"{MODULE}._finalize_model_grads")
    def test_per_token_loss_path_skips_scaling_when_zero_tokens(
        self, mock_finalize, mock_dp, mock_in_grid, mock_dist
    ):
        """Zero global tokens must not trigger a divide-by-zero in encoder scaling."""
        s = _build_two_module_setup(llm_dp=2, encoder_dp=4)
        mock_in_grid.return_value = True
        mock_dp.side_effect = lambda grid: s.dp_by_grid[id(grid)]

        mpu.finalize_model_grads_multimodule(
            [MagicMock()],
            torch.tensor(0),
            module_to_grid_map=s.module_to_grid_map,
            pg_collections=s.pg_collections,
            module_to_grid_tuple=s.module_to_grid_tuple,
        )

        s.encoder_module.scale_gradients.assert_not_called()
        s.llm_module.scale_gradients.assert_not_called()

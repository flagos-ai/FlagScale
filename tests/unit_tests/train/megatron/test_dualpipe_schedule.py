"""Unit tests for the DualPipe pipeline parallel schedule.

These tests validate:
1. Configuration validation rejects invalid parameter combinations.
2. The :class:`WeightGradStore` and :class:`_SliceIterator` utilities work
   correctly in isolation (no distributed environment required).
3. The :func:`_split_data_iterator` helper splits iterators correctly.

The tests in this file work without a full Megatron installation by stubbing
the megatron namespace packages at import time.
"""

import importlib.util
import os
import queue
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Bootstrap: load dualpipe_schedule with stubbed Megatron dependencies
# ---------------------------------------------------------------------------

_DUALPIPE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # tests/unit_tests/train/megatron/
        "../../../../flagscale/train/megatron/training/dualpipe_schedule.py",
    )
)

_MODULE_NAME = "dualpipe_schedule_under_test"


def _load_dualpipe_module():
    """Load dualpipe_schedule.py with stubs for missing Megatron deps."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    # Stub out every Megatron submodule the schedule imports at the top level.
    for name in [
        "megatron",
        "megatron.core",
        "megatron.core.parallel_state",
        "megatron.training",
        "megatron.core.utils",
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # Populate parallel_state stub
    ps = sys.modules["megatron.core.parallel_state"]
    ps.get_pipeline_model_parallel_group = lambda: None
    ps.get_pipeline_model_parallel_world_size = lambda: 4
    ps.get_pipeline_model_parallel_rank = lambda: 0
    ps.get_tensor_model_parallel_world_size = lambda: 1

    # Populate megatron.core stub
    core = sys.modules["megatron.core"]
    core.parallel_state = ps

    # Populate megatron.core.utils stub
    utils = sys.modules["megatron.core.utils"]
    utils.get_attr_wrapped_model = lambda m, k: getattr(m, k, None)
    utils.get_model_config = lambda m: None
    core.utils = utils

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _DUALPIPE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


# Load once at module import time.
_dp = _load_dualpipe_module()


# ---------------------------------------------------------------------------
# WeightGradStore tests
# ---------------------------------------------------------------------------

class TestWeightGradStore:
    """Tests for :class:`WeightGradStore` (no torch.distributed needed)."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        """Clear WeightGradStore before and after each test."""
        _dp.WeightGradStore.clear()
        yield
        _dp.WeightGradStore.clear()

    def test_put_flush_pop_in_order(self):
        """WeightGradStore accumulates functions, flushes, and pops in order."""
        WGS = _dp.WeightGradStore
        results = []

        WGS.put(lambda: results.append(1))
        WGS.put(lambda: results.append(2))
        WGS.flush()

        assert WGS.funcs_queue.qsize() == 1
        assert WGS.cache == []

        WGS.pop()
        assert results == [1, 2]
        assert WGS.funcs_queue.empty()

    def test_multiple_flushes_fifo(self):
        """Multiple flush/pop cycles respect FIFO order."""
        WGS = _dp.WeightGradStore
        order = []

        WGS.put(lambda: order.append("a"))
        WGS.flush()
        WGS.put(lambda: order.append("b"))
        WGS.flush()

        WGS.pop()
        assert order == ["a"]
        WGS.pop()
        assert order == ["a", "b"]

    def test_clear_resets_all_state(self):
        """clear() empties both cache and queue."""
        WGS = _dp.WeightGradStore
        WGS.put(lambda: None)
        WGS.flush()
        WGS.put(lambda: None)
        WGS.clear()

        assert WGS.cache == []
        assert WGS.funcs_queue.empty()

    def test_pop_empty_raises(self):
        """pop() on an empty queue raises AssertionError."""
        WGS = _dp.WeightGradStore
        WGS.clear()
        with pytest.raises(AssertionError):
            WGS.pop()


# ---------------------------------------------------------------------------
# _SliceIterator tests
# ---------------------------------------------------------------------------

class TestSliceIterator:
    """Tests for :class:`_SliceIterator`."""

    def test_basic_iteration(self):
        it = _dp._SliceIterator([1, 2, 3])
        assert next(it) == 1
        assert next(it) == 2
        assert next(it) == 3
        with pytest.raises(StopIteration):
            next(it)

    def test_empty_list(self):
        it = _dp._SliceIterator([])
        with pytest.raises(StopIteration):
            next(it)

    def test_for_loop(self):
        result = list(_dp._SliceIterator([10, 20, 30]))
        assert result == [10, 20, 30]

    def test_iterable_protocol(self):
        it = _dp._SliceIterator([42])
        assert iter(it) is it


# ---------------------------------------------------------------------------
# _split_data_iterator tests
# ---------------------------------------------------------------------------

class TestSplitDataIterator:
    """Tests for :func:`_split_data_iterator`."""

    def test_equal_halves(self):
        source = iter(range(8))
        iters = _dp._split_data_iterator(source, num_microbatches=8)
        assert len(iters) == 2
        assert list(iters[0]) == [0, 1, 2, 3]
        assert list(iters[1]) == [4, 5, 6, 7]

    def test_consumes_all_items(self):
        source = iter(range(6))
        _dp._split_data_iterator(source, num_microbatches=6)
        with pytest.raises(StopIteration):
            next(source)

    def test_minimal_split(self):
        source = iter(["a", "b"])
        iters = _dp._split_data_iterator(source, num_microbatches=2)
        assert list(iters[0]) == ["a"]
        assert list(iters[1]) == ["b"]


# ---------------------------------------------------------------------------
# DualPipe argument validation tests (pure logic, no Megatron dependency)
# ---------------------------------------------------------------------------

def _validate_dualpipe(args):
    """Standalone re-implementation of the DualPipe validation logic.

    Mirrors the assertions in ``FSTrainArguments.post_validate_args``
    so they can be tested without a full Megatron environment.
    """
    if not getattr(args, 'use_dualpipe', False):
        return
    assert args.pipeline_model_parallel_size > 1, \
        "DualPipe requires pipeline parallelism."
    assert args.pipeline_model_parallel_size % 2 == 0, \
        "DualPipe requires an even pipeline-model-parallel-size."
    assert getattr(args, 'virtual_pipeline_model_parallel_size', None) is None, \
        "DualPipe is incompatible with virtual pipeline parallelism."
    assert not getattr(args, 'use_dualpipev', False), \
        "DualPipe and DualPipeV cannot be enabled simultaneously."
    assert getattr(args, 'untie_embeddings_and_output_weights', True) is True, \
        "DualPipe requires untied embeddings and output weights."
    if args.micro_batch_size is not None and args.data_parallel_size is not None:
        num_micro = args.global_batch_size // (
            args.micro_batch_size * args.data_parallel_size
        )
        assert num_micro % 2 == 0, \
            f"DualPipe requires an even number of micro-batches, got {num_micro}."
        assert num_micro >= args.pipeline_model_parallel_size * 2, \
            "DualPipe requires num_microbatches >= pipeline_parallel_size * 2."


def _make_args(**overrides):
    """Create a minimal args namespace for DualPipe validation tests."""
    defaults = dict(
        use_dualpipe=True,
        use_dualpipev=False,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=None,
        untie_embeddings_and_output_weights=True,
        micro_batch_size=2,
        global_batch_size=64,
        data_parallel_size=2,   # num_micro = 64/(2*2) = 16  >=  4*2=8  OK
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestDualPipeValidation:
    """Tests for DualPipe configuration validation rules."""

    def test_valid_config_passes(self):
        _validate_dualpipe(_make_args())

    def test_odd_pp_size_raises(self):
        with pytest.raises(AssertionError, match="even"):
            _validate_dualpipe(_make_args(pipeline_model_parallel_size=3))

    def test_pp_size_1_raises(self):
        with pytest.raises(AssertionError):
            _validate_dualpipe(_make_args(pipeline_model_parallel_size=1))

    def test_virtual_pp_raises(self):
        with pytest.raises(AssertionError, match="virtual"):
            _validate_dualpipe(_make_args(virtual_pipeline_model_parallel_size=2))

    def test_simultaneous_dualpipev_raises(self):
        with pytest.raises(AssertionError, match="DualPipeV"):
            _validate_dualpipe(_make_args(use_dualpipev=True))

    def test_tied_embeddings_raises(self):
        with pytest.raises(AssertionError, match="untied"):
            _validate_dualpipe(_make_args(untie_embeddings_and_output_weights=False))

    def test_odd_num_microbatches_raises(self):
        # global=60, micro=2, dp=2  =>  num_micro = 15 (odd)
        with pytest.raises(AssertionError, match="even"):
            _validate_dualpipe(_make_args(global_batch_size=60))

    def test_too_few_microbatches_raises(self):
        # pp=4 => need num_micro>=8; global=16/micro=2/dp=2 => num_micro=4 (<8)
        with pytest.raises(AssertionError, match="num_microbatches"):
            _validate_dualpipe(_make_args(
                pipeline_model_parallel_size=4,
                global_batch_size=16,
                micro_batch_size=2,
                data_parallel_size=2,
            ))

    def test_disabled_skips_validation(self):
        # use_dualpipe=False => odd pp_size should not raise.
        _validate_dualpipe(_make_args(use_dualpipe=False, pipeline_model_parallel_size=3))


# ---------------------------------------------------------------------------
# Smoke test: forward_backward_dualpipe is callable
# ---------------------------------------------------------------------------

def test_forward_backward_dualpipe_callable():
    """forward_backward_dualpipe must be a callable."""
    assert callable(_dp.forward_backward_dualpipe)


def test_get_dualpipe_forward_backward_func_returns_correct_callable():
    """get_dualpipe_forward_backward_func() must return forward_backward_dualpipe."""
    func = _dp.get_dualpipe_forward_backward_func()
    assert callable(func)
    assert func is _dp.forward_backward_dualpipe

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import builtins
import functools
import logging
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

TRAINING = Path(__file__).resolve().parents[4] / "flagscale/train/megatron/training"


def definitions(filename, names, namespace):
    tree = ast.parse((TRAINING / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def globals_namespace():
    namespace = {"_GLOBAL_MEGALENS_RUNTIME": None}
    return definitions(
        "global_vars.py",
        {
            "_set_megalens_runtime",
            "get_megalens_runtime",
            "get_tracer",
            "shutdown_megalens_runtime",
            "_ensure_var_is_initialized",
            "_ensure_var_is_not_initialized",
        },
        namespace,
    )


def trace_args(**overrides):
    namespace = definitions(
        "arguments.py", {"_add_megalens_args", "_validate_megalens_args"}, {"warnings": warnings}
    )
    parser = namespace["_add_megalens_args"](argparse.ArgumentParser())
    args = parser.parse_args([])
    args.cuda_graph_impl = "none"
    args.profile = False
    args.use_pytorch_profiler = False
    vars(args).update(overrides)
    return namespace, args


def test_trace_defaults_remain_disabled():
    namespace, args = trace_args()
    assert not args.trace
    assert args.trace_cupti_kernels == "off"
    namespace["_validate_megalens_args"](args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"hardware_monitor": True},
        {"trace": True, "trace_interval": 0},
        {"trace": True, "continuous_trace_iterations": 0},
        {"trace": True, "continuous_trace_iterations": 1001},
        {"trace": True, "sentinel_hw_sample_ms": 0},
        {"trace": True, "sentinel_flush_interval": 0},
        {"trace": True, "trace_cupti_kernels": "on", "profile": True, "use_pytorch_profiler": True},
    ],
)
def test_invalid_trace_configuration_is_rejected(overrides):
    namespace, args = trace_args(**overrides)
    with pytest.raises(ValueError):
        namespace["_validate_megalens_args"](args)


def test_disabled_runtime_does_not_import_megalens(monkeypatch):
    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name.startswith("megatron.megalens"):
            pytest.fail("trace-off must not load MegaLens")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    namespace = globals_namespace()
    namespace["_set_megalens_runtime"](SimpleNamespace(trace=False))
    assert namespace["get_tracer"]() is None
    namespace["shutdown_megalens_runtime"](graceful=True)


def test_runtime_creation_and_shutdown_are_lazy_and_idempotent(monkeypatch):
    original = builtins.__import__
    calls = []
    runtime = SimpleNamespace(tracer=object(), shutdown=lambda **kwargs: calls.append(kwargs))

    def import_module(name, *args, **kwargs):
        if name == "megatron.megalens.runtime":
            return SimpleNamespace(MegaLensRuntime=lambda args: runtime)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    namespace = globals_namespace()
    namespace["_set_megalens_runtime"](SimpleNamespace(trace=True))
    assert namespace["get_megalens_runtime"]() is runtime
    assert namespace["get_tracer"]() is runtime.tracer
    namespace["shutdown_megalens_runtime"](graceful=True)
    namespace["shutdown_megalens_runtime"](graceful=False)
    assert calls == [{"graceful": True}]
    assert namespace["_GLOBAL_MEGALENS_RUNTIME"] is None


@pytest.mark.parametrize("failure", [False, True])
def test_pretrain_owner_preserves_result_or_original_exception(failure):
    calls = []
    namespace = definitions(
        "training.py",
        {"_owns_megalens_runtime"},
        {
            "functools": functools,
            "logging": logging,
            "shutdown_megalens_runtime": lambda **kwargs: calls.append(kwargs),
        },
    )
    expected = RuntimeError("training failed") if failure else object()

    def pretrain():
        if failure:
            raise expected
        return expected

    wrapped = namespace["_owns_megalens_runtime"](pretrain)
    if failure:
        with pytest.raises(RuntimeError) as error:
            wrapped()
        assert error.value is expected
    else:
        assert wrapped() is expected
    assert calls == [{"graceful": not failure}]


def run_training_envelope(trace, skip, fail=False, mode0=True):
    tree = ast.parse((TRAINING / "training.py").read_text())
    train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    loop = next(node for node in train.body if isinstance(node, ast.While))
    end = next(
        index
        for index, node in enumerate(loop.body)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "iteration"
    )
    loop.body = loop.body[: end + 1]
    wrapper = ast.parse(
        "def run(train_data_iterator):\n    iteration = 0\n    start_iteration = -1\n    num_microbatches = 1\n"
    ).body[0]
    wrapper.body.append(loop)
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    events = []

    @contextmanager
    def iteration_scope(iteration, **kwargs):
        events.append(("begin", iteration))
        try:
            yield
        finally:
            events.append(("end", iteration))

    args = SimpleNamespace(
        trace=trace,
        train_iters=3,
        iterations_to_skip=[2] if skip == "iteration" else [],
        skip_iters_range=[0, 2] if skip == "range" else None,
        skip_samples_range=None,
        profile=False,
        distributed_timeout_seconds_after_init=None,
        consumed_train_samples=0,
        skipped_train_samples=0,
        cuda_graph_impl="none",
        perform_rl_step=False,
        skip_train=False,
        micro_batch_size=1,
        data_parallel_size=1,
    )
    data = iter(range(10))
    runtime = SimpleNamespace(
        tracer=SimpleNamespace(
            is_mode0=lambda: mode0, close_trace_window=lambda: events.append(("close",))
        ),
        iteration=iteration_scope,
    )

    def train_step(*unused, iteration):
        events.append(("train", iteration))
        if fail:
            raise RuntimeError("training failed")
        return {}, 0, False, False, 0, None, None, None

    def dummy_step(iterator):
        next(iterator)
        events.append(("skip",))

    no_op = lambda *args, **kwargs: None
    namespace = {
        "args": args,
        "ExitStack": ExitStack,
        "get_megalens_runtime": lambda: runtime
        if trace
        else pytest.fail("trace-off runtime access"),
        "get_current_global_batch_size": lambda: 1,
        "get_num_microbatches": lambda: 1,
        "update_num_microbatches": no_op,
        "ft_integration": SimpleNamespace(
            on_checkpointing_start=no_op,
            on_checkpointing_end=no_op,
            on_training_step_start=no_op,
            on_training_step_end=no_op,
        ),
        "maybe_finalize_async_save": no_op,
        "dummy_train_step": dummy_step,
        "mpu": SimpleNamespace(
            is_pipeline_first_stage=lambda: True,
            is_pipeline_last_stage=lambda: False,
            get_data_parallel_world_size=lambda: 1,
        ),
        "rerun_state_machine": SimpleNamespace(current_iteration=0),
        "print_rank_0": no_op,
        "torch": SimpleNamespace(
            distributed=SimpleNamespace(barrier=lambda: events.append(("barrier",)))
        ),
        "train_step": train_step,
        "train_data_iterator": data,
        "forward_step_func": object(),
        "model": object(),
        "optimizer": object(),
        "opt_param_scheduler": object(),
        "config": object(),
        "forward_backward_func": object(),
        "perf_callback": None,
        "_maybe_raise_workload_exception": None,
    }
    exec(compile(module, str(TRAINING / "training.py"), "exec"), namespace)
    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            namespace["run"](data)
    else:
        namespace["run"](data)
    return events


@pytest.mark.parametrize("skip", ["none", "iteration", "range"])
def test_trace_off_preserves_native_iteration_and_skip_behavior(skip):
    events = run_training_envelope(False, skip)
    expected = (
        [("train", 0), ("skip",), ("train", 2)]
        if skip == "iteration"
        else ([("train", 2)] if skip == "range" else [("train", 0), ("train", 1), ("train", 2)])
    )
    assert events == expected


@pytest.mark.parametrize("skip,steps", [("none", [1, 2, 3]), ("iteration", [1, 3]), ("range", [3])])
def test_trace_windows_follow_actual_training_after_skip(skip, steps):
    events = run_training_envelope(True, skip)
    assert [event[1] for event in events if event[0] == "begin"] == steps
    assert [event[1] for event in events if event[0] == "end"] == steps
    assert sum(event[0] == "close" for event in events) == (skip != "none")
    assert [event[1] + 1 for event in events if event[0] == "train"] == steps


def test_iteration_scope_closes_on_training_exception():
    assert run_training_envelope(True, "none", fail=True) == [
        ("begin", 1),
        ("train", 0),
        ("end", 1),
    ]


def test_mode1_keeps_one_alignment_barrier_per_trained_iteration():
    events = run_training_envelope(True, "iteration", mode0=False)
    assert sum(event[0] == "barrier" for event in events) == 2


@pytest.mark.parametrize(
    "trace,skip,schedule,rejected",
    [
        (True, [0, 2], [0, 1, 2], True),
        (True, [0, 2], None, False),
        (False, [0, 2], [0, 1, 2], False),
        (True, None, [0, 1, 2], False),
    ],
)
def test_sample_range_skip_uses_current_dynamic_batch_schedule_guard(
    trace, skip, schedule, rejected
):
    tree = ast.parse((TRAINING / "training.py").read_text())
    train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    guard = next(
        node
        for node in train.body
        if isinstance(node, ast.If) and "step_batch_size_schedule" in ast.unparse(node.test)
    )
    namespace = {
        "args": SimpleNamespace(
            trace=trace, skip_samples_range=skip, step_batch_size_schedule=schedule
        )
    }
    code = compile(ast.Module(body=[guard], type_ignores=[]), "sample_range_guard", "exec")
    if rejected:
        with pytest.raises(ValueError, match="step-batch-size-schedule"):
            exec(code, namespace)
    else:
        exec(code, namespace)

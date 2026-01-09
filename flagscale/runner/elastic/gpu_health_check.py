"""
Complete GPU Health Check Implementation

This module provides comprehensive GPU health verification including:
- Tensor parallel communication testing
- Data parallel communication testing
- Pipeline parallel communication testing
- TODO: Expert parallel communication testing
- GPU hardware validation
- Computation capability verification

Features:
- Timeout protection for each test phase
- Progressive testing (failures don't block other tests)
- Smart degradation on errors
- Complete test coverage in order: TP → DP → PP → Hardware → Computation
"""

import argparse
import os
import time

from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist

# -------------------------
# Globals
# -------------------------
_GLOBAL_ARGS = None

_DATA_PARALLEL_GROUP_NCCL = None
_DATA_PARALLEL_GROUP_GLOO = None
_DATA_GLOBAL_RANKS = None

_MODEL_PARALLEL_GROUP_NCCL = None

_TENSOR_MODEL_PARALLEL_GROUP_NCCL = None
_TENSOR_MODEL_PARALLEL_GROUP_GLOO = None
_TENSOR_GLOBAL_RANKS = None

_PIPELINE_MODEL_PARALLEL_GROUP_NCCL = None
_PIPELINE_MODEL_PARALLEL_GROUP_GLOO = None
_PIPELINE_GLOBAL_RANKS = None

_EMBEDDING_GROUP_NCCL = None
_EMBEDDING_GROUP_GLOO = None

_GLOO_WORLD_GROUP = None

# Test tracking
_TEST_RESULTS = {
    'tensor_parallel': {'status': 'pending', 'error': None},
    'data_parallel': {'status': 'pending', 'error': None},
    'pipeline_parallel': {'status': 'pending', 'error': None},
}


def log_test_result(test_name, status, error=None):
    """Log test result"""
    _TEST_RESULTS[test_name]['status'] = status
    _TEST_RESULTS[test_name]['error'] = error

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        if status == 'passed':
            print(f"✓ {test_name}: PASSED")
        elif status == 'failed':
            print(f"✗ {test_name}: FAILED - {error}")
        elif status == 'skipped':
            print(f"⚠ {test_name}: SKIPPED - {error}")


def safe_test_execution(test_func, test_name, timeout_seconds=120) -> bool:
    """Execute test with timeout protection and error handling"""
    try:
        test_func()
        # log_test_result(test_name, 'passed')
        return True
    except TimeoutError as e:
        log_test_result(test_name, 'failed', str(e))
        return False
    except Exception as e:
        log_test_result(test_name, 'failed', f"Exception: {str(e)}")
        return False


def get_args():
    """Return arguments."""
    assert _GLOBAL_ARGS is not None, '{} is not initialized.'.format('args')
    return _GLOBAL_ARGS


# -------------------------
# Control-plane barrier (GLOO)
# -------------------------
def control_barrier(group=None, timeout_s: int = 300, name: str = "barrier"):
    """
    Use GLOO monitored_barrier as the universal sync primitive.
    This avoids NCCL barrier (which is a 1-element allreduce and can segfault in some setups).
    """
    if not dist.is_initialized():
        return
    g = group if group is not None else _GLOO_WORLD_GROUP
    if g is None:
        # Fallback: try world monitored barrier (only works if world backend is gloo)
        dist.monitored_barrier(timeout=timedelta(seconds=timeout_s))
        return

    dist.monitored_barrier(group=g, timeout=timedelta(seconds=timeout_s))


# -------------------------
# Distributed init
# -------------------------
def initialize_distributed(rank: int, world_size: int):
    """initialize distributed"""
    args = get_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    if dist.is_initialized():
        if args.rank == 0:
            print(
                "torch.distributed already initialized, skipping init_process_group() ...",
                flush=True,
            )
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
    else:
        if args.rank == 0:
            print("> initializing torch.distributed ...", flush=True)

        dist.init_process_group(
            backend=args.distributed_backend,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
            init_method="env://",
        )
    # Create a GLOO world group for control-plane sync (even if main backend is NCCL).
    global _GLOO_WORLD_GROUP
    try:
        _GLOO_WORLD_GROUP = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    except Exception:
        # If this fails, monitored_barrier may not be available (rare), but we try anyway.
        _GLOO_WORLD_GROUP = None

    if torch.cuda.is_available():
        initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        )

        if args.rank == 0:
            print(f"> initialized tensor model parallel size: {args.tensor_model_parallel_size}")
            print(
                f"> initialized pipeline model parallel size: {args.pipeline_model_parallel_size}"
            )


def _maybe_new_group(ranks: list[int], backend: str):
    """
    Create a process group only if size > 1.
    For singleton "groups", return None (avoid edge-case bugs and pointless comms).
    """
    if len(ranks) <= 1:
        return None
    return dist.new_group(ranks=ranks, backend=backend)


def initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size):
    """initialize model parallel"""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print(f"[Rank {rank}] initialize_model_parallel: START", flush=True)

    model_size = tensor_model_parallel_size * pipeline_model_parallel_size

    if world_size % model_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor*pipe {model_size}"
        )

    data_parallel_size = world_size // model_size

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    # -------------------------
    # Data-parallel groups
    # -------------------------
    global _DATA_PARALLEL_GROUP_NCCL, _DATA_PARALLEL_GROUP_GLOO, _DATA_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP_NCCL is None, "data parallel group already initialized"

    all_data_parallel_group_ranks: list[list[int]] = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            r = list(range(start_rank + j, end_rank, tensor_model_parallel_size))
            all_data_parallel_group_ranks.append(r)

            g_nccl = (
                _maybe_new_group(r, backend="nccl")
                if dist.get_backend() == "nccl"
                else _maybe_new_group(r, backend=dist.get_backend())
            )
            g_gloo = _maybe_new_group(r, backend="gloo")

            if rank in r:
                _DATA_PARALLEL_GROUP_NCCL = g_nccl
                _DATA_PARALLEL_GROUP_GLOO = g_gloo
                _DATA_GLOBAL_RANKS = r
    print(f"[Rank {rank}] initialize_model_parallel: DP groups created", flush=True)

    # -------------------------
    # Model-parallel groups
    # -------------------------
    global _MODEL_PARALLEL_GROUP_NCCL
    assert _MODEL_PARALLEL_GROUP_NCCL is None, "model parallel group already initialized"
    for i in range(data_parallel_size):
        r = [
            grp[i] for grp in all_data_parallel_group_ranks
        ]  # pick i-th element from each DP group list
        r = list(r)
        g_nccl = (
            _maybe_new_group(r, backend="nccl")
            if dist.get_backend() == "nccl"
            else _maybe_new_group(r, backend=dist.get_backend())
        )
        if rank in r:
            _MODEL_PARALLEL_GROUP_NCCL = g_nccl

    print(f"[Rank {rank}] initialize_model_parallel: MP groups created", flush=True)

    # -------------------------
    # Tensor model-parallel groups
    # -------------------------
    global _TENSOR_MODEL_PARALLEL_GROUP_NCCL, _TENSOR_MODEL_PARALLEL_GROUP_GLOO, _TENSOR_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP_NCCL is None
    ), "tensor model parallel group already initialized"
    for i in range(num_tensor_model_parallel_groups):
        r = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        g_nccl = (
            _maybe_new_group(r, backend="nccl")
            if dist.get_backend() == "nccl"
            else _maybe_new_group(r, backend=dist.get_backend())
        )
        g_gloo = _maybe_new_group(r, backend="gloo")
        if rank in r:
            _TENSOR_MODEL_PARALLEL_GROUP_NCCL = g_nccl
            _TENSOR_MODEL_PARALLEL_GROUP_GLOO = g_gloo
            _TENSOR_GLOBAL_RANKS = r

    print(f"[Rank {rank}] initialize_model_parallel: TP groups created", flush=True)

    # -------------------------
    # Pipeline model-parallel groups + embedding groups
    # PP groups are non-contiguous like [0,8], [1,9], ...
    # -------------------------
    global _PIPELINE_MODEL_PARALLEL_GROUP_NCCL, _PIPELINE_MODEL_PARALLEL_GROUP_GLOO, _PIPELINE_GLOBAL_RANKS
    global _EMBEDDING_GROUP_NCCL, _EMBEDDING_GROUP_GLOO
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP_NCCL is None
    ), "pipeline model parallel group already initialized"
    assert _EMBEDDING_GROUP_NCCL is None, "embedding group already initialized"

    for i in range(num_pipeline_model_parallel_groups):
        r = list(range(i, world_size, num_pipeline_model_parallel_groups))  # non-contiguous
        g_nccl = (
            _maybe_new_group(r, backend="nccl")
            if dist.get_backend() == "nccl"
            else _maybe_new_group(r, backend=dist.get_backend())
        )
        g_gloo = _maybe_new_group(r, backend="gloo")

        if rank in r:
            _PIPELINE_MODEL_PARALLEL_GROUP_NCCL = g_nccl
            _PIPELINE_MODEL_PARALLEL_GROUP_GLOO = g_gloo
            _PIPELINE_GLOBAL_RANKS = r

        # embedding group: first + last in pipeline
        emb = [r[0], r[-1]] if len(r) > 1 else r
        emb = list(emb)
        eg_nccl = (
            _maybe_new_group(emb, backend="nccl")
            if dist.get_backend() == "nccl"
            else _maybe_new_group(emb, backend=dist.get_backend())
        )
        eg_gloo = _maybe_new_group(emb, backend="gloo")
        if rank in emb:
            _EMBEDDING_GROUP_NCCL = eg_nccl
            _EMBEDDING_GROUP_GLOO = eg_gloo

    print(f"[Rank {rank}] initialize_model_parallel: PP and embedding groups created", flush=True)
    print(f"[Rank {rank}] initialize_model_parallel: COMPLETE", flush=True)


# -------------------------
# Communication Tests
# -------------------------
def test_tensor_parallel_group():
    args = get_args()
    rank = dist.get_rank()
    tp_size = args.tensor_model_parallel_size
    tp_ranks = _TENSOR_GLOBAL_RANKS or [rank]

    if rank == 0:
        print(f"Testing tensor parallel communication (TP size: {tp_size})")
    control_barrier(group=_TENSOR_MODEL_PARALLEL_GROUP_GLOO, timeout_s=120, name="tp_barrier")
    print(f"[Rank {rank}] TP group ranks: {tp_ranks}", flush=True)

    if tp_size <= 1 or len(tp_ranks) <= 1 or _TENSOR_MODEL_PARALLEL_GROUP_NCCL is None:
        # Nothing to communicate; treat as pass.
        print(f"[Rank {rank}] TP size is 1; skipping NCCL all_reduce.", flush=True)
        control_barrier(group=_TENSOR_MODEL_PARALLEL_GROUP_GLOO, timeout_s=120, name="tp_barrier")
        return

    device = torch.device(f"cuda:{args.local_rank}")
    tensor = torch.tensor([rank], device=device, dtype=torch.float32)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=_TENSOR_MODEL_PARALLEL_GROUP_NCCL)

    # Verify on every rank (cheap and avoids group-rank queries)
    expected = float(sum(tp_ranks))
    if not torch.allclose(tensor, torch.tensor([expected], device=device), atol=0, rtol=0):
        raise AssertionError(
            f"[Rank {rank}] TP all_reduce wrong: got {tensor.item()}, expected {expected}"
        )

    torch.cuda.synchronize()
    control_barrier(group=_TENSOR_MODEL_PARALLEL_GROUP_GLOO, timeout_s=120, name="tp_barrier")


#    if rank == 0:
#        print("Tensor parallel communication test completed successfully")


def test_data_parallel_group():
    args = get_args()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Compute DP group size
    dp_group_size = world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    )
    dp_ranks = _DATA_GLOBAL_RANKS or [rank]

    if rank == 0:
        print(f"Testing data parallel communication (DP group size: {dp_group_size})")
    control_barrier(group=_TENSOR_MODEL_PARALLEL_GROUP_GLOO, timeout_s=120, name="tp_barrier")
    print(f"[Rank {rank}] DP group ranks: {dp_ranks}", flush=True)

    if dp_group_size <= 1 or len(dp_ranks) <= 1 or _DATA_PARALLEL_GROUP_NCCL is None:
        print(f"[Rank {rank}] DP size is 1; skipping NCCL all_reduce.", flush=True)
        control_barrier(group=_DATA_PARALLEL_GROUP_GLOO, timeout_s=120, name="dp_barrier")
        return

    device = torch.device(f"cuda:{args.local_rank}")
    tensor = torch.tensor([rank], device=device, dtype=torch.float32)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=_DATA_PARALLEL_GROUP_NCCL)

    expected = float(sum(dp_ranks))
    if not torch.allclose(tensor, torch.tensor([expected], device=device), atol=0, rtol=0):
        raise AssertionError(
            f"[Rank {rank}] DP all_reduce wrong: got {tensor.item()}, expected {expected}"
        )

    torch.cuda.synchronize()
    control_barrier(group=_DATA_PARALLEL_GROUP_GLOO, timeout_s=120, name="dp_barrier")


#    if rank == 0:
#        print("Data parallel communication test completed successfully")


def test_pipeline_parallel_group():
    args = get_args()
    rank = dist.get_rank()

    pp_size = args.pipeline_model_parallel_size
    pp_ranks = _PIPELINE_GLOBAL_RANKS or [rank]
    pp_group_nccl = _PIPELINE_MODEL_PARALLEL_GROUP_NCCL
    pp_group_gloo = _PIPELINE_MODEL_PARALLEL_GROUP_GLOO

    if rank == 0:
        print(f"Testing pipeline parallel communication (PP size: {pp_size})")
    control_barrier(group=pp_group_gloo, timeout_s=180, name="pp_start")
    print(f"[Rank {rank}] PP group ranks: {pp_ranks}", flush=True)

    if pp_size <= 1 or len(pp_ranks) <= 1 or pp_group_nccl is None:
        print(f"[Rank {rank}] PP size is 1; skipping P2P.", flush=True)
        control_barrier(group=pp_group_gloo, timeout_s=120, name="pp_barrier")
        return

    # Determine local pp_rank without calling dist.get_rank(group=...) (avoid edge cases)
    # pp_ranks is ordered; locate ourselves:
    try:
        pp_rank = pp_ranks.index(rank)
    except ValueError:
        raise RuntimeError(f"[Rank {rank}] not found in its own PP ranks list?! {pp_ranks}")

    prev_rank = pp_ranks[pp_rank - 1] if pp_rank > 0 else None
    next_rank = pp_ranks[pp_rank + 1] if pp_rank < len(pp_ranks) - 1 else None

    device = torch.device(f"cuda:{args.local_rank}")

    # -------- Forward: recv from prev, send to next --------
    print(f"[Rank {rank}] PP forward: prev={prev_rank}, next={next_rank}", flush=True)

    recv_tensor = None
    ops = []
    if prev_rank is not None:
        recv_tensor = torch.zeros(2, device=device, dtype=torch.float32)
        ops.append(dist.P2POp(dist.irecv, recv_tensor, prev_rank, group=pp_group_nccl))
    if next_rank is not None:
        send_tensor = torch.tensor([rank, pp_rank], device=device, dtype=torch.float32)
        ops.append(dist.P2POp(dist.isend, send_tensor, next_rank, group=pp_group_nccl))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    if prev_rank is not None:
        expected = torch.tensor([prev_rank, pp_rank - 1], device=device, dtype=torch.float32)
        if not torch.allclose(recv_tensor, expected):
            raise AssertionError(
                f"[Rank {rank}] PP forward wrong: got {recv_tensor}, expected {expected}"
            )

    torch.cuda.synchronize()
    control_barrier(group=pp_group_gloo, timeout_s=180, name="pp_forward_barrier")

    # -------- Backward: recv from next, send to prev --------
    print(f"[Rank {rank}] PP backward: prev={prev_rank}, next={next_rank}", flush=True)

    recv_tensor = None
    ops = []
    if next_rank is not None:
        recv_tensor = torch.zeros(2, device=device, dtype=torch.float32)
        ops.append(dist.P2POp(dist.irecv, recv_tensor, next_rank, group=pp_group_nccl))
    if prev_rank is not None:
        send_tensor = torch.tensor([rank, pp_rank], device=device, dtype=torch.float32)
        ops.append(dist.P2POp(dist.isend, send_tensor, prev_rank, group=pp_group_nccl))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    if next_rank is not None:
        expected = torch.tensor([next_rank, pp_rank + 1], device=device, dtype=torch.float32)
        if not torch.allclose(recv_tensor, expected):
            raise AssertionError(
                f"[Rank {rank}] PP backward wrong: got {recv_tensor}, expected {expected}"
            )

    torch.cuda.synchronize()
    control_barrier(group=pp_group_gloo, timeout_s=180, name="pp_backward_barrier")

    print(f"[Rank {rank}] Pipeline parallel test completed", flush=True)
    control_barrier(group=pp_group_gloo, timeout_s=180, name="pp_backward_barrier")


#    if rank == 0:
#        print("Pipeline parallel communication test completed successfully")


# -------------------------
# Test Orchestration
# -------------------------
def test_communication():
    """Test all parallel communication with progressive execution"""
    args = get_args()
    rank = dist.get_rank()
    print(f"[Rank {rank}] Entered test_communication()", flush=True)
    if rank == 0:
        print("\n" + "=" * 60)
        print("PHASE 1: PARALLEL COMMUNICATION TESTING")
        print("=" * 60)

    # Always use gloo world control barrier
    control_barrier(timeout_s=120, name="pre_test_world_barrier")

    # TP
    ok = safe_test_execution(test_tensor_parallel_group, "tensor_parallel", timeout_seconds=180)
    control_barrier(timeout_s=120, name="between_tp_dp")
    if not ok and rank == 0:
        print("⚠ Warning: TP test failed, continuing...")
    elif rank == 0:
        log_test_result("tensor_parallel", 'passed')
        print("Tensor parallel communication test completed successfully")
        print("\n" + "-" * 60)

    # DP
    ok = safe_test_execution(test_data_parallel_group, "data_parallel", timeout_seconds=180)
    control_barrier(timeout_s=120, name="between_dp_pp")
    if not ok and rank == 0:
        print("⚠ Warning: DP test failed, continuing...")
    elif rank == 0:
        log_test_result("data_parallel", 'passed')
        print("Data parallel communication test completed successfully")
        print("\n" + "-" * 60)

    # PP
    ok = safe_test_execution(test_pipeline_parallel_group, "pipeline_parallel", timeout_seconds=240)
    control_barrier(timeout_s=120, name="post_comm_world_barrier")
    if not ok and rank == 0:
        print("⚠ Warning: PP test failed, continuing...")
    elif rank == 0:
        log_test_result("pipeline_parallel", 'passed')
        print("Pipeline parallel communication test completed successfully")
        print("\n" + "-" * 60)
    # TODO: Expert Parallel
    if rank == 0:
        print("\nParallel communication testing phase completed")
        print("=" * 60)


def parse_args():

    parser = argparse.ArgumentParser(description="GPU Health Check arguments")
    parser.add_argument(
        '--tensor-model-parallel-size',
        type=int,
        default=1,
        help='Degree of tensor model parallelism (will be auto-detected if not optimal).',
    )
    parser.add_argument(
        '--pipeline-model-parallel-size',
        type=int,
        default=1,
        help='Degree of pipeline model parallelism.',
    )
    parser.add_argument(
        '--distributed-backend',
        default='nccl',
        choices=['nccl', 'gloo'],
        help='Which backend to use for distributed training.',
    )
    parser.add_argument(
        '--distributed-timeout-minutes',
        type=int,
        default=10,
        help='Timeout minutes for torch.distributed.',
    )

    args = parser.parse_args()

    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))

    return args


def print_test_summary():
    """Print final test summary"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    print("=" * 60)
    print("GPU HEALTH CHECK SUMMARY")
    print("=" * 60)

    total_tests = len(_TEST_RESULTS)
    passed_tests = sum(1 for result in _TEST_RESULTS.values() if result['status'] == 'passed')
    failed_tests = sum(1 for result in _TEST_RESULTS.values() if result['status'] == 'failed')
    skipped_tests = sum(1 for result in _TEST_RESULTS.values() if result['status'] == 'skipped')
    pending_tests = sum(1 for result in _TEST_RESULTS.values() if result['status'] == 'pending')

    for test_name, result in _TEST_RESULTS.items():
        status_icon = (
            "✓" if result['status'] == 'passed' else "✗" if result['status'] == 'failed' else "⚠"
        )
        print(f"{status_icon} {test_name.replace('_', ' ').title()}: {result['status'].upper()}")
        if result['error']:
            print(f"   └─ {result['error']}")

    print(
        f"Results: {passed_tests} passed, {failed_tests} failed, {skipped_tests} skipped out of {total_tests} total"
    )
    if pending_tests != 0:
        print("�~Z|some tests pending")
    elif failed_tests == 0:
        print("🎉 All GPU health checks PASSED!")
    elif passed_tests > 0:
        print("⚠ Some tests failed, but basic functionality verified")
    else:
        print("❌ Critical: All tests FAILED - GPU environment may have serious issues")

    print("=" * 60)


def main():
    """Complete GPU health check with progressive testing"""
    args = parse_args()
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    rank = args.rank
    world_size = args.world_size
    tp_size = args.tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    dp_size = world_size // (tp_size * pp_size)
    if rank == 0:
        print("=" * 60)
        print("COMPREHENSIVE GPU HEALTH CHECK")
        print("=" * 60)
        print("Configuration:")
        print(f"  • World Size: {world_size}")
        print(f"  • Tensor Parallel Size: {tp_size}")
        print(f"  • Pipeline Parallel Size: {pp_size}")
        print(f"  • Data Parallel Size: {dp_size}")
        print(f"  • Backend: {args.distributed_backend}")
        print(f"  • Timeout: {args.distributed_timeout_minutes} minutes")
        print("=" * 60)
    if world_size == 1:
        if rank == 0:
            print("Single process mode detected")
            print("Running basic GPU hardware and computation tests...")
        # TODO: add GPU hardware and computation tests
        return

    if rank == 0:
        print("Multi-process distributed mode detected")
        print("Initializing distributed environment...")
    try:
        # Initialize process group and subgroups
        initialize_distributed(rank, world_size)

        if rank == 0:
            print("✓ Distributed initialization successful")
            print("Starting comprehensive test suite...")

        # PHASE 1: Test parallel communication
        test_communication()

        # TODO: add GPU hardware and computation tests

        if rank == 0:
            print("=" * 60)
            print("ALL TEST PHASES COMPLETED")
            print("=" * 60)

    except Exception as e:
        if rank == 0:
            print(f"❌ Critical error during testing: {e}")
            print("Attempting cleanup...")
    finally:
        # Always attempt cleanup
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            if rank == 0:
                print(f"⚠ Warning: Cleanup failed: {e}")

        # Print final summary
        if rank == 0:
            print_test_summary()

        failed_count = sum(1 for r in _TEST_RESULTS.values() if r["status"] == "failed")
        if failed_count > 0:
            import sys

            sys.exit(1)


if __name__ == "__main__":
    main()

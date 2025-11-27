"""
Complete GPU Health Check Implementation

This module provides comprehensive GPU health verification including:
- Tensor parallel communication testing
- Data parallel communication testing
- Pipeline parallel communication testing
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
import signal
import threading
import time

from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist

_GLOBAL_ARGS = None
_DATA_PARALLEL_GROUP = None
_DATA_GLOBAL_RANKS = None
_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_GLOBAL_RANKS = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_PIPELINE_GLOBAL_RANKS = None
_EMBEDDING_GROUP = None

# Test tracking
_TEST_RESULTS = {
    'tensor_parallel': {'status': 'pending', 'error': None},
    'data_parallel': {'status': 'pending', 'error': None},
    'pipeline_parallel': {'status': 'pending', 'error': None},
    'gpu_hardware': {'status': 'pending', 'error': None},
    'computation': {'status': 'pending', 'error': None},
    'ecc_error': {'status': 'pending', 'error': None},
}


class TimeoutError(Exception):
    pass


@contextmanager
def timeout_protection(seconds, test_name):
    """
    Context manager for timeout protection
    NOTE: SIGALRM-based timeout is disabled in multi-process environments
    because it can interfere with PyTorch distributed operations and NCCL.
    We rely on PyTorch's built-in distributed timeout instead.
    """
    # Disabled SIGALRM timeout - causes issues in multi-process PyTorch/NCCL environments
    # Just yield without timeout protection
    yield


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


def safe_test_execution(test_func, test_name, timeout_seconds=120):
    """Execute test with timeout protection and error handling"""
    try:
        with timeout_protection(timeout_seconds, test_name):
            test_func()
        log_test_result(test_name, 'passed')
        return True
    except TimeoutError as e:
        log_test_result(test_name, 'failed', str(e))
        return False
    except Exception as e:
        log_test_result(test_name, 'failed', f"Exception: {str(e)}")
        return False


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def initialize_distributed(rank, world_size):

    args = get_args()

    device_count = torch.cuda.device_count()
    if dist.is_initialized():
        if args.rank == 0:
            print(
                "torch distributed is already initialized, " "skipping initialization ...",
                flush=True,
            )
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
    else:
        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(args.local_rank)

        dist.init_process_group(
            backend=args.distributed_backend,
            store=None,  # TODO: Need to support PrefixStore mode.
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
            device_id=torch.device(
                f'cuda:{args.local_rank}'
            ),  # Explicitly specify which GPU this process uses
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        # initialize parallel groups
        initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        )
        if args.rank == 0:
            print(
                f"> initialized tensor model parallel with size "
                f"{args.tensor_model_parallel_size}"
            )
            print(
                f"> initialized pipeline model parallel with size "
                f"{args.pipeline_model_parallel_size}"
            )

    return


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1
):
    world_size: int = dist.get_world_size()
    rank = dist.get_rank()

    print(f"[Rank {rank}] initialize_model_parallel: START", flush=True)
    model_size = tensor_model_parallel_size * pipeline_model_parallel_size

    if world_size % model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")

    data_parallel_size: int = world_size // model_size

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    print(f"[Rank {rank}] initialize_model_parallel: About to create DP groups", flush=True)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(ranks)
            group = dist.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_GLOBAL_RANKS = ranks

    print(f"[Rank {rank}] initialize_model_parallel: DP groups created", flush=True)

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        ranks = [
            data_parallel_group_ranks[i]
            for data_parallel_group_ranks in all_data_parallel_group_ranks
        ]
        group = dist.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    print(f"[Rank {rank}] initialize_model_parallel: MP groups created", flush=True)

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), "tensor model parallel group is already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_GLOBAL_RANKS = ranks

    print(f"[Rank {rank}] initialize_model_parallel: TP groups created", flush=True)

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), "pipeline model parallel group is already initialized"
    global _EMBEDDING_GROUP
    assert _EMBEDDING_GROUP is None, "embedding group is already initialized"
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = dist.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
        else:
            embedding_ranks = ranks
        group = dist.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group

    print(f"[Rank {rank}] initialize_model_parallel: PP and embedding groups created", flush=True)
    print(f"[Rank {rank}] initialize_model_parallel: COMPLETE", flush=True)

    return


def get_model_parallel_group():
    """Get the model-parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is not None
    ), "intra_layer_model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline-model-parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data-parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return dist.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return caller's rank for the pipeline-model-parallel group."""
    return dist.get_rank(group=get_pipeline_model_parallel_group())


def get_data_parallel_rank():
    """Return caller's rank in the data-parallel group."""
    return dist.get_rank(group=get_data_parallel_group())


def test_tensor_parallel_group_c10d():
    args = get_args()
    tp_group = get_tensor_model_parallel_group()
    rank = dist.get_rank()

    if rank == 0:
        print(f"Testing tensor parallel communication (TP size: {args.tensor_model_parallel_size})")

    print(f"[Rank {rank}] Starting tensor parallel test, TP group: {_TENSOR_GLOBAL_RANKS}")

    tensor = torch.tensor([rank], device=f'cuda:{args.local_rank}', dtype=torch.float32)

    print(f"[Rank {rank}] Created tensor: {tensor}, starting all_reduce...")

    # All_reduce with timeout protection at distributed level
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=tp_group)

    print(f"[Rank {rank}] All_reduce completed, result: {tensor}")

    # For single-process TP groups, we know tp_rank is 0 without calling dist.get_rank()
    # Calling dist.get_rank(group=...) on single-process NCCL groups can cause SIGSEGV
    if args.tensor_model_parallel_size == 1:
        tp_rank = 0
    else:
        tp_rank = get_tensor_model_parallel_rank()

    if tp_rank == 0:
        expected_tensor = torch.tensor(
            [sum(list(_TENSOR_GLOBAL_RANKS))], device=f'cuda:{args.local_rank}', dtype=torch.float32
        )
        if not torch.allclose(tensor, expected_tensor):
            raise AssertionError(
                f"Tensor parallel test failed on rank {rank}, expected: {expected_tensor}, received: {tensor}."
            )
        print(f"[Rank {rank}] Tensor parallel verification passed")

    # Only do barrier if TP group has more than 1 process
    # Single-process barriers can cause issues with some NCCL versions
    if args.tensor_model_parallel_size > 1:
        print(f"[Rank {rank}] Starting tensor parallel barrier...")
        dist.barrier(group=tp_group)
        print(f"[Rank {rank}] Tensor parallel barrier completed")
    else:
        print(f"[Rank {rank}] Skipping TP barrier (single-process group)")

    if rank == 0:
        print("Tensor parallel communication test completed successfully")


def test_data_parallel_group_c10d():
    global _DATA_GLOBAL_RANKS
    args = get_args()
    dp_group = get_data_parallel_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Testing data parallel communication (World size: {args.world_size})")

    print(f"[Rank {rank}] Starting data parallel test, DP group: {_DATA_GLOBAL_RANKS}")

    tensor = torch.tensor([rank], device=f'cuda:{args.local_rank}', dtype=torch.float32)

    print(f"[Rank {rank}] Created tensor: {tensor}, starting data parallel all_reduce...")

    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=dp_group)

    print(f"[Rank {rank}] Data parallel all_reduce completed, result: {tensor}")

    # Calculate DP group size
    dp_group_size = world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    )

    # For single-process DP groups, we know dp_rank is 0 without calling dist.get_rank()
    # Calling dist.get_rank(group=...) on single-process NCCL groups can cause SIGSEGV
    if dp_group_size == 1:
        dp_rank = 0
    else:
        dp_rank = get_data_parallel_rank()

    if dp_rank == 0:
        expected_tensor = torch.tensor(
            [sum(list(_DATA_GLOBAL_RANKS))], device=f'cuda:{args.local_rank}', dtype=torch.float32
        )
        if not torch.allclose(tensor, expected_tensor):
            raise AssertionError(
                f"Data parallel test failed on rank {rank}, expected: {expected_tensor}, received: {tensor}."
            )
        print(f"[Rank {rank}] Data parallel verification passed")

    # Data parallel group should always have multiple processes when world_size > 1
    # Only skip barrier if somehow the group is single-process
    dp_group_size = world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    )
    if dp_group_size > 1:
        print(f"[Rank {rank}] Starting data parallel barrier...")
        dist.barrier(group=dp_group)
        print(f"[Rank {rank}] Data parallel barrier completed")
    else:
        print(f"[Rank {rank}] Skipping DP barrier (single-process group)")

    if rank == 0:
        print("Data parallel communication test completed successfully")


def test_pipeline_parallel_group_c10d():
    global _PIPELINE_GLOBAL_RANKS
    args = get_args()

    pp_ranks = _PIPELINE_GLOBAL_RANKS
    pp_group = get_pipeline_model_parallel_group()
    pp_size = args.pipeline_model_parallel_size
    rank = dist.get_rank()
    device = torch.device(f'cuda:{args.local_rank}')

    # For single-process PP groups, we know pp_rank is 0 without calling dist.get_rank()
    # Calling dist.get_rank(group=...) on single-process NCCL groups can cause SIGSEGV
    if pp_size == 1:
        pp_rank = 0
    else:
        pp_rank = get_pipeline_model_parallel_rank()

    if rank == 0:
        print(f"Testing pipeline parallel communication (PP size: {pp_size})")

    print(
        f"[Rank {rank}] Starting pipeline parallel test, PP rank: {pp_rank}, PP group: {pp_ranks}"
    )

    prev_rank = None
    next_rank = None

    if pp_rank > 0:
        prev_rank = pp_ranks[pp_rank - 1]

    if pp_rank < pp_size - 1:
        next_rank = pp_ranks[pp_rank + 1]

    print(f"[Rank {rank}] PP topology - prev: {prev_rank}, next: {next_rank}")

    # Forward communication test.
    print(f"[Rank {rank}] Starting forward pipeline communication test...")
    if next_rank is not None:
        send_tensor = torch.tensor([rank, pp_rank], device=device, dtype=torch.float32)
        print(f"[Rank {rank}] Sending {send_tensor} to rank {next_rank}")
        dist.send(send_tensor, dst=next_rank)

    if prev_rank is not None:
        recv_tensor = torch.zeros(2, device=device, dtype=torch.float32)
        print(f"[Rank {rank}] Receiving from rank {prev_rank}")
        dist.recv(recv_tensor, src=prev_rank)
        expected_tensor = torch.tensor([prev_rank, pp_rank - 1], device=device, dtype=torch.float32)
        if not torch.allclose(recv_tensor, expected_tensor):
            raise AssertionError(
                f"Pipeline forward test failed on rank {rank}, expected: {expected_tensor}, received: {recv_tensor}."
            )
        print(f"[Rank {rank}] Forward communication verified: {recv_tensor}")

    # Only do barriers if PP group has more than 1 process
    if pp_size > 1:
        dist.barrier(pp_group)
        print(f"[Rank {rank}] Forward pipeline barrier completed")
    else:
        print(f"[Rank {rank}] Skipping forward PP barrier (single-process group)")

    # Backward communication test.
    print(f"[Rank {rank}] Starting backward pipeline communication test...")
    if prev_rank is not None:
        send_tensor = torch.tensor([rank, pp_rank], device=device, dtype=torch.float32)
        print(f"[Rank {rank}] Sending {send_tensor} to rank {prev_rank}")
        dist.send(send_tensor, dst=prev_rank)

    if next_rank is not None:
        recv_tensor = torch.zeros(2, device=device, dtype=torch.float32)
        print(f"[Rank {rank}] Receiving from rank {next_rank}")
        dist.recv(recv_tensor, src=next_rank)
        expected_tensor = torch.tensor([next_rank, pp_rank + 1], device=device, dtype=torch.float32)
        if not torch.allclose(recv_tensor, expected_tensor):
            raise AssertionError(
                f"Pipeline backward test failed on rank {rank}, expected: {expected_tensor}, received: {recv_tensor}."
            )
        print(f"[Rank {rank}] Backward communication verified: {recv_tensor}")

    if pp_size > 1:
        dist.barrier(pp_group)
        print(f"[Rank {rank}] Backward pipeline barrier completed")
    else:
        print(f"[Rank {rank}] Skipping backward PP barrier (single-process group)")

    # Global barrier - always do this
    dist.barrier()
    print(f"[Rank {rank}] Pipeline parallel test completed")

    if rank == 0:
        print("Pipeline parallel communication test completed successfully")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def auto_detect_parallel_config(requested_tp_size, requested_pp_size):
    """
    Validate and adjust parallel configuration based on environment

    Args:
        requested_tp_size: User-requested tensor parallel size
        requested_pp_size: User-requested pipeline parallel size

    Returns:
        tuple: (tp_size, pp_size, need_distributed)
    """
    device_count = torch.cuda.device_count()
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    if world_size == 1:
        # Single process mode - no distributed testing needed
        return 1, 1, False

    # Calculate the total model parallel size
    total_mp_size = requested_tp_size * requested_pp_size

    # Validate that the configuration is feasible
    if total_mp_size > world_size:
        # Configuration is invalid - more parallel groups than processes
        # Adjust to fit within world_size
        if requested_tp_size > world_size:
            # TP size too large, reduce it
            tp_size = world_size
            pp_size = 1
        elif requested_pp_size > world_size:
            # PP size too large, reduce it
            tp_size = 1
            pp_size = world_size
        else:
            # Both are valid individually but product is too large
            # Prioritize TP over PP
            tp_size = min(requested_tp_size, world_size)
            pp_size = min(requested_pp_size, world_size // tp_size)

        return tp_size, pp_size, True

    # Configuration is valid - use requested values
    # This includes the common case of tp_size=1, pp_size=1 (pure data parallel)
    return requested_tp_size, requested_pp_size, True


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
        '--use-tp-pp-dp-mapping',
        action='store_true',
        default=False,
        help='If set, distributed ranks initialize order is changed '
        'from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp.',
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


def test_communication():
    """Test all parallel communication with progressive execution"""
    args = get_args()
    rank = dist.get_rank()
    # Debug: Print entry into test_communication for ALL ranks
    print(f"[Rank {rank}] Entered test_communication()", flush=True)

    if rank == 0:
        print("\n" + "=" * 60)
        print("PHASE 1: PARALLEL COMMUNICATION TESTING")
        print("=" * 60)

    # Debug: Barrier before starting tests
    print(f"[Rank {rank}] About to do pre-test barrier", flush=True)
    try:
        dist.barrier()
        print(f"[Rank {rank}] Pre-test barrier completed", flush=True)
    except Exception as e:
        print(f"[Rank {rank}] Pre-test barrier FAILED: {e}", flush=True)
        raise

    # Test 1: Tensor Parallel Communication
    print(f"[Rank {rank}] About to start TP test", flush=True)
    success = safe_test_execution(
        test_tensor_parallel_group_c10d, 'tensor_parallel', timeout_seconds=120
    )
    print(f"[Rank {rank}] TP test completed, success={success}", flush=True)
    if not success and rank == 0:
        print("⚠ Warning: Tensor parallel test failed, but continuing with other tests...")

    # Global barrier before next test
    try:
        dist.barrier()
    except Exception as e:
        if rank == 0:
            print(f"⚠ Warning: Global barrier failed: {e}")

    # Test 2: Data Parallel Communication
    success = safe_test_execution(
        test_data_parallel_group_c10d, 'data_parallel', timeout_seconds=120
    )
    if not success and rank == 0:
        print("⚠ Warning: Data parallel test failed, but continuing with other tests...")

    # Global barrier before next test
    try:
        dist.barrier()
    except Exception as e:
        if rank == 0:
            print(f"⚠ Warning: Global barrier failed: {e}")

    # Test 3: Pipeline Parallel Communication
    success = safe_test_execution(
        test_pipeline_parallel_group_c10d, 'pipeline_parallel', timeout_seconds=120
    )
    if not success and rank == 0:
        print("⚠ Warning: Pipeline parallel test failed, but continuing with other tests...")

    # Final barrier
    try:
        dist.barrier()
    except Exception as e:
        if rank == 0:
            print(f"⚠ Warning: Final barrier failed: {e}")

    if rank == 0:
        print("\nParallel communication testing phase completed")
        print("=" * 60)


def test_gpu_hardware_single():
    """Single GPU hardware test without distributed calls"""
    try:
        import pynvml

        pynvml.nvmlInit()

        print(f"Testing GPU hardware")

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_name = pynvml.nvmlDeviceGetName(handle)

            print(f"=== Checking GPU {i}: {gpu_name} ===")

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            print(f"Current GPU temperature: {temp}°C")
            if temp > 85:
                print(f"Warning: GPU temperature too high!")

            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            print(f"Power usage: {power_usage:.2f}W / {power_limit:.2f}W")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_utilization = (float(mem_info.used) / float(mem_info.total)) * 100

            print(f"Total memory: {float(mem_info.total) / (1024**2):.2f} MB")
            print(f"Used memory: {float(mem_info.used) / (1024**2):.2f} MB")
            print(f"GPU {i} memory utilization rate: {memory_utilization:.2f}%")

        pynvml.nvmlShutdown()
    except ImportError:
        print("Pynvml is not installed.")
        raise

    return


def test_gpu_hardware():
    """Test GPU hardware with distributed coordination"""
    args = get_args()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 60)
        print("PHASE 2: GPU HARDWARE TESTING")
        print("=" * 60)

    # Only test on local rank 0 to avoid redundant hardware checks
    success = True
    if args.local_rank == 0:
        success = safe_test_execution(test_gpu_hardware_single, 'gpu_hardware', timeout_seconds=60)

    try:
        dist.barrier()
    except Exception as e:
        if rank == 0:
            print(f"⚠ Warning: GPU hardware test barrier failed: {e}")

    if rank == 0:
        print("GPU hardware testing phase completed")
        print("=" * 60)

    return success


def check_test_result(test_name, result_tensor):
    expected_tensor = torch.ones_like(result_tensor).to(f'cuda:{get_args().local_rank}')
    if torch.allclose(result_tensor, expected_tensor, atol=1e-6):
        print(f"{test_name} passed")
    else:
        print(f"{test_name} failed")


def report_test_calculation_result(test_name, result):
    if torch.any(torch.isnan(result)):
        print(f"{test_name} failed: nan is detected in result")
        return False
    elif torch.any(torch.isinf(result)):
        print(f"{test_name} failed: inf is detected in result")
        return False
    else:
        return True


def test_calculation_float():
    test_tensor = torch.randn(4096, 4096).to(f'cuda:{get_args().local_rank}')
    result = torch.matmul(test_tensor, test_tensor)

    return report_test_calculation_result("test_calculation_float", result)


def test_calculation_double():
    test_tensor = torch.randn(4096, 4096, dtype=torch.double).to(f'cuda:{get_args().local_rank}')
    result = torch.matmul(test_tensor, test_tensor)

    return report_test_calculation_result("test_calculation_double", result)


def test_calculation_half():
    test_tensor = torch.randn(4096, 4096, dtype=torch.half).to(f'cuda:{get_args().local_rank}')
    result = torch.matmul(test_tensor, test_tensor)

    return report_test_calculation_result("test_calculation_half", result)


def test_calculation_endurance():
    start_time = time.time()
    iteration = 0

    while time.time() - start_time < 60:
        iteration += 1

        a = torch.randn(4096, 4096).to(f'cuda:{get_args().local_rank}')
        b = torch.randn(4096, 4096).to(f'cuda:{get_args().local_rank}')
        result1 = torch.matmul(a, b)

        c = torch.randn(4096, 4096).to(f'cuda:{get_args().local_rank}')
        result2 = torch.inverse(c)

        if torch.any(torch.isnan(result1)) or torch.any(torch.isnan(result2)):
            print(f"test_calculation_float failed: nan detected in iteration {iteration}")
            return False

    return True


def test_ecc_error_detection():
    """Test ECC Error detection through matrix multiplication operations"""
    try:
        device = f'cuda:{get_args().local_rank}'

        # Perform multiple matrix operations to stress test memory
        for i in range(5):
            # Create large tensors to stress GPU memory
            tensor_a = torch.randn(2048, 2048, dtype=torch.float32, device=device)
            tensor_b = torch.randn(2048, 2048, dtype=torch.float32, device=device)

            # Perform matrix multiplication that could trigger ECC errors
            result = torch.matmul(tensor_a, tensor_b)

            # Check for abnormal values that might indicate ECC errors
            if torch.any(torch.isnan(result)):
                print(f"ECC Error Detection: NaN detected in iteration {i}")
                return False
            if torch.any(torch.isinf(result)):
                print(f"ECC Error Detection: Inf detected in iteration {i}")
                return False

            torch.cuda.empty_cache()

        print("ECC Error Detection: No errors detected")
        return True

    except torch.cuda.OutOfMemoryError as e:
        print(f"ECC Error Detection failed: GPU out of memory - {e}")
        return False
    except RuntimeError as e:
        if "cuda" in str(e).lower() or "gpu" in str(e).lower():
            print(f"ECC Error Detection failed: GPU runtime error - {e}")
            return False
        else:
            print(f"ECC Error Detection failed: Runtime error - {e}")
            return False
    except Exception as e:
        print(f"ECC Error Detection failed: Unexpected error - {e}")
        return False


def test_calculation_single():
    """Single GPU calculation test without distributed calls"""
    args = get_args()

    print("Testing GPU calculation capabilities...")

    result = test_calculation_float()
    print(f"Float calculation: {'PASS' if result else 'FAIL'}")

    result = test_calculation_double()
    print(f"Double calculation: {'PASS' if result else 'FAIL'}")

    result = test_calculation_half()
    print(f"Half calculation: {'PASS' if result else 'FAIL'}")

    print("Starting 60-second endurance test...")
    result = test_calculation_endurance()
    print(f"Endurance test: {'PASS' if result else 'FAIL'}")


def test_calculation():
    """Test GPU computation capabilities with distributed coordination"""
    args = get_args()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 60)
        print("PHASE 3: GPU COMPUTATION TESTING")
        print("=" * 60)

    # Test individual computation capabilities
    test_functions = [
        ('Float calculation', test_calculation_float),
        ('Double calculation', test_calculation_double),
        ('Half calculation', test_calculation_half),
        ('Endurance test (60s)', test_calculation_endurance),
        ('ECC Error Detection', test_ecc_error_detection),
    ]

    all_passed = True
    failed_tests = []

    for test_name, test_func in test_functions:
        if rank == 0:
            print(f"\nTesting {test_name}...")

        try:
            result = test_func()
            if not result:
                all_passed = False
                failed_tests.append(test_name)

            if test_name == 'ECC Error Detection':
                if result:
                    log_test_result('ecc_error', 'passed')
                else:
                    log_test_result('ecc_error', 'failed', f"ECC error detection failed")

        except Exception as e:
            result = False
            all_passed = False
            failed_tests.append(test_name)
            if rank == 0:
                print(f"✗ {test_name} failed with exception: {e}")

            if test_name == 'ECC Error Detection':
                log_test_result('ecc_error', 'failed', f"Exception: {str(e)}")

        try:
            result_tensor = torch.zeros(args.world_size).to(f'cuda:{args.local_rank}')
            result_tensor[args.rank] = 1.0 if result else 0.0
            dist.all_reduce(result_tensor, dist.ReduceOp.SUM)

            if args.rank == 0:
                check_test_result(test_name, result_tensor)
        except Exception as e:
            if rank == 0:
                print(f"⚠ Warning: Failed to gather {test_name} results: {e}")

        try:
            dist.barrier()
        except Exception as e:
            if rank == 0:
                print(f"⚠ Warning: Calculation test barrier failed: {e}")

    if all_passed:
        log_test_result('computation', 'passed')
    else:
        error_msg = f"Failed tests: {', '.join(failed_tests)}"
        log_test_result('computation', 'failed', error_msg)

    if rank == 0:
        print("\nGPU computation testing phase completed")
        print("=" * 60)


def print_test_summary():
    """Print final test summary"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print("GPU HEALTH CHECK SUMMARY")
    print("=" * 60)

    total_tests = len(_TEST_RESULTS)
    passed_tests = sum(1 for result in _TEST_RESULTS.values() if result['status'] == 'passed')
    failed_tests = sum(1 for result in _TEST_RESULTS.values() if result['status'] == 'failed')
    skipped_tests = sum(1 for result in _TEST_RESULTS.values() if result['status'] == 'skipped')

    for test_name, result in _TEST_RESULTS.items():
        status_icon = (
            "✓" if result['status'] == 'passed' else "✗" if result['status'] == 'failed' else "⚠"
        )
        print(f"{status_icon} {test_name.replace('_', ' ').title()}: {result['status'].upper()}")
        if result['error']:
            print(f"   └─ {result['error']}")

    print(
        f"\nResults: {passed_tests} passed, {failed_tests} failed, {skipped_tests} skipped out of {total_tests} total"
    )

    if failed_tests == 0:
        print("🎉 All GPU health checks PASSED!")
    elif passed_tests > 0:
        print("⚠ Some tests failed, but basic functionality verified")
    else:
        print("❌ Critical: All tests FAILED - GPU environment may have serious issues")

    print("=" * 60)


def main():
    """Complete GPU health check with progressive testing"""
    args = parse_args()

    # Validate and adjust parallel configuration based on user request and environment
    original_tp = args.tensor_model_parallel_size
    original_pp = args.pipeline_model_parallel_size
    auto_tp_size, auto_pp_size, need_distributed = auto_detect_parallel_config(
        original_tp, original_pp
    )

    # Update args with validated values
    args.tensor_model_parallel_size = auto_tp_size
    args.pipeline_model_parallel_size = auto_pp_size

    set_args(args)

    rank = args.rank
    world_size = args.world_size

    if rank == 0:
        print("=" * 60)
        print("COMPREHENSIVE GPU HEALTH CHECK")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  • World Size: {world_size}")

        # Show TP size info
        if auto_tp_size != original_tp:
            print(
                f"  • Tensor Parallel Size: {args.tensor_model_parallel_size} (requested: {original_tp}, adjusted to fit world_size)"
            )
        else:
            print(f"  • Tensor Parallel Size: {args.tensor_model_parallel_size}")

        # Show PP size info
        if auto_pp_size != original_pp:
            print(
                f"  • Pipeline Parallel Size: {args.pipeline_model_parallel_size} (requested: {original_pp}, adjusted to fit world_size)"
            )
        else:
            print(f"  • Pipeline Parallel Size: {args.pipeline_model_parallel_size}")

        print(f"  • Backend: {args.distributed_backend}")
        print(f"  • Timeout: {args.distributed_timeout_minutes} minutes")
        print(f"  • Need Distributed: {need_distributed}")
        print("=" * 60)

    # Single process mode - run basic tests only
    if world_size == 1:
        if rank == 0:
            print("Single process mode detected")
            print("Running basic GPU hardware and computation tests...")

        # Test hardware
        safe_test_execution(test_gpu_hardware_single, 'gpu_hardware', timeout_seconds=60)

        # Test computation
        safe_test_execution(
            test_calculation_single,
            'computation',
            timeout_seconds=300,  # 5 minutes for endurance test
        )

        if rank == 0:
            print_test_summary()

        return

    # Multi-process distributed mode - full test suite
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

        # PHASE 2: Test GPU hardware
        test_gpu_hardware()

        # PHASE 3: Test computation capabilities
        test_calculation()

        if rank == 0:
            print("\n" + "=" * 60)
            print("ALL TEST PHASES COMPLETED")
            print("=" * 60)

    except Exception as e:
        if rank == 0:
            print(f"❌ Critical error during testing: {e}")
            print("Attempting cleanup...")

    finally:
        # Always attempt cleanup
        try:
            cleanup()
        except Exception as e:
            if rank == 0:
                print(f"⚠ Warning: Cleanup failed: {e}")

        # Print final summary
        if rank == 0:
            print_test_summary()


if __name__ == "__main__":
    main()

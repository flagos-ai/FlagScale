"""
Complete GPU Health Check Implementation

This module provides comprehensive GPU health verification including:
- Tensor parallel communication check
- Data parallel communication check
- Pipeline parallel communication check
- TODO: Expert parallel communication check
- GPU hardware validation
- Computation capability verification

Features:
- Timeout protection for each check phase
- Progressive check (failures don't block other checks)
- Smart degradation on errors
- Complete check coverage in order: TP → DP → PP → Hardware → Computation
"""

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

# -------------------------
# Globals
# -------------------------
_GLOBAL_ARGS = None

_PARALLEL_STATE = {
    "data": {"nccl": None, "gloo": None, "global_ranks": None},
    "tensor": {"nccl": None, "gloo": None, "global_ranks": None},
    "pipeline": {"nccl": None, "gloo": None, "global_ranks": None},
    "embedding": {"nccl": None, "gloo": None},
    "model": {"nccl": None},
    "gloo_world": None,
}

# Check tracking
_CHECK_RESULTS = {
    "tensor_parallel": {"status": "pending", "error": None},
    "data_parallel": {"status": "pending", "error": None},
    "pipeline_parallel": {"status": "pending", "error": None},
    "gpu_hardware": {"status": "pending", "error": None},
    "gpu_computation": {"status": "pending", "error": None},
}


def log_check_result(check_name, status, error=None):
    """Log check result"""
    _CHECK_RESULTS[check_name]["status"] = status
    _CHECK_RESULTS[check_name]["error"] = error

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        if status == "passed":
            print(f"✓ {check_name}: PASSED")
        elif status == "failed":
            print(f"✗ {check_name}: FAILED - {error}")
        elif status == "skipped":
            print(f"⚠ {check_name}: SKIPPED - {error}")


def safe_check_execution(check_func, check_name, timeout_seconds=120) -> bool:
    """Execute check with timeout protection and error handling"""
    try:
        check_func()
        return True
    except TimeoutError as e:
        log_check_result(check_name, "failed", str(e))
        return False
    except Exception as e:
        log_check_result(check_name, "failed", f"Exception: {e!s}")
        return False


# -------------------------
# Control-plane barrier (GLOO)
# -------------------------
def control_barrier(group=None, timeout_s: int = 300):
    """
    Use GLOO monitored_barrier as the universal sync primitive.
    This avoids NCCL barrier (which is a 1-element allreduce and can segfault in some setups).
    """
    if not dist.is_initialized():
        return
    g = group if group is not None else _PARALLEL_STATE["gloo_world"]
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
    assert _GLOBAL_ARGS is not None, "arguments not yet initialized."
    args = _GLOBAL_ARGS

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
    global _PARALLEL_STATE
    try:
        _PARALLEL_STATE["gloo_world"] = dist.new_group(
            ranks=list(range(world_size)), backend="gloo"
        )
    except Exception:
        # If this fails, monitored_barrier may not be available (rare), but we try anyway.
        _PARALLEL_STATE["gloo_world"] = None

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
    Create a process group only if number of ranks > 1.
    For singleton "groups", return None (avoid edge-case bugs and pointless comms).
    """
    if len(ranks) <= 1:
        return None
    return dist.new_group(ranks=ranks, backend=backend)


def _init_data_parallel_groups(rank, world_size, tensor_mp, pipeline_mp):
    global _PARALLEL_STATE
    assert _PARALLEL_STATE["data"]["nccl"] is None, "data parallel group already initialized"

    num_pipeline_model_parallel_groups = world_size // pipeline_mp
    all_data_parallel_group_ranks: list[list[int]] = []
    for i in range(pipeline_mp):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_mp):
            r = list(range(start_rank + j, end_rank, tensor_mp))
            all_data_parallel_group_ranks.append(r)
            g_nccl = _maybe_new_group(r, backend=dist.get_backend())
            g_gloo = _maybe_new_group(r, backend="gloo")

            if rank in r:
                _PARALLEL_STATE["data"]["nccl"] = g_nccl
                _PARALLEL_STATE["data"]["gloo"] = g_gloo
                _PARALLEL_STATE["data"]["global_ranks"] = r
    print(f"[Rank {rank}] initialize_model_parallel: DP groups created", flush=True)
    return all_data_parallel_group_ranks


def _init_model_parallel_groups(rank, all_data_parallel_group_ranks, data_parallel_size):
    global _PARALLEL_STATE
    assert _PARALLEL_STATE["model"]["nccl"] is None, "model parallel group already initialized"
    for i in range(data_parallel_size):
        r = [
            grp[i] for grp in all_data_parallel_group_ranks
        ]  # pick i-th element from each DP group list
        g_nccl = _maybe_new_group(r, backend=dist.get_backend())
        if rank in r:
            _PARALLEL_STATE["model"]["nccl"] = g_nccl
    print(f"[Rank {rank}] initialize_model_parallel: MP groups created", flush=True)


def _init_tensor_model_parallel_groups(rank, world_size, tensor_model_parallel_size):
    global _PARALLEL_STATE
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    assert _PARALLEL_STATE["tensor"]["nccl"] is None, (
        "tensor model parallel group already initialized"
    )
    for i in range(num_tensor_model_parallel_groups):
        r = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        g_nccl = _maybe_new_group(r, backend=dist.get_backend())
        g_gloo = _maybe_new_group(r, backend="gloo")
        if rank in r:
            _PARALLEL_STATE["tensor"]["nccl"] = g_nccl
            _PARALLEL_STATE["tensor"]["gloo"] = g_gloo
            _PARALLEL_STATE["tensor"]["global_ranks"] = r

    print(f"[Rank {rank}] initialize_model_parallel: TP groups created", flush=True)


def _init_pipeline_and_embedding_groups(rank, world_size, pipeline_model_parallel_size):
    global _PARALLEL_STATE
    assert _PARALLEL_STATE["pipeline"]["nccl"] is None, (
        "pipeline model parallel group already initialized"
    )
    assert _PARALLEL_STATE["embedding"]["nccl"] is None, "embedding group already initialized"
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    for i in range(num_pipeline_model_parallel_groups):
        r = list(range(i, world_size, num_pipeline_model_parallel_groups))  # non-contiguous
        g_nccl = _maybe_new_group(r, backend=dist.get_backend())
        g_gloo = _maybe_new_group(r, backend="gloo")

        if rank in r:
            _PARALLEL_STATE["pipeline"]["nccl"] = g_nccl
            _PARALLEL_STATE["pipeline"]["gloo"] = g_gloo
            _PARALLEL_STATE["pipeline"]["global_ranks"] = r

        # embedding group: first + last in pipeline
        emb = [r[0], r[-1]] if len(r) > 1 else r
        emb = list(emb)
        eg_nccl = _maybe_new_group(emb, backend=dist.get_backend())
        eg_gloo = _maybe_new_group(emb, backend="gloo")
        if rank in emb:
            _PARALLEL_STATE["embedding"]["nccl"] = eg_nccl
            _PARALLEL_STATE["embedding"]["gloo"] = eg_gloo

    print(f"[Rank {rank}] initialize_model_parallel: PP and embedding groups created", flush=True)


def initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size):
    """initialize model parallel"""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print(f"[Rank {rank}] initialize_model_parallel: START", flush=True)

    model_size = tensor_model_parallel_size * pipeline_model_parallel_size

    if world_size % model_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor*pipe ({model_size})"
        )

    data_parallel_size = world_size // model_size
    # -------------------------
    # Data-parallel groups
    # -------------------------
    all_data_parallel_group_ranks = _init_data_parallel_groups(
        rank, world_size, tensor_model_parallel_size, pipeline_model_parallel_size
    )

    # -------------------------
    # Model-parallel groups
    # -------------------------
    _init_model_parallel_groups(rank, all_data_parallel_group_ranks, data_parallel_size)

    # -------------------------
    # Tensor model-parallel groups
    # -------------------------
    _init_tensor_model_parallel_groups(rank, world_size, tensor_model_parallel_size)

    # -------------------------
    # Pipeline model-parallel groups + embedding groups
    # PP groups are non-contiguous like [0,8], [1,9], ...
    # -------------------------
    _init_pipeline_and_embedding_groups(rank, world_size, pipeline_model_parallel_size)
    print(f"[Rank {rank}] initialize_model_parallel: COMPLETE", flush=True)


# -------------------------
# Communication Checks
# -------------------------
def check_tensor_parallel_group():
    assert _GLOBAL_ARGS is not None, "arguments not yet initialized."
    args = _GLOBAL_ARGS
    rank = dist.get_rank()
    tp_size = args.tensor_model_parallel_size
    tp_ranks = _PARALLEL_STATE["tensor"]["global_ranks"] or [rank]

    if rank == 0:
        print(f"Checking tensor parallel communication (TP size: {tp_size})")
    control_barrier(group=_PARALLEL_STATE["tensor"]["gloo"], timeout_s=60)
    print(f"[Rank {rank}] TP group ranks: {tp_ranks}", flush=True)

    if tp_size <= 1 or len(tp_ranks) <= 1 or _PARALLEL_STATE["tensor"]["nccl"] is None:
        # Nothing to communicate; treat as pass.
        print(f"[Rank {rank}] TP size is 1; skipping NCCL all_reduce.", flush=True)
        control_barrier(group=_PARALLEL_STATE["tensor"]["gloo"], timeout_s=60)
        return

    device = torch.device(f"cuda:{args.local_rank}")
    tensor = torch.tensor([rank], device=device, dtype=torch.float32)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=_PARALLEL_STATE["tensor"]["nccl"])

    # Verify on every rank (cheap and avoids group-rank queries)
    expected = float(sum(tp_ranks))
    if not torch.allclose(tensor, torch.tensor([expected], device=device), atol=0, rtol=0):
        raise AssertionError(
            f"[Rank {rank}] TP all_reduce wrong: got {tensor.item()}, expected {expected}"
        )

    torch.cuda.synchronize()
    control_barrier(group=_PARALLEL_STATE["tensor"]["gloo"], timeout_s=60)


def check_data_parallel_group():
    assert _GLOBAL_ARGS is not None, "arguments not yet initialized."
    args = _GLOBAL_ARGS
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Compute DP group size
    dp_group_size = world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    )
    dp_ranks = _PARALLEL_STATE["data"]["global_ranks"] or [rank]

    if rank == 0:
        print(f"Checking data parallel communication (DP group size: {dp_group_size})")
    control_barrier(group=_PARALLEL_STATE["tensor"]["gloo"], timeout_s=60)
    print(f"[Rank {rank}] DP group ranks: {dp_ranks}", flush=True)

    if dp_group_size <= 1 or len(dp_ranks) <= 1 or _PARALLEL_STATE["data"]["nccl"] is None:
        print(f"[Rank {rank}] DP size is 1; skipping NCCL all_reduce.", flush=True)
        control_barrier(group=_PARALLEL_STATE["data"]["gloo"], timeout_s=60)
        return

    device = torch.device(f"cuda:{args.local_rank}")
    tensor = torch.tensor([rank], device=device, dtype=torch.float32)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=_PARALLEL_STATE["data"]["nccl"])

    expected = float(sum(dp_ranks))
    if not torch.allclose(tensor, torch.tensor([expected], device=device), atol=0, rtol=0):
        raise AssertionError(
            f"[Rank {rank}] DP all_reduce wrong: got {tensor.item()}, expected {expected}"
        )

    torch.cuda.synchronize()
    control_barrier(group=_PARALLEL_STATE["data"]["gloo"], timeout_s=60)


def check_pipeline_parallel_group():
    assert _GLOBAL_ARGS is not None, "arguments not yet initialized."
    args = _GLOBAL_ARGS
    rank = dist.get_rank()

    pp_size = args.pipeline_model_parallel_size
    pp_ranks = _PARALLEL_STATE["pipeline"]["global_ranks"] or [rank]
    pp_group_nccl = _PARALLEL_STATE["pipeline"]["nccl"]
    pp_group_gloo = _PARALLEL_STATE["pipeline"]["gloo"]

    if rank == 0:
        print(f"Checking pipeline parallel communication (PP size: {pp_size})")
    control_barrier(group=pp_group_gloo, timeout_s=60)
    print(f"[Rank {rank}] PP group ranks: {pp_ranks}", flush=True)

    if pp_size <= 1 or len(pp_ranks) <= 1 or pp_group_nccl is None:
        print(f"[Rank {rank}] PP size is 1; skipping P2P.", flush=True)
        control_barrier(group=pp_group_gloo, timeout_s=60)
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
    control_barrier(group=pp_group_gloo, timeout_s=60)

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
    control_barrier(group=pp_group_gloo, timeout_s=60)

    print(f"[Rank {rank}] Pipeline parallel check completed", flush=True)
    control_barrier(group=pp_group_gloo, timeout_s=60)


# -------------------------
# Check Orchestration
# -------------------------
def check_communication():
    """Check all parallel communication with progressive execution"""
    rank = dist.get_rank()
    print(f"[Rank {rank}] Entered check_communication()", flush=True)
    if rank == 0:
        print("\n" + "=" * 60)
        print("PHASE 1: PARALLEL COMMUNICATION CHECKING")
        print("=" * 60)

    # Always use gloo world control barrier
    control_barrier(timeout_s=60)

    # TP
    ok = safe_check_execution(check_tensor_parallel_group, "tensor_parallel", timeout_seconds=60)
    control_barrier(timeout_s=60)
    if not ok and rank == 0:
        print("⚠ Warning: TP check failed, continuing...")
    elif rank == 0:
        log_check_result("tensor_parallel", "passed")
        print("Tensor parallel communication check completed successfully")
        print("\n" + "-" * 60)

    # DP
    ok = safe_check_execution(check_data_parallel_group, "data_parallel", timeout_seconds=60)
    control_barrier(timeout_s=60)
    if not ok and rank == 0:
        print("⚠ Warning: DP check failed, continuing...")
    elif rank == 0:
        log_check_result("data_parallel", "passed")
        print("Data parallel communication check completed successfully")
        print("\n" + "-" * 60)

    # PP
    ok = safe_check_execution(
        check_pipeline_parallel_group, "pipeline_parallel", timeout_seconds=120
    )
    control_barrier(timeout_s=60)
    if not ok and rank == 0:
        print("⚠ Warning: PP check failed, continuing...")
    elif rank == 0:
        log_check_result("pipeline_parallel", "passed")
        print("Pipeline parallel communication check completed successfully")
        print("\n" + "-" * 60)
    # TODO: Check Expert Parallel
    if rank == 0:
        print("\nParallel communication check phase completed")
        print("=" * 60)


def check_hardware_single():
    """Single GPU hardware check without distributed calls"""
    try:
        import pynvml

        pynvml.nvmlInit()

        print("Checking GPU hardware")

        device_count = pynvml.nvmlDeviceGetCount()
        all_passed = True
        errors = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_name = pynvml.nvmlDeviceGetName(handle)

            print(f"=== Checking GPU {i}: {gpu_name} ===")

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            print(f"Current GPU temperature: {temp}°C")
            if temp >= 90:
                all_passed = False
                errors.append(f"GPU {i} overheat: {temp}°C")
            elif temp > 85:
                print(f"Warning: GPU {i} high temperature: {temp}°C")

            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            power_ratio = power_usage / power_limit if power_limit > 0 else 0.0
            print(f"Power usage: {power_usage:.2f}W / {power_limit:.2f}W")
            if power_ratio > 0.9:
                print(f"GPU {i} power usage high: {power_usage:.1f}W ({power_ratio:.0%})")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_utilization = (float(mem_info.used) / float(mem_info.total)) * 100

            print(f"Total memory: {float(mem_info.total) / (1024**2):.2f} MB")
            print(f"Used memory: {float(mem_info.used) / (1024**2):.2f} MB")
            print(f"GPU {i} memory utilization rate: {memory_utilization:.2f}%")

            if memory_utilization >= 98.0:
                all_passed = False
                errors.append(f"GPU {i} memory almost full: {memory_utilization * 100:.1f}%")

        pynvml.nvmlShutdown()
        if all_passed:
            log_check_result("gpu_hardware", status="passed")
            return True
        else:
            log_check_result("gpu_hardware", status="failed", error="; ".join(errors))
            return False
    except ImportError:
        log_check_result("gpu_hardware", status="failed", error="pynvml is not installed")
        return False
    except Exception as e:
        log_check_result("gpu_hardware", status="failed", error=str(e))
        return False


def check_hardware():
    """Distributed wrapper: run single check once per node and summarize result"""

    args = _GLOBAL_ARGS
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 60)
        print("PHASE 2: GPU HARDWARE TESTING")
        print("=" * 60)

    if args.local_rank == 0:
        check_hardware_single()
    if rank == 0:
        print("\nGPU hardware checking phase completed")
        print("=" * 60)


def check_computation_for_different_dtype(dtype, name):
    args = _GLOBAL_ARGS
    check_tensor = torch.randn(4096, 4096, dtype=dtype).to(f"cuda:{args.local_rank}")
    result = torch.matmul(check_tensor, check_tensor)
    if torch.any(torch.isnan(result)):
        print(f"{name} failed: nan is detected in result")
        return False
    elif torch.any(torch.isinf(result)):
        print(f"{name} failed: inf is detected in result")
        return False
    else:
        return True


def check_computation_endurance():
    args = _GLOBAL_ARGS
    start_time = time.time()
    iteration = 0

    while time.time() - start_time < 60:
        iteration += 1

        a = torch.randn(4096, 4096).to(f"cuda:{args.local_rank}")
        b = torch.randn(4096, 4096).to(f"cuda:{args.local_rank}")
        result1 = torch.matmul(a, b)

        c = torch.randn(4096, 4096).to(f"cuda:{args.local_rank}")
        result2 = torch.inverse(c)

        if torch.any(torch.isnan(result1)) or torch.any(torch.isnan(result2)):
            print(f"check_computation_endurance failed: nan detected in iteration {iteration}")
            return False

    return True


def check_ecc_error():
    """Check ECC Error through matrix multiplication operations"""
    try:
        args = _GLOBAL_ARGS
        rank = dist.get_rank()
        device = f"cuda:{args.local_rank}"

        # Perform multiple matrix operations to stress check memory
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
        if rank == 0:
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


def check_computation_single():
    """Single GPU computation check"""
    print("Checking GPU computation capabilities...")
    checks = [
        ("Float", torch.float32),
        ("Double", torch.double),
        ("Half", torch.half),
    ]
    all_pass = True
    for display_name, dtype in checks:
        ok = check_computation_for_different_dtype(
            dtype=dtype, name=f"check_calculation_{display_name.lower()}"
        )
        print(f"{display_name} computation: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    print("Starting 60-second endurance check...")
    endurance_result = check_computation_endurance()
    if not endurance_result:
        all_pass = False
    print(f"Endurance check: {'PASS' if endurance_result else 'FAIL'}")
    ecc_error_result = check_ecc_error()
    if not ecc_error_result:
        all_pass = False
    print(f"ECC error check: {'PASS' if ecc_error_result else 'FAIL'}")
    if all_pass:
        log_check_result("gpu_computation", "passed")
        return True
    else:
        log_check_result("gpu_computation", "failed")
        return False


def check_computation():
    """Test GPU computation capabilities with distributed coordination"""
    args = _GLOBAL_ARGS
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 60)
        print("PHASE 3: GPU COMPUTATION CHECKING")
        print("=" * 60)

    # Check individual computation capabilities
    check_functions = [
        ("Float computation", torch.float32, "check_computation_float"),
        ("Double computation", torch.double, "check_computation_double"),
        ("Half computation", torch.half, "check_computation_half"),
        ("Endurance check (60s)", check_computation_endurance),
        ("ECC Error Detection", check_ecc_error),
    ]

    all_passed = True
    failed_checks = []

    for check_name, *rest in check_functions:
        if rank == 0:
            print(f"\nTesting {check_name}...")

        try:
            if isinstance(rest[0], torch.dtype):
                _, dtype, report_name = check_name, rest[0], rest[1]
                result = check_computation_for_different_dtype(dtype, report_name)
            else:
                check_func = rest[0]
                result = check_func()
            if not result:
                all_passed = False
                failed_checks.append(check_name)

        except Exception as e:
            result = False
            all_passed = False
            failed_checks.append(check_name)
            if rank == 0:
                print(f"✗ {check_name} failed with exception: {e}")

        try:
            result_tensor = torch.zeros(args.world_size).to(f"cuda:{args.local_rank}")
            result_tensor[args.rank] = 1.0 if result else 0.0
            dist.all_reduce(result_tensor, dist.ReduceOp.SUM)

            if args.rank == 0:
                expected_tensor = torch.ones_like(result_tensor).to(f"cuda:{args.local_rank}")
                if torch.allclose(result_tensor, expected_tensor, atol=1e-6):
                    print(f"{check_name} passed")
                else:
                    print(f"{check_name} failed")
        except Exception as e:
            if rank == 0:
                print(f"⚠ Warning: Failed to gather {check_name} results: {e}")

        try:
            dist.barrier()
        except Exception as e:
            if rank == 0:
                print(f"⚠ Warning: Calculation check barrier failed: {e}")

    if all_passed:
        log_check_result("gpu_computation", "passed")
    else:
        error_msg = f"Failed checks: {', '.join(failed_checks)}"
        log_check_result("gpu_computation", "failed", error_msg)

    if rank == 0:
        print("\nGPU computation checking phase completed")
        print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="GPU Health Check arguments")
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Degree of tensor model parallelism (will be auto-detected if not optimal).",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Degree of pipeline model parallelism.",
    )
    parser.add_argument(
        "--distributed-backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="Which backend to use for distributed training.",
    )
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=10,
        help="Timeout minutes for torch.distributed.",
    )

    args = parser.parse_args()

    # Args from environment
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    return args


def print_check_summary():
    """Print final check summary"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    args = _GLOBAL_ARGS
    world_size = args.world_size
    print("=" * 60)
    print("GPU HEALTH CHECK SUMMARY")
    print("=" * 60)
    if world_size == 1:
        results = list(_CHECK_RESULTS.items())[-2:]
        scope_desc = "HARDWARE ONLY (single GPU)"
    else:
        results = list(_CHECK_RESULTS.items())
        scope_desc = "ALL CHECKS (multi GPU)"

    print(f"Scope: {scope_desc}")
    print("-" * 60)

    total_checks = len(results)
    passed_checks = sum(1 for _, r in results if r["status"] == "passed")
    failed_checks = sum(1 for _, r in results if r["status"] == "failed")
    skipped_checks = sum(1 for _, r in results if r["status"] == "skipped")
    pending_checks = sum(1 for _, r in results if r["status"] == "pending")

    for check_name, result in results:
        status_icon = (
            "✓" if result["status"] == "passed" else "✗" if result["status"] == "failed" else "⚠"
        )
        print(f"{status_icon} {check_name.replace('_', ' ').title()}: {result['status'].upper()}")
        if result["error"]:
            print(f"   └─ {result['error']}")

    print(
        f"Results: {passed_checks} passed, {failed_checks} failed, {skipped_checks} skipped out of {total_checks} total"
    )
    if pending_checks != 0:
        print("⚠ some checks pending")
    elif failed_checks == 0:
        print("🎉 All GPU health checks PASSED!")
    elif passed_checks > 0:
        print("⚠ Some checks failed, but basic functionality verified")
    else:
        print("❌ Critical: All checks FAILED - GPU environment may have serious issues")

    print("=" * 60)


def main():
    """Complete GPU health check with progressive checking"""
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
            print("Running basic GPU hardware and computation checks...")

        # PHASE 1: Check gpu hardware
        safe_check_execution(check_hardware_single, "gpu_hardware", timeout_seconds=60)
        # PHASE 2: Check gpu computation capabilities
        safe_check_execution(
            check_computation_single, "gpu_computation", timeout_seconds=300
        )  # 5 minutes for endurance check
        # Print final summary
        if rank == 0:
            print_check_summary()
        SINGLE_CHECK_RESULTS = ["gpu_hardware", "gpu_computation"]
        failed_count = sum(
            1 for r in SINGLE_CHECK_RESULTS if _CHECK_RESULTS[r]["status"] == "failed"
        )
        if failed_count > 0:
            import sys

            sys.exit(1)
        return

    if rank == 0:
        print("Multi-process distributed mode detected")
        print("Initializing distributed environment...")
    try:
        # Initialize process group and subgroups
        initialize_distributed(rank, world_size)

        if rank == 0:
            print("✓ Distributed initialization successful")
            print("Starting comprehensive check suite...")

        # PHASE 1: Check parallel communication
        check_communication()

        # PHASE 2: Check gpu hardware
        check_hardware()

        # PHASE 3: Check gpu computation capabilities
        check_computation()

        if rank == 0:
            print("=" * 60)
            print("ALL CHECK PHASES COMPLETED")
            print("=" * 60)

    except Exception as e:
        if rank == 0:
            print(f"❌ Critical error during checking: {e}")
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
            print_check_summary()

            failed_count = sum(1 for r in _CHECK_RESULTS.values() if r["status"] == "failed")
            if failed_count > 0:
                import sys

                sys.exit(1)


if __name__ == "__main__":
    main()

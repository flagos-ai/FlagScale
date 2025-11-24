import argparse
import os
import time

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

    model_size = tensor_model_parallel_size * pipeline_model_parallel_size

    if world_size % model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")

    data_parallel_size: int = world_size // model_size

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    rank = dist.get_rank()

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
    tp_group = get_tensor_model_parallel_group()
    rank = dist.get_rank()
    tensor = torch.tensor(
        [rank], device=f'cuda:{rank % torch.cuda.device_count()}', dtype=torch.float32
    )

    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=tp_group)

    if get_tensor_model_parallel_rank() == 0:
        expected_tensor = torch.tensor(
            [sum(list(_TENSOR_GLOBAL_RANKS))],
            device=f'cuda:{rank % torch.cuda.device_count()}',
            dtype=torch.float32,
        )
        assert torch.allclose(
            tensor, expected_tensor
        ), f"test_tensor_parallel_group_c10d failed on rank {rank}, expected: {expected_tensor}, received: {tensor}."

    dist.barrier()
    if rank == 0:
        print(f"test_tensor_parallel_group_c10d passed")


def test_data_parallel_group_c10d():
    global _DATA_GLOBAL_RANKS
    dp_group = get_data_parallel_group()
    rank = dist.get_rank()
    tensor = torch.tensor(
        [rank], device=f'cuda:{rank % torch.cuda.device_count()}', dtype=torch.float32
    )

    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=dp_group)

    if get_data_parallel_rank() == 0:
        expected_tensor = torch.tensor(
            [sum(list(_DATA_GLOBAL_RANKS))],
            device=f'cuda:{rank % torch.cuda.device_count()}',
            dtype=torch.float32,
        )
        assert torch.allclose(
            tensor, expected_tensor
        ), f"test_data_parallel_group_c10d failed on rank {rank}, expected: {expected_tensor}, received: {tensor}."

    dist.barrier()
    if rank == 0:
        print(f"test_data_parallel_group_c10d passed")


def test_pipeline_parallel_group_c10d():
    global _PIPELINE_GLOBAL_RANKS
    args = get_args()

    pp_ranks = _PIPELINE_GLOBAL_RANKS
    pp_rank = get_pipeline_model_parallel_rank()
    pp_group = get_pipeline_model_parallel_group()
    pp_size = args.pipeline_model_parallel_size
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

    prev_rank = None
    next_rank = None

    if pp_rank > 0:
        prev_rank = pp_ranks[pp_rank - 1]

    if pp_rank < pp_size - 1:
        next_rank = pp_ranks[pp_rank + 1]

    # Forward communication test.
    if next_rank is not None:
        send_tensor = torch.tensor([rank, pp_rank], device=device, dtype=torch.float32)
        dist.send(send_tensor, dst=next_rank)

    if prev_rank is not None:
        recv_tensor = torch.zeros(2, device=device, dtype=torch.float32)
        dist.recv(recv_tensor, src=prev_rank)
        expected_tensor = torch.tensor([prev_rank, pp_rank - 1], device=device, dtype=torch.float32)
        assert torch.allclose(
            recv_tensor, expected_tensor
        ), f"test_pipeline_parallel_group_c10d failed on rank {rank}, expected: {expected_tensor}, received: {recv_tensor}."

    dist.barrier(pp_group)

    # Backward communication test.
    if prev_rank is not None:
        send_tensor = torch.tensor([rank, pp_rank], device=device, dtype=torch.float32)
        dist.send(send_tensor, dst=prev_rank)

    if next_rank is not None:
        recv_tensor = torch.zeros(2, device=device, dtype=torch.float32)
        dist.recv(recv_tensor, src=next_rank)
        expected_tensor = torch.tensor([next_rank, pp_rank + 1], device=device, dtype=torch.float32)
        assert torch.allclose(
            recv_tensor, expected_tensor
        ), f"test_pipeline_parallel_group_c10d failed on rank {rank}, expected: {expected_tensor}, received: {recv_tensor}."

    dist.barrier(pp_group)

    dist.barrier()
    if rank == 0:
        print(f"test_pipeline_parallel_group_c10d passed")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():

    parser = argparse.ArgumentParser(description="GPU Health Check arguments")
    parser.add_argument(
        '--tensor-model-parallel-size',
        type=int,
        default=1,
        help='Degree of pipeline model parallelism.',
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
    args = get_args()

    if args.rank == 0:
        print(f"Testing tensor parallel group communication.")
    test_tensor_parallel_group_c10d()

    dist.barrier()

    if args.rank == 0:
        print(f"Testing data parallel group communication.")
    test_data_parallel_group_c10d()

    dist.barrier()

    if args.rank == 0:
        print(f"Testing pipeline parallel group communication.")
    test_pipeline_parallel_group_c10d()

    dist.barrier()


def test_gpu_hardware():

    args = get_args()
    if args.local_rank == 0:
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

    dist.barrier()
    return


def check_test_result(test_name, result_tensor):
    expected_tensor = torch.ones_like(result_tensor).cuda()
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
    test_tensor = torch.randn(4096, 4096).cuda()
    result = torch.matmul(test_tensor, test_tensor)

    return report_test_calculation_result("test_calculation_float", result)


def test_calculation_double():
    test_tensor = torch.randn(4096, 4096, dtype=torch.double).cuda()
    result = torch.matmul(test_tensor, test_tensor)

    return report_test_calculation_result("test_calculation_double", result)


def test_calculation_half():
    test_tensor = torch.randn(4096, 4096, dtype=torch.half).cuda()
    result = torch.matmul(test_tensor, test_tensor)

    return report_test_calculation_result("test_calculation_half", result)


def test_calculation_endurance():
    start_time = time.time()
    iteration = 0

    while time.time() - start_time < 60:
        iteration += 1

        a = torch.randn(4096, 4096).cuda()
        b = torch.randn(4096, 4096).cuda()
        result1 = torch.matmul(a, b)

        c = torch.randn(4096, 4096).cuda()
        result2 = torch.inverse(c)

        if torch.any(torch.isnan(result1)) or torch.any(torch.isnan(result2)):
            print(f"test_calculation_float failed: nan detected in iteration {iteration}")
            return False

    return True


def test_calculation():
    args = get_args()

    result = test_calculation_float()
    result_tensor = torch.zeros(args.world_size).cuda()
    result_tensor[args.rank] = 1.0 if result else 0.0
    dist.all_reduce(result_tensor, dist.ReduceOp.SUM)
    if args.rank == 0:
        check_test_result("test_calculation_float", result_tensor)

    result = test_calculation_double()
    result_tensor = torch.zeros(args.world_size).cuda()
    result_tensor[args.rank] = 1.0 if result else 0.0
    dist.all_reduce(result_tensor, dist.ReduceOp.SUM)
    if args.rank == 0:
        check_test_result("test_calculation_double", result_tensor)

    result = test_calculation_half()
    result_tensor = torch.zeros(args.world_size).cuda()
    result_tensor[args.rank] = 1.0 if result else 0.0
    dist.all_reduce(result_tensor, dist.ReduceOp.SUM)
    if args.rank == 0:
        check_test_result("test_calculation_half", result_tensor)

    result = test_calculation_endurance()
    result_tensor = torch.zeros(args.world_size).cuda()
    result_tensor[args.rank] = 1.0 if result else 0.0
    dist.all_reduce(result_tensor, dist.ReduceOp.SUM)
    if args.rank == 0:
        check_test_result("test_calculation_endurance", result_tensor)


def main():
    args = parse_args()

    set_args(args)

    # initialize process group and subgroups
    initialize_distributed(args.rank, args.world_size)

    # test communication within process group
    test_communication()

    test_gpu_hardware()

    test_calculation()

    cleanup()


if __name__ == "__main__":
    main()

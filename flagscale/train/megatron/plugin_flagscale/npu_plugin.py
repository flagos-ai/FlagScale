import time

import torch
import torch_npu

def get_device_arch_version():
    return 8

def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.core.datasets.utils import compile_helpers

        compile_helpers()
        print(
            '>>> done with dataset index builder. Compilation time: {:.3f} seconds'.format(time.time() - start_time),
            flush=True,
        )

def get_device(local_rank=None):
    backend = torch.distributed.get_backend()
    if backend == 'hccl':
        if local_rank is None:
            device = torch.device('cuda')
        else:
            device = torch.device(f'cuda:{local_rank}')
    elif backend == 'nccl':
        if local_rank is None:
            device = torch.device(cur_platform.device_name())
        else:
            device = torch.device(f'{cur_platform.device_name()}:{local_rank}')
    elif backend == 'gloo':
        device = torch.device('cpu')
    else:
        raise RuntimeError(f"Unsupported distributed backend: {backend}")
    return device
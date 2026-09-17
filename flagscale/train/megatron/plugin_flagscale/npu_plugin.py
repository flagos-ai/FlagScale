# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from pathlib import Path

import torch
import torch_npu
import torch_npu.profiler as npu_profiler


def create_pytorch_profiler(args, rank):
    from megatron.training.utils import print_rank_0

    def trace_handler(p):
        profile_dir = Path(f"{args.tensorboard_dir}/../torch_profile")
        profile_dir.mkdir(parents=True, exist_ok=True)
        p.export_chrome_trace(f"{profile_dir}/rank-{rank}.json.gz")

    print_rank_0("Using torch_npu.profiler for NPU profiling")
    prof = npu_profiler.profile(
        schedule=npu_profiler.schedule(
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=trace_handler,
        record_shapes=args.pytorch_profiler_collect_shapes,
        profile_memory=args.pytorch_profiler_collect_memory,
        with_stack=args.pytorch_profiler_collect_callstack,
    )
    return prof


def stop_pytorch_profiler(profiler):
    """Stop NPU profiling and finish native trace analysis."""
    profiler.stop()


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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT DeepSeek-V4.

GPU-local NUMA placement, guarded fixed-shape pipeline communication, and
compilation of the original FP32 mHC aggregate backward are enabled at startup.
"""

import copy
import csv
import ctypes
import json
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path


def _expand_cpu_list(value):
    result = set()
    for part in value.strip().split(','):
        bounds = [int(x) for x in part.split('-')]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _bind_gpu_local_numa():
    """Apply and verify process-local placement; no CUDA APIs or tensor casts."""
    local_rank = int(os.environ['LOCAL_RANK'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    physical_gpu = int(visible.split(',')[local_rank]) if visible else local_rank
    query = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True,
    )
    buses = {int(row[0]): row[1].strip().lower() for row in csv.reader(query.stdout.splitlines())}
    parts = buses[physical_gpu].split(':')
    pci = ':'.join([parts[0][-4:], *parts[1:]])
    node = int((Path('/sys/bus/pci/devices') / pci / 'numa_node').read_text())
    if node < 0:
        raise RuntimeError(f'No GPU-local NUMA node reported for {pci}')
    cpus = _expand_cpu_list((Path('/sys/devices/system/node') / f'node{node}' / 'cpulist').read_text())
    original_cpus = os.sched_getaffinity(0)
    selected = cpus & original_cpus
    if not selected:
        raise RuntimeError('GPU-local CPUs are outside the worker CPU allowance')
    os.sched_setaffinity(0, selected)

    numa = ctypes.CDLL('libnuma.so.1', use_errno=True)
    numa.numa_allocate_nodemask.restype = ctypes.c_void_p
    numa.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    numa.numa_bitmask_setbit.restype = ctypes.c_void_p
    numa.numa_set_membind.argtypes = [ctypes.c_void_p]
    numa.numa_set_membind.restype = None
    numa.numa_bitmask_free.argtypes = [ctypes.c_void_p]
    numa.numa_bitmask_free.restype = None
    mask = numa.numa_allocate_nodemask()
    if not mask:
        raise RuntimeError('Could not allocate a NUMA policy mask')
    try:
        numa.numa_bitmask_setbit(mask, node)
        numa.numa_set_membind(mask)
    finally:
        numa.numa_bitmask_free(mask)
    policy = subprocess.run(['numactl', '--show'], capture_output=True, text=True, check=True)
    fields = dict(line.split(':', 1) for line in policy.stdout.splitlines() if ':' in line)
    actual_cpus = os.sched_getaffinity(0)
    actual_nodes = [int(x) for x in fields['membind'].split()]
    if actual_cpus != selected or fields['policy'].strip() != 'bind' or actual_nodes != [node]:
        raise RuntimeError(f'CPU/NUMA placement did not apply: {policy.stdout} {policy.stderr}')
    record = dict(
        rank=int(os.environ.get('RANK', '-1')),
        local_rank=local_rank,
        physical_gpu=physical_gpu,
        pci=pci,
        numa_node=node,
        cpu_count=len(actual_cpus),
        cpus=sorted(actual_cpus),
        memory_policy=fields['policy'].strip(),
        memory_nodes=actual_nodes,
        before_torch_import='torch' not in sys.modules,
    )
    print('[dsv4-gpu-local-numa] ' + json.dumps(record), flush=True)
    return record


def _install_fixed_pp_transport():
    import torch

    from megatron.core import parallel_state, utils
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
    from megatron.training import get_args

    if getattr(P2PCommunicator, '_dsv4_fixed_pp_installed', False):
        return
    original_init = P2PCommunicator.__init__
    original_communicate = P2PCommunicator._communicate

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        config = self.config
        args = get_args()
        if (
            config.tensor_model_parallel_size != 1
            or config.pipeline_model_parallel_size <= 1
            or getattr(config, 'enable_hetero', False)
            or config.mtp_standalone
        ):
            raise RuntimeError('Fixed PP probe supports homogeneous TP1 PP>1 without standalone MTP')
        width = config.hidden_size
        if getattr(config, 'enable_hyper_connections', False):
            width *= config.num_residual_streams
        self._dsv4_fixed_pp_shape = (
            args.seq_length // config.context_parallel_size,
            args.micro_batch_size,
            width,
        )
        record = {
            'rank': int(os.environ['RANK']),
            'expected_shape': self._dsv4_fixed_pp_shape,
            'model_variable_seq_lengths': config.variable_seq_lengths,
            'guard_every_send': True,
            'only_transport_call_uses_fixed_config': True,
        }
        (Path(os.environ['OUTPUT_PATH']) / f"fixed-pp-rank-{os.environ['RANK']}.json").write_text(
            json.dumps(record, indent=2)
        )

    @wraps(original_communicate)
    def communicate(self, *args, **kwargs):
        expected = getattr(self, '_dsv4_fixed_pp_shape', None)
        original_config = self.config
        if expected is None or not original_config.variable_seq_lengths:
            # Preserve the existing communicator bootstrap, including its [1] probe.
            return original_communicate(self, *args, **kwargs)
        shape_hint = kwargs.get('tensor_shape')
        if shape_hint is not None and tuple(shape_hint) != expected:
            raise RuntimeError(f'Pipeline schedule shape changed: {shape_hint} != {expected}')
        for name in ('tensor_send_next', 'tensor_send_prev'):
            tensor = kwargs.get(name)
            if tensor is not None:
                if tuple(tensor.shape) != expected or tensor.dtype != original_config.pipeline_dtype:
                    raise RuntimeError(
                        f'Fixed PP send contract violated: {name} shape={tuple(tensor.shape)} '
                        f'dtype={tensor.dtype}, expected={expected}/{original_config.pipeline_dtype}'
                    )
        # Keep model/data config and mutable callback updates intact. Only this
        # synchronous Python transport call sees a fresh shallow copy.
        transport_config = copy.copy(original_config)
        transport_config.variable_seq_lengths = False
        self.config = transport_config
        enabled = getattr(utils, '_nvtx_enabled', False)
        if enabled:
            actual = kwargs.get('group') or self.pp_group
            known = parallel_state._PIPELINE_MODEL_PARALLEL_GROUP
            is_pp = actual in known if isinstance(known, list) else actual is known
            torch.cuda.nvtx.range_push('COMM|' + ('PP' if is_pp else 'UNKNOWN_P2P') + '|fixed_pp')
        try:
            return original_communicate(self, *args, **kwargs)
        finally:
            if enabled:
                torch.cuda.nvtx.range_pop()
            self.config = original_config

    P2PCommunicator.__init__ = initialize
    P2PCommunicator._communicate = communicate
    P2PCommunicator._dsv4_fixed_pp_installed = True


def _install_hagg_backward_compile():
    import torch

    from megatron.core.fusions import fused_mhc_kernels

    original = fused_mhc_kernels._torch_h_aggregate_bwd
    if hasattr(original, "_torchdynamo_orig_callable"):
        return
    fused_mhc_kernels._torch_h_aggregate_bwd = torch.compile(original)
    print("[dsv4-bf16] compiled original mHC aggregate backward", flush=True)


def _configure_runtime():
    # CPU/memory placement must precede Torch and model imports.
    _bind_gpu_local_numa()
    _install_fixed_pp_transport()
    _install_hagg_backward_compile()


if __name__ == "__main__":
    _configure_runtime()


from flagscale.models.megatron.deepseek_v4.deepseek_builder import deepseek_builder  # noqa: E402
from train_gpt import main  # noqa: E402
from model_specific_batch import get_batch_for_model_with_all_stage_tokens  # noqa: E402


def get_batch(data_iterator, vp_stage=None, dualpipev_stage=None):
    return get_batch_for_model_with_all_stage_tokens(data_iterator, vp_stage, dualpipev_stage)


if __name__ == "__main__":
    main(deepseek_builder, get_batch_func=get_batch, build_dataset_on_all_pipeline_stages=True)

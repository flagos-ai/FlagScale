#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
nproc="${IMAGE_BUILD_RUNTIME_SMOKE_NPROC:?IMAGE_BUILD_RUNTIME_SMOKE_NPROC is required}"
device_count="${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:?IMAGE_BUILD_RUNTIME_DEVICE_COUNT is required}"

if [ "$phase" != post ]; then
    exit 0
fi

if [ "$task" = inference ]; then
    docker run --rm --entrypoint sh "$candidate" -eu -c '
for cache_dir in /root/.triton/cache /root/.cache/torch/kernels; do
    cached_file=$(find "$cache_dir" -type f -print -quit 2>/dev/null || true)
    if [ -n "$cached_file" ]; then
        echo "Hygon inference image contains runtime JIT cache: $cached_file" >&2
        exit 1
    fi
done
'
fi

runtime_options=(
    --rm
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --ipc=host
    --volume /opt/hyhal:/opt/hyhal:ro
    --env HSA_FORCE_FINE_GRAIN_PCIE=1
    # Select the FlagOS implementation so this contract verifies the backend
    # expected by Hygon training rather than Transformer Engine's fallback.
    --env TE_FL_PREFER=flagos
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
)

if [ "$task" = train ]; then
    docker run "${runtime_options[@]}" \
        --env EXPECTED_DEVICE_COUNT="$device_count" \
        --entrypoint python "$candidate" -c '
import os
import torch
import flag_gems
import megatron.core
import transformer_engine
from megatron.plugin.platform import get_platform
from transformer_engine.plugin.core import get_manager

required = int(os.environ["EXPECTED_DEVICE_COUNT"])
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= required
assert flag_gems.vendor_name == "hygon"
assert get_platform().device_name() == "cuda"
assert get_manager().get_selected_impl_id("generic_gemm") == "default.flagos"
value = torch.tensor(list(range(8)), dtype=torch.float32, device="cuda")
assert (value * 2).cpu().tolist() == [0., 2., 4., 6., 8., 10., 12., 14.]
'
elif [ "$task" = inference ]; then
    docker run "${runtime_options[@]}" \
        --env EXPECTED_DEVICE_COUNT="$device_count" \
        --env GEMS_VENDOR=hygon \
        --env VLLM_PLUGINS=fl \
        --env VLLM_FL_PLATFORM=hygon \
        --entrypoint python "$candidate" -c '
import importlib.metadata as metadata
import os

import flag_gems
import torch
import vllm
import vllm_fl

required = int(os.environ["EXPECTED_DEVICE_COUNT"])
plugins = {
    entry.name: entry.value
    for entry in metadata.entry_points(group="vllm.platform_plugins")
}
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= required
assert flag_gems.vendor_name == "hygon"
assert vllm.__version__.startswith("0.20.2"), vllm.__version__
assert plugins.get("fl") == "vllm_fl:register", plugins
value = torch.tensor([1.0, 2.0], device="cuda")
assert (value + 1).cpu().tolist() == [2.0, 3.0]
'
else
    echo "Unsupported Hygon image task: $task" >&2
    exit 1
fi

# Each smoke test runs in its own container network namespace. Use a fixed
# local rendezvous port because torchrun's automatic port selection is flaky
# with the DTK TCPStore and can resolve to localhost:0 on this runner.
docker run "${runtime_options[@]}" \
    --env EXPECTED_WORLD_SIZE="$nproc" \
    --env GEMS_VENDOR=hygon \
    "$candidate" \
    torchrun --nnodes=1 --node_rank=0 --nproc_per_node="$nproc" \
        --master_addr=127.0.0.1 --master_port=29500 \
        --no-python python -c '
import os
import torch

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["EXPECTED_WORLD_SIZE"])
torch.cuda.set_device(rank)
torch.distributed.init_process_group(backend="nccl")
value = torch.tensor([rank], dtype=torch.int64, device="cuda")
torch.distributed.all_reduce(value)
assert value.item() == world * (world - 1) // 2
torch.distributed.destroy_process_group()
'

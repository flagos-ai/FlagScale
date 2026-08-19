#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
nproc="${IMAGE_BUILD_RUNTIME_SMOKE_NPROC:-2}"
device_count="${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:-}"

[ "$phase" = post ] || exit 0

docker_args=(
    --privileged
    --env MTHREADS_VISIBLE_DEVICES=all
    --env MTHREADS_DRIVER_CAPABILITIES=all
    --ipc=host
)

validate_runtime() {
    local runtime_task="$1"
    local runtime_env=()

    if [ "$runtime_task" = train ]; then
        runtime_env=(
            --env TORCH_DEVICE_BACKEND_AUTOLOAD=0
            --env TORCHDYNAMO_DISABLE=1
            --env TORCH_COMPILE_DISABLE=1
            --env NVTE_TORCH_COMPILE=0
        )
    fi

    docker run --rm "${docker_args[@]}" \
        "${runtime_env[@]}" \
        --env FLAGSCALE_RUNTIME_TASK="$runtime_task" \
        --env EXPECTED_WORLD_SIZE="$nproc" \
        --env EXPECTED_DEVICE_COUNT="$device_count" \
        --entrypoint bash "$candidate" -lc '
set -euo pipefail
runtime_task="${FLAGSCALE_RUNTIME_TASK:?}"
python - "$runtime_task" <<"PY"
import os
import sys

import torch
import torch_musa

task = sys.argv[1]
assert torch.musa.is_available()
expected_devices = int(
    os.environ.get("EXPECTED_DEVICE_COUNT")
    or os.environ["EXPECTED_WORLD_SIZE"]
)
assert torch.musa.device_count() >= expected_devices
value = torch.tensor(range(8), dtype=torch.float32, device="musa")
assert (value * 2).cpu().tolist() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]

if task == "train":
    import megatron.core
    import transformer_engine
    import transformer_engine_torch as tex
    from megatron.plugin.platform import get_platform
    from transformer_engine.pytorch import Linear

    assert get_platform().device_name() == "musa"
    for symbol in ("generic_gemm", "layernorm_fwd", "layernorm_bwd", "rmsnorm_fwd", "rmsnorm_bwd"):
        assert hasattr(tex, symbol), symbol
    torch.musa.set_device(0)
    layer = Linear(16, 8).to("musa")
    input_value = torch.randn(4, 16, device="musa", requires_grad=True)
    output = layer(input_value)
    output.sum().backward()
    assert bool(output.isfinite().all().item())
    assert input_value.grad is not None and bool(input_value.grad.isfinite().all().item())
    assert layer.weight.grad is not None and bool(layer.weight.grad.isfinite().all().item())

    print("MUSA train runtime:", torch.__version__, transformer_engine.__version__)
    print("Megatron:", megatron.core.__file__)
else:
    assert hasattr(torch_musa, "_MUSAC")
    import vllm

    print("MUSA inference/serve runtime:", torch.__version__, vllm.__version__)
PY
'
}

validate_collective() {
    docker run --rm "${docker_args[@]}" \
        --env EXPECTED_WORLD_SIZE="$nproc" \
        --entrypoint bash "$candidate" -lc '
set -euo pipefail
cat >/tmp/musa_collective.py <<"PY"
import os

import torch
import torch_musa

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["EXPECTED_WORLD_SIZE"])
torch.musa.set_device(rank)
torch.distributed.init_process_group(backend="mccl")
value = torch.tensor([rank], dtype=torch.int64, device="musa")
torch.distributed.all_reduce(value)
assert value.item() == world * (world - 1) // 2
torch.distributed.destroy_process_group()
PY
torchrun --standalone --nproc_per_node="${EXPECTED_WORLD_SIZE}" \
    --no-python python /tmp/musa_collective.py
'
}

case "$task" in
    train)
        validate_runtime train
        validate_collective
        ;;
    inference)
        validate_runtime inference
        ;;
    *)
        echo "Unsupported MUSA image task: $task" >&2
        exit 1
        ;;
esac

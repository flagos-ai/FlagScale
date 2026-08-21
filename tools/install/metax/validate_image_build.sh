#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
base_image="${IMAGE_BUILD_BASE_IMAGE:?IMAGE_BUILD_BASE_IMAGE is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
expected_devices="${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:-2}"
smoke_nproc="${IMAGE_BUILD_RUNTIME_SMOKE_NPROC:-$expected_devices}"

case "$task" in
    train|inference) ;;
    *) exit 0 ;;
esac

if [ "$phase" = pre ]; then
    docker pull "$base_image"
    docker run --rm \
        --env EXPECTED_DEVICE_COUNT="$expected_devices" \
        --env EXPECTED_WORLD_SIZE="$smoke_nproc" \
        --ipc=host --group-add video \
        --device=/dev/dri --device=/dev/mxcd --device=/dev/infiniband \
        --entrypoint bash "$base_image" -lc '
set -euo pipefail
python - <<"PY"
import os
import torch

count = torch.cuda.device_count()
print("torch:", torch.__version__)
print("devices:", count)
expected = int(os.environ["EXPECTED_DEVICE_COUNT"])
assert count >= expected, (
    f"expected at least {expected} MetaX devices, found {count}"
)
PY
cat >/tmp/metax_collective.py <<"PY"
import os
import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
dist.all_reduce(value)
assert value.item() == world * (world + 1) / 2, value
print(f"rank={rank} all_reduce={value.item()}")
dist.destroy_process_group()
PY
torchrun --nnodes=1 --nproc-per-node="${EXPECTED_WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port=29500 /tmp/metax_collective.py
'
    exit 0
fi

[ "$phase" = post ] || exit 0

if [ "$task" = inference ]; then
    docker run --rm \
        --env EXPECTED_DEVICE_COUNT="$expected_devices" \
        --ipc=host --group-add video \
        --device=/dev/dri --device=/dev/mxcd --device=/dev/infiniband \
        --entrypoint python "$candidate" -c '
import importlib.metadata as metadata
import pandas
import os
import torch
import deep_ep_cpp
import vllm_fl
from vllm.platforms import current_platform

print("torch:", torch.__version__)
print("vllm:", metadata.version("vllm"))
print("vllm-plugin-fl:", metadata.version("vllm-plugin-fl"))
print("pandas:", pandas.__version__)
print("platform:", type(current_platform).__module__, type(current_platform).__name__)
print("vendor:", current_platform.vendor_name)
print("device_type:", current_platform.device_type)

assert torch.cuda.device_count() >= int(os.environ["EXPECTED_DEVICE_COUNT"])
assert type(current_platform).__module__ == "vllm_fl.platform"
assert type(current_platform).__name__ == "PlatformFL"
assert current_platform.vendor_name == "metax"
assert current_platform.device_type == "cuda"

value = torch.ones(16, device="cuda:0")
assert value.sum().item() == 16
'
    exit 0
fi

docker run --rm \
    --env EXPECTED_DEVICE_COUNT="$expected_devices" \
    --ipc=host --group-add video \
    --device=/dev/dri --device=/dev/mxcd \
    --entrypoint python "$candidate" -c '
import os
import torch
import transformer_engine
from transformer_engine.pytorch import DotProductAttention, LayerNormLinear
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt import GPTModel

assert torch.cuda.device_count() >= int(os.environ["EXPECTED_DEVICE_COUNT"])
assert HAVE_TE
assert TESpecProvider is not None
print("transformer_engine:", transformer_engine.__file__)
print("TE modules:", DotProductAttention, LayerNormLinear)
print("megatron GPTModel:", GPTModel)
'

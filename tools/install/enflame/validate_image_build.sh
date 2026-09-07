#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
base_image="${IMAGE_BUILD_BASE_IMAGE:?IMAGE_BUILD_BASE_IMAGE is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
expected_devices="${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:-8}"

case "$task" in
    train) ;;
    *) exit 0 ;;
esac

case "$phase" in
    pre) image="$base_image" ;;
    post) image="$candidate" ;;
    *) exit 0 ;;
esac

[ "$phase" != pre ] || docker pull "$image"

docker run --rm \
    --privileged \
    --ipc=host \
    --network host \
    --volume /dev:/dev \
    --volume /sys:/sys \
    --env EXPECTED_DEVICE_COUNT="$expected_devices" \
    --env IMAGE_BUILD_PHASE="$phase" \
    --env IMAGE_BUILD_TASK="$task" \
    --entrypoint python \
    "$image" -c '
import importlib.metadata as metadata
import os
import subprocess
import torch

device_count = torch.gcu.device_count()
task = os.environ["IMAGE_BUILD_TASK"]
phase = os.environ["IMAGE_BUILD_PHASE"]

print("torch:", torch.__version__)
print("devices:", device_count)

assert device_count >= int(os.environ["EXPECTED_DEVICE_COUNT"])

value = torch.ones(16, device="gcu:0")
assert value.sum().item() == 16

import transformer_engine

print("transformer-engine:", metadata.version("transformer-engine"))
if phase == "post":
    import megatron
    import megatron.core

    source_path = os.environ["FLAGSCALE_MEGATRON_PATH"]
    source_revision = subprocess.check_output(
        ["git", "-C", source_path, "rev-parse", "HEAD"], text=True
    ).strip()
    print("megatron-source:", source_revision)
    assert source_revision == os.environ["FLAGSCALE_MEGATRON_REF"]

'

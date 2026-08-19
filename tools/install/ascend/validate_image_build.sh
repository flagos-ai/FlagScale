#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"

if [ "$phase" != post ]; then
    exit 0
fi

docker_args=(
    --rm
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    --volume /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro
    --volume /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro
    --privileged
)

[ "$task" = inference ] || exit 0

docker run "${docker_args[@]}" \
    --entrypoint python \
    "$candidate" -c '
import flag_gems
import vllm
import vllm_fl
from vllm.platforms import current_platform

print("platform:", type(current_platform).__module__, type(current_platform).__name__)
print("device_type:", current_platform.device_type)
print("dist_backend:", current_platform.dist_backend)
assert type(current_platform).__module__ == "vllm_fl.platform"
assert type(current_platform).__name__ == "PlatformFL"
assert current_platform.device_type == "npu"
assert current_platform.dist_backend == "hccl"
'

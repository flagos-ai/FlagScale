#!/bin/bash

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

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"

[ "$phase" = post ] || exit 0

validate_runtime() {
    local runtime_task="$1"
    local runtime_mode="$2"

    docker run --rm --gpus all \
        --env FLAGSCALE_RUNTIME_TASK="$runtime_task" \
        --env FLAGSCALE_RUNTIME_MODE="$runtime_mode" \
        --entrypoint bash "$candidate" -lc '
set -euo pipefail
runtime_task="${FLAGSCALE_RUNTIME_TASK:?}"
if [ "${FLAGSCALE_RUNTIME_MODE:?}" = all ]; then
    conda_root="${FLAGSCALE_CONDA:?}"
    env_name="${FLAGSCALE_ENV_NAME:?}"
    test -f "$conda_root/etc/profile.d/conda.sh"
    test -d "$conda_root/envs/$env_name"
    . "$conda_root/etc/profile.d/conda.sh"
    conda activate "$env_name"
else
    case "$runtime_task" in
        train) env_name=flagscale-train ;;
        inference|serve) env_name=flagscale-inference ;;
        *) echo "Unsupported CUDA runtime task: $runtime_task" >&2; exit 1 ;;
    esac
    conda_root="${FLAGSCALE_CONDA:-}"
    if [ -n "$conda_root" ] && [ -f "$conda_root/etc/profile.d/conda.sh" ] && \
       [ -d "$conda_root/envs/$env_name" ]; then
        . "$conda_root/etc/profile.d/conda.sh"
        conda activate "$env_name"
    elif [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && \
         [ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
        . "$UV_PROJECT_ENVIRONMENT/bin/activate"
    fi
fi
python - "$runtime_task" <<"PY"
import sys

import torch

task = sys.argv[1]
assert torch.cuda.is_available()
assert torch.cuda.device_count() > 0
value = torch.tensor(range(8), dtype=torch.float32, device="cuda")
assert (value * 2).cpu().tolist() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]

if task == "train":
    import megatron.core
    import transformer_engine

    print("CUDA train runtime:", torch.__version__, transformer_engine.__file__)
    print("Megatron:", megatron.core.__file__)
else:
    import vllm

    print("CUDA inference/serve runtime:", torch.__version__, vllm.__version__)
PY
'
}

case "$task" in
    train)
        validate_runtime train direct
        ;;
    inference)
        validate_runtime inference direct
        ;;
    all)
        validate_runtime train all
        validate_runtime inference all
        ;;
    *)
        echo "Unsupported CUDA image task: $task" >&2
        exit 1
        ;;
esac

#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$script_dir/../../.." && pwd)
phase=${IMAGE_BUILD_PHASE:?Expected pre or post}
test "${IMAGE_BUILD_TASK:?Expected train}" = train
nproc=${IMAGE_BUILD_RUNTIME_SMOKE_NPROC:?IMAGE_BUILD_RUNTIME_SMOKE_NPROC is required}
device_count=${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:?IMAGE_BUILD_RUNTIME_DEVICE_COUNT is required}
if ! [[ "$nproc" =~ ^[1-9][0-9]*$ ]] || [ "$nproc" -lt 2 ]; then
    echo "Expected a smoke process count of at least two: $nproc" >&2
    exit 2
fi
if ! [[ "$device_count" =~ ^[1-9][0-9]*$ ]] || [ "$device_count" -lt "$nproc" ]; then
    echo "Expected device count >= smoke process count: $device_count" >&2
    exit 2
fi
case "$phase" in
    pre) image=${IMAGE_BUILD_BASE_IMAGE:?}; mode=sdk; docker pull "$image" ;;
    post) image=${IMAGE_BUILD_CANDIDATE_IMAGE:?}; mode=train ;;
    *) echo "Unknown validation phase: $phase" >&2; exit 2 ;;
esac
options=$(${YQ_BIN:-yq} -r '.container_options' "$root/.github/configs/iluvatar.yml")
read -r -a docker_args <<< "$options"
name="flagscale-iluvatar-validate-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
timeout 600 docker run --rm -i --name "$name" "${docker_args[@]}" \
    --volume "$script_dir/env.sh:/tmp/flagscale-iluvatar-env.sh:ro" \
    --env EXPECTED_DEVICE_COUNT="$device_count" \
    --env SMOKE_NPROC="$nproc" --env SMOKE_MODE="$mode" \
    --entrypoint bash "$image" -c '
set -euo pipefail
source /tmp/flagscale-iluvatar-env.sh
if [ "$SMOKE_MODE" = sdk ]; then
    # SDK-only base: Python/PyTorch are intentionally installed by the Dockerfile.
    test -d "$COREX_ROOT"
    command -v ixsmi
    ixsmi
    /opt/cmake-corex/bin/cmake --version
    exit 0
fi
script=$(mktemp /tmp/iluvatar-image-check-XXXXXX.py)
trap '\''rm -f "$script"'\'' EXIT
cat > "$script"
python -m torch.distributed.run --standalone --nproc-per-node="$SMOKE_NPROC" "$script"
' <<'PY'
import os
from datetime import timedelta

from importlib.metadata import version

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
# COREX wheel metadata includes its vendor suffix; torch/version.py does not.
assert version("torch") == "2.10.0+corex.4.5.0.20260804", version("torch")
assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= int(os.environ["EXPECTED_DEVICE_COUNT"])
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)
print("torch:", torch.__version__, "device:", rank, torch.cuda.get_device_name(rank), flush=True)
x = torch.ones((16, 16), device=device, requires_grad=True)
(x @ x).sum().backward()
torch.testing.assert_close(x.grad.cpu(), torch.full((16, 16), 32.0))
assert dist.is_nccl_available(), "COREX torch NCCL-compatible backend is unavailable"
dist.init_process_group("nccl", timeout=timedelta(seconds=120))
try:
    value = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(value)
    assert value.item() == world * (world + 1) / 2
    if os.environ["SMOKE_MODE"] == "train":
        from transformer_engine.plugin.core import get_manager
        from transformer_engine.pytorch import Linear
        from megatron.core.extensions.transformer_engine import HAVE_TE
        from megatron.core.models.gpt import GPTModel

        assert HAVE_TE and GPTModel is not None
        selected = get_manager().get_selected_impl_id("generic_gemm")
        assert selected == "reference.torch", selected
        layer = Linear(32, 32, params_dtype=torch.bfloat16).to(device)
        inputs = torch.randn(8, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        output = layer(inputs)
        output.float().square().mean().backward()
        assert torch.isfinite(output).all().item()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all().item()
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all().item()
        before = layer.weight.detach().clone()
        torch.optim.AdamW(layer.parameters(), lr=1e-3).step()
        assert torch.isfinite(layer.weight).all().item()
        assert not torch.equal(before, layer.weight), "Optimizer did not update weights"
    torch.cuda.synchronize()
    print(f"rank={rank}: {os.environ['SMOKE_MODE']} image validation PASS", flush=True)
finally:
    dist.destroy_process_group()
PY

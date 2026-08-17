#!/usr/bin/env bash
# Copyright (c) 2026, BAAI. All rights reserved.

# Gloo process-group smoke for the module-namespaced MIMO sharded optimizer
# state (2-rank and 8-rank variants of the non-colocated grid layout).
# CPU-only; no GPUs needed.  Exercises the REAL torch_dist save/load path
# (megatron.core.dist_checkpointing) with sharding-integrity validation on
# the combined images+language optimizer state.  Run from the FlagScale repo
# root: ./tests/run_mimo_checkpointing_gloo_smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
SMOKE="tests/unit_tests/gloo_smoke_mimo_checkpointing.py"

run_smoke() {
    local world="$1"
    local port="$2"
    echo "==> world ${world} Gloo MIMO checkpoint smoke"
    torchrun \
        --nproc-per-node "${world}" \
        --master-port "${port}" \
        --standalone \
        "${SMOKE}" --world-size "${world}"
    echo "==> world ${world} Gloo MIMO checkpoint smoke PASSED"
}

run_smoke 2 29557
run_smoke 8 29558

echo "==> all MIMO checkpoint Gloo smokes PASSED"

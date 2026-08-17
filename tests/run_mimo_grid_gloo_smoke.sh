#!/usr/bin/env bash
# Copyright (c) 2026, BAAI. All rights reserved.

# Gloo process-group smoke for the non-colocated grid MIMO infra (2-rank and
# 8-rank variants).  CPU-only; no GPUs needed.  Run from the FlagScale repo
# root: ./tests/run_mimo_grid_gloo_smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
SMOKE="tests/unit_tests/gloo_smoke_mimo_grid.py"

run_smoke() {
    local world="$1"
    local port="$2"
    echo "==> world ${world} Gloo grid smoke"
    torchrun \
        --nproc-per-node "${world}" \
        --master-port "${port}" \
        --standalone \
        "${SMOKE}" --world-size "${world}"
    echo "==> world ${world} Gloo grid smoke PASSED"
}

run_smoke 2 29555
run_smoke 8 29556

echo "==> all grid Gloo smokes PASSED"

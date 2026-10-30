#!/usr/bin/env bash
set -euo pipefail

# Run both FlagScale inference entries with one matched workload. Timing is
# collected by inference_diffusion.py through InferenceEngine.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HORIZONKV_ENV:-$ROOT/.venvs/horizonkv-worker}/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DURATION_SECONDS="${DURATION_SECONDS:-30}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURE_RUNS="${MEASURE_RUNS:-1}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
PROMPT="${PROMPT:-A cinematic shot of a red fox walking slowly through a sunlit green meadow, natural motion, realistic fur, stable camera.}"
GPU="${HORIZONKV_GPU:-0}"
ASSETS="${HORIZONKV_ASSETS:-$ROOT/models/horizonkv}"
CHECKPOINT="${KREA_CHECKPOINT:-$ASSETS/checkpoints/krea/krea-realtime-video-14b.safetensors}"

[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid RUN_ID" >&2; exit 1; }
[[ "$DURATION_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "DURATION_SECONDS must be positive" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Worker Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$ASSETS" ]] || { echo "HORIZONKV_ASSETS not found: $ASSETS" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "KREA_CHECKPOINT not found: $CHECKPOINT" >&2; exit 1; }

RUN_ROOT="$ROOT/benchmarks/efficiency_${DURATION_SECONDS}s/$RUN_ID"
LOG_ROOT="$ROOT/logs/efficiency_${DURATION_SECONDS}s/$RUN_ID"
[[ ! -e "$RUN_ROOT" && ! -e "$LOG_ROOT" ]] || { echo "Run already exists: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

# 24 FPS output, three latent frames per block, and the shared six-pixel crop.
LATENT_FRAMES="${LATENT_FRAMES:-$((3 * ((DURATION_SECONDS * 24 + 17) / 12)))}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HORIZONKV_ASSETS="$ASSETS" KREA_CHECKPOINT="$CHECKPOINT"
export HORIZONKV_GPU="$GPU" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
ACTION=test
[[ "${DRY_RUN:-0}" == "1" ]] && ACTION=dryrun

result=0
for case in horizonkv ar_diffusion_baseline; do
  config_name=inference
  [[ "$case" == "ar_diffusion_baseline" ]] && config_name=inference_baseline
  case_root="$RUN_ROOT/$case"
  case_log="$LOG_ROOT/$case"
  mkdir -p "$case_root" "$case_log"
  echo "[efficiency] case=$case duration=${DURATION_SECONDS}s latent_frames=$LATENT_FRAMES"
  if ! "$PYTHON_BIN" -m flagscale.cli run \
      -p "$ROOT/examples/horizonkv/conf" -n "$config_name" -a "$ACTION" \
      "inference.engine.model=$ASSETS" \
      "inference.engine.results_path=$case_root" \
      "inference.generate.latent_frames=$LATENT_FRAMES" \
      "inference.generate.height=$HEIGHT" \
      "inference.generate.width=$WIDTH" \
      "inference.generate.prompt='$PROMPT'" \
      "inference.generate.benchmark=true" \
      "inference.generate.warmup=$WARMUP_RUNS" \
      "inference.generate.runs=$MEASURE_RUNS" \
      "inference.generate.duration_seconds=$DURATION_SECONDS" \
      "inference.generate.save=true" \
      "inference.generate.profile=true" \
      "inference.generate.output_type=np" \
      "inference.generate.return_latents=false" \
      >"$case_log/runner.log" 2>&1; then
    tail -40 "$case_log/runner.log" >&2
    result=1
  elif [[ "$ACTION" == "dryrun" ]]; then
    echo "[efficiency] case=$case config=valid"
  elif [[ ! -f "$case_root/benchmark.json" ]]; then
    echo "Missing benchmark result: $case_root/benchmark.json" >&2
    result=1
  else
    "$PYTHON_BIN" - "$case_root/benchmark.json" "$case" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
print(f"[efficiency] case={sys.argv[2]} median_e2e={d['median_end_to_end_ms']:.0f}ms runs={d['runs']}")
PY
  fi
done

if [[ "$result" -eq 0 && "$ACTION" != "dryrun" ]]; then
  "$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
h = json.loads((root / "horizonkv/benchmark.json").read_text())
b = json.loads((root / "ar_diffusion_baseline/benchmark.json").read_text())
hm, bm = h["median_end_to_end_ms"], b["median_end_to_end_ms"]
speedup = bm / hm
report = {"horizonkv_median_e2e_ms": hm, "baseline_median_e2e_ms": bm,
          "speedup": speedup, "latency_reduction_percent": (1 - hm / bm) * 100}
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"[efficiency] speedup={speedup:.3f}x latency_reduction={report['latency_reduction_percent']:+.2f}%")
PY
fi
exit "$result"

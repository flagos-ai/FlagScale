# Perf Monitor on Metax C550

## Scope

This note documents the current `perf_monitor` path for Metax C550 on `main-legacy`.

Important:

- The current runnable branch is `main-legacy`.
- The active code path is the legacy runner:
  - `flagscale/runner/runner_train.py`
- The new runner launcher path is not active in this branch.

## Current Code Path

- Runner integration:
  - `flagscale/runner/runner_train.py`
  - `run.py`
  - `flagscale/runner/auto_tuner/tuner.py`
- Monitor service:
  - `flagscale/runner/elastic/monitor_launcher.py`
  - `flagscale/runner/elastic/monitor_service.py`
  - `flagscale/runner/elastic/diagnostic.py`

## Metax-Specific Notes

- This monitor path is mostly process/log based and does not depend on `nvidia-smi`.
- Metax-specific diagnostic keywords were added for:
  - `maca out of memory`
  - `mxkw`
  - `ioctl create queue block timeout`

## Compatibility Aliases

For convenience, the legacy monitor path also accepts:

- `++experiment.runner.enable_perf_monitor=true`
- `++experiment.runner.perf_monitor_interval=5`

These are mapped internally to the legacy keys:

- `enable_perf_monitor` -> `enable_monitoring`
- `perf_monitor_interval` -> `monitor_interval`

## Known Pitfalls

- In the legacy runner, monitor enablement must be propagated to each node. This path was fixed in `runner_train.py`; do not bypass it with custom launch wrappers.
- Use a fresh `exp_dir` and a missing `checkpoint.load` during validation to avoid resume mismatches.
- Validate this first on the mini Aquila config, not on the original full 7B config.

## Smoke Test

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy

TS=$(date +%Y%m%d_%H%M%S)

python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=test \
  experiment.exp_dir=/workspace/exp/aquila_perf_smoke_${TS} \
  train.system.checkpoint.load=/workspace/exp/__no_ckpt__/does_not_exist \
  train.system.checkpoint.save=/workspace/exp/aquila_perf_smoke_${TS}/checkpoints \
  train.system.use_flash_attn=false \
  train.model.attention_backend=unfused \
  train.model.num_layers=8 \
  train.model.hidden_size=1024 \
  train.model.num_attention_heads=16 \
  train.model.seq_length=512 \
  train.model.max_position_embeddings=512 \
  train.model.multiple_of=128 \
  train.model.micro_batch_size=1 \
  train.model.global_batch_size=8 \
  train.model.train_samples=16 \
  ++experiment.runner.enable_perf_monitor=true \
  ++experiment.runner.perf_monitor_interval=5
```

## Expected Result

- The short training run completes successfully.
- Monitor outputs are written under:

```bash
/workspace/exp/aquila_perf_smoke_${TS}/logs/monitor
```

Typical files:

- `status.log`
- `host_*_diagnostic.txt`
- `host_*_current.log`

## Full Run Example

```bash
TS=$(date +%Y%m%d_%H%M%S)

python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=run \
  experiment.exp_dir=/workspace/exp/aquila_perf_run_${TS} \
  train.system.checkpoint.load=/workspace/exp/__no_ckpt__/does_not_exist \
  train.system.checkpoint.save=/workspace/exp/aquila_perf_run_${TS}/checkpoints \
  train.system.use_flash_attn=false \
  train.model.attention_backend=unfused \
  train.model.num_layers=8 \
  train.model.hidden_size=1024 \
  train.model.num_attention_heads=16 \
  train.model.seq_length=512 \
  train.model.max_position_embeddings=512 \
  train.model.multiple_of=128 \
  train.model.micro_batch_size=1 \
  train.model.global_batch_size=8 \
  train.model.train_samples=1600 \
  ++experiment.runner.enable_perf_monitor=true \
  ++experiment.runner.perf_monitor_interval=5
```

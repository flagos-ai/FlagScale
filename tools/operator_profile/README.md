# Training Operator Profiling

This tool extracts operator-to-kernel reports from a FlagScale training
timeline. Training only records the PyTorch profiler trace; report generation
is performed offline so that profiling does not add report-processing work to
the training process.

Run the following commands from the FlagScale repository root.

## Quick Start

### 1. Enable profiling

Add the following settings to the training configuration. Keep the model,
precision, dataset, batch size, sequence length, and other workload settings
unchanged between scenarios.

```yaml
system:
  logging:
    tensorboard_dir: /workspace/train_profile/tensorboard
  profiler:
    profile: true
    use_pytorch_profiler: true
    profile_ranks: [0]
    profile_step_start: 5
    profile_step_end: 10
    pytorch_profiler_collect_shapes: true
    pytorch_profiler_collect_memory: false
    pytorch_profiler_collect_callstack: false

model:
  train_iters: 15
```

`profile_ranks` contains global ranks. `[0]` is recommended for the standard
coverage report; an empty list profiles every rank. Select steady-state steps
and keep warmup or initialization outside the measured window.

### 2. Run training

Use the normal command for the target model, for example:

```bash
flagscale train qwen3 --config ./examples/qwen3/conf/train.yaml
```

The trace is written next to the configured TensorBoard directory:

```text
/workspace/train_profile/torch_profile/rank-0.json.gz
```

Only the timeline is generated during training. CSV and JSON reports are
created in the next offline step.

### 3. Extract one rank offline

```bash
python3 \
  flagscale/train/megatron/training/profiler_reports.py \
  --trace /workspace/train_profile/torch_profile/rank-0.json.gz \
  --output-dir /workspace/train_profile/results
```

If the trace filename does not contain `rank-N`, pass the rank explicitly:

```bash
python3 \
  flagscale/train/megatron/training/profiler_reports.py \
  --trace /path/to/profile.json.gz \
  --rank 0 \
  --output-dir /workspace/train_profile/results
```

## Output Files

For each rank, the extractor writes the same four core report types used by
inference, with a `rank-N_` prefix:

- `rank-N_operator_list.csv`: normalized operator inventory, operator kind,
  stable identity, related kernels, and each row's share of total kernel time.
- `rank-N_kernel_time.csv`: operator/kernel call counts, accumulated GPU time,
  and time percentage.
- `rank-N_kernel_shape_dtype.csv`: operator/kernel mapping variants with input
  shapes, dtypes, mapping status, event count, and time.
- `rank-N_summary.json`: trace scope, CPU/GPU activity statistics, mapping
  coverage, and conservation checks.

Training-specific hierarchy fields such as `custom_operator` and
`execution_operator` are retained after the common inference-compatible
columns.

Communication kernels are included in the common reports rather than written
to a separate communication file. Unattributed kernels are retained with an
explicit mapping status. The percentage denominator is the sum of all GPU
kernel durations in the selected rank and profiling window; CPU time,
`gpu_memcpy`, and `gpu_memset` are not included in that denominator.

Every Boolean check under `summary.json.conservation` must be `true`.

## Extract Multiple Ranks

To process every `rank-N.json` or `rank-N.json.gz` trace in a directory:

```bash
python3 \
  flagscale/train/megatron/training/profiler_reports.py \
  --trace-dir /workspace/train_profile/torch_profile \
  --output-dir /workspace/train_profile/results
```

In addition to the four per-rank files, directory mode writes these
training-specific aggregate reports:

- `cross_rank_operator_summary.csv`
- `cross_rank_kernel_report_summary.json`

To rebuild only the cross-rank reports from existing per-rank CSV files:

```bash
python3 \
  flagscale/train/megatron/training/profiler_reports.py \
  --aggregate-profile-dir /workspace/train_profile/results
```

## Recommended Coverage Scenarios

Use three runs with an otherwise identical workload:

| Scenario | `te_fl_prefer` | `enable_flag_gems` | Purpose |
|---|---|---:|---|
| Vendor baseline | `vendor` | `false` | Coverage denominator |
| FlagOS backend | `flagos` | `false` | FlagOS backend comparison |
| FlagOS + FlagGems | `flagos` | `true` | Full FlagOS operator evidence |

Coverage is calculated from normalized, deduplicated operator kinds. It is not
weighted by call count or execution time. Kernel time percentage is provided
only to help prioritize optimization work.

FlagOS numerator entries must have auditable evidence that they are provided
or handled by FlagGems, FlagTree, or FlagCX. Communication belongs to the
denominator but enters the numerator only when FlagCX takeover is confirmed.

## Troubleshooting

- **No trace is generated:** confirm that `profile`,
  `use_pytorch_profiler`, and the target global rank are enabled, and that
  `train_iters` runs beyond `profile_step_end`.
- **Shape or dtype is `null`:** graph replay and some runtime kernels do not
  expose complete tensor metadata. The event remains in the report and is
  identified by `mapping_status`.
- **A conservation check is false:** do not use the report for coverage
  calculation. Preserve the original trace and investigate the failed check.
- **The output directory already contains results:** use a separate stable
  directory for each scenario to prevent baseline and FlagOS reports from
  being mixed.

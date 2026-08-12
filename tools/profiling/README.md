# Hipprof Step Profiling

FlagScale can start one hipprof session for each selected global rank and collect only the
configured training-step window.

## Configuration

Add the hipprof executable and output root to the runner configuration:

```yaml
experiment:
  runner:
    hipprof_bin_path: /opt/dtk/bin/hipprof
    hipprof_output_dir: ${experiment.exp_dir}/hipprof_report
```

Enable step profiling in the model configuration:

```yaml
train:
  model:
    profile: true
    use_hipprof_profiler: true
    profile_step_start: 5
    profile_step_end: 6
    profile_ranks: [0]
```

`hipprof_bin_path` accepts either the executable itself or its containing directory. Both runner
options are required. Nsys, the PyTorch profiler, and hipprof must not be enabled together.

The launcher automatically configures torchrun to use the hipprof Python wrapper. Users do not
need to set `PYTHON_EXEC` or session environment variables.

## Output

Each selected rank writes to:

```text
<output-root>/<host>_global<rank>_local<rank>_pid<pid>/
```

The hipprof result prefix is `result` inside that directory.

## Defaults and overrides

The default trace set is `HIP,RCCL,HSA`. Stream grouping is enabled and the segment size is 50000.
Advanced users can override these values under `experiment.envs`:

```yaml
experiment:
  envs:
    HIPPROF_TRACE: HIP,RCCL,HSA
    HIPPROF_GROUP_STREAM: 1
    HIPPROF_SEGMENT_SIZE: 50000
    HIPPROF_REAL_PYTHON: /path/to/python
```

Ranks outside `profile_ranks` run with the normal Python interpreter and do not start hipprof.

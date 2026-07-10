# Hipprof YAML Step Profiling Design

## Goal

Make hipprof step profiling usable through the same FlagScale YAML workflow as the existing
Nsight Systems integration. Users should not need to set `PYTHON_EXEC`, construct session IDs,
or invoke the Python wrapper manually.

The supported configuration is:

```yaml
experiment:
  runner:
    hipprof_bin_path: /opt/dtk/bin/hipprof
    hipprof_output_dir: ${experiment.exp_dir}/hipprof_report

train:
  model:
    profile: true
    use_hipprof_profiler: true
    profile_step_start: 5
    profile_step_end: 6
    profile_ranks: [0]
```

This design targets the default V1 SSH launcher, matching the scope of the current nsys launcher
integration. Extending the legacy runner or adding real-hardware CI is outside this change.

## Chosen Approach

The SSH launcher consumes the two hipprof runner options and automatically configures torchrun to
use `tools/profiling/hipprof_python_wrapper.sh` as its per-rank Python executable. It does this by
injecting `PYTHON_EXEC`, `HIPPROF_BIN_PATH`, and `HIPPROF_OUTPUT_DIR` into the generated command.

Using torchrun's per-rank executable hook is preferable to prefixing hipprof around the entire
torchrun process. Hipprof needs an independent session and output directory for each selected
global rank, while nsys can profile the node-level torchrun process tree.

The alternatives rejected are:

1. Replacing the torchrun training entrypoint and using `--no-python`. This changes the existing
   entrypoint contract and complicates argument forwarding.
2. Prefixing hipprof directly before torchrun. This does not provide reliable per-rank session
   isolation.

## Launcher Behavior

The launcher will:

1. Remove `hipprof_bin_path` and `hipprof_output_dir` from torchrun arguments because they are
   launcher-owned settings.
2. Require both options when `train.model.use_hipprof_profiler` is enabled.
3. Require `train.model.profile: true` so step start and stop callbacks are active.
4. Reject simultaneous nsys and hipprof launcher configuration to avoid nested system profilers.
5. Accept either the full hipprof executable path or a directory containing `hipprof`, matching
   the nsys path convention.
6. Set `PYTHON_EXEC=tools/profiling/hipprof_python_wrapper.sh` for torchrun workers.
7. Preserve a user-supplied `PYTHON_EXEC` as `HIPPROF_REAL_PYTHON` unless the latter is already
   explicitly configured.

The generated wrapper path is relative to the FlagScale package root. Generated run scripts
already change to that root before executing the torchrun command, including node-specific build
directories.

## Wrapper Behavior

For ranks outside `profile_ranks`, the wrapper directly `exec`s the real Python interpreter.

For selected ranks, the wrapper:

1. Resolves the real Python executable and hipprof executable.
2. Generates a unique session ID from hostname, global rank, and wrapper PID.
3. Creates a unique output directory from hostname, global rank, local rank, and PID.
4. Builds the requested trace, grouping, and segmentation arguments.
5. Executes hipprof with `--trace-off` so no startup data is collected.
6. Uses `exec` rather than a background child, so torchrun signals and exit status apply directly
   to hipprof and its wrapped training process.

The fixed warmup sleep, initial best-effort `--stop`, EXIT trap, and background `wait` are removed.
Training remains responsible for sending session `--start` and `--stop` at the configured steps.

## Training Configuration and Session Control

`use_hipprof_profiler` remains part of `ProfilingConfig`. The duplicate model-level
`hipprof_bin_path` and `hipprof_session_id` fields are removed because the launcher and wrapper are
the single source of truth.

The training process reads `HIPPROF_SESSION_ID` and `HIPPROF_BIN_PATH` exported by the wrapper.
Missing session configuration produces an error that points users to the YAML runner settings.
The existing mutual exclusion between PyTorch profiler and hipprof remains in place.

## Error Handling

Configuration errors fail before torchrun starts:

- only one of the two hipprof runner options is set;
- hipprof runner options are present without `profile` and `use_hipprof_profiler`;
- nsys and hipprof launcher options are enabled together.

Runtime session-control failures remain fatal and include hipprof stdout and stderr. This prevents
one or more ranks from silently continuing with an invalid profiling window.

## Tests

Tests use fake hipprof and Python executables and do not require DTK or accelerator hardware.

Launcher tests cover:

- stripping hipprof-only runner keys from torchrun arguments;
- automatic per-rank wrapper environment injection;
- executable-directory normalization;
- preservation of a custom real Python executable;
- incomplete, disabled, and nsys-conflicting configurations.

Wrapper tests cover:

- bypassing hipprof on unselected ranks;
- selected-rank launch with `--trace-off`;
- equality between the launched session ID and the ID visible to training;
- trace and output argument forwarding;
- direct `exec` process/signal behavior without an orphan wrapper child.

Targeted pytest, Bash syntax checking, Python compilation, and `git diff --check` form the simulated
verification suite. A real DTK hipprof run remains a follow-up hardware validation step.

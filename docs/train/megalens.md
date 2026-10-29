# MegaLens training traces

This integration requires a Megatron-LM-FL build containing
[MegaLens PR #104](https://github.com/flagos-ai/Megatron-LM-FL/pull/104).
The Core observability facade and `megatron.megalens.runtime` must be available
from that build. Released builds without these interfaces cannot use this integration.
The current integration is based on FlagScale main
`f8e9a62c499d526a6ffa47f533bcf342330b7326`.

Tracing is disabled by default. Enable it in the training model configuration:

```yaml
train:
  model:
    trace: true
    trace_mode: 1
    trace_dir: /path/to/trace-output
    trace_interval: 10
    continuous_trace_iterations: 2
    trace_granularity: full
    trace_cupti_kernels: "off"
```

Mode 0 records the lightweight sentinel; mode 1 records framework scopes.
Mode 1 retains MegaLens's WORLD barrier at each observed training iteration start.
Kernel collection (`trace_cupti_kernels: on`) and hardware monitoring
(`hardware_monitor: true`) are optional. Kernel collection cannot nest the
Megatron PyTorch profiler. CUDA Graph runs expose graph-external framework scopes;
Python scopes inside managed capture or replay are unavailable.

Skipped iterations close the current trace window. FlagScale range skips retain
their native data and iteration advancement, and the next trace iteration uses
the advanced iteration number. Tracing with `skip_samples_range` and
`step_batch_size_schedule` is rejected because the next global batch size cannot
be previewed without changing the training state. Use `skip_iters_range` or a
fixed global batch size for this combination.

Normal completion closes the runtime gracefully. Training failures and global
teardown close it without requiring all ranks to reach a trace-export collective.
When tracing is disabled, runtime construction and the MegaLens runtime import
are skipped. Core supplies the lightweight scope facade.

The independent dense/packed batch broadcast correction belongs in its own PR;
this integration does not modify `get_batch_on_this_tp_rank`.
Use the [MegaLens guide](https://github.com/Chlience/Megatron-LM-FL/blob/feature/megalens-cuda/docs/user-guide/features/megalens.md)
for capture output and analysis commands.

CPU tests cover configuration validation, runtime ownership, native iteration
statements with training/transport stubs, skip boundaries, error cleanup, and
mode-1 alignment barriers. This revision has not been validated in GPU/NCCL
training, CUDA Graph replay, or a new combined FlagScale work image.

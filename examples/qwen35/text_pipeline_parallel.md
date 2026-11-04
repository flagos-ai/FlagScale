# Qwen3.5 text training with pipeline parallelism

The text entry point (`--no-enable-vision`, or `no_enable_vision: true` in YAML)
retains native `position_ids` on every physical pipeline stage when all of the
following apply:

- `position_embedding_type: mrope` and `pipeline_model_parallel_size > 1`;
- dense text batches (`sft: false`) and `context_parallel_size: 1`;
- no Hybrid CP, MTP, virtual pipeline parallelism or DualPipeV.

No additional configuration flag is needed for this path. Each stage's TP source
rank builds the native GPT dataset and consumes the same DP sample stream. The
existing Megatron TP helper broadcasts a complete batch within each TP group.
The entry then keeps tokens only on the first stage and targets only on the last,
while retaining positions on every stage. Text positions are expanded to the
existing `[3, batch, sequence]` mRoPE format, including any document-boundary resets.

The helper's complete-batch field selection does not change the actual TP/PP
process groups, layer partition or pipeline schedule. It temporarily transfers
input and target tensors on stages that later discard them; this follows the
validated replicated-batch workaround rather than introducing a new PP metadata
communication protocol. Each PP stage needs access to the same dataset files,
configuration and resume state.

PP1 and the other modes listed above retain their existing batch paths. This
change does not establish support for previously unsupported CP/SFT/MTP or
interleaved scheduling combinations.

The ten-field GPT batch interface adaptation is tracked separately in
[FlagScale #1317](https://github.com/flagos-ai/FlagScale/pull/1317). Retaining PP
positions addresses a separate failure: a valid ten-field batch may still contain
`position_ids=None` on middle or last stages, which mRoPE cannot consume.
[FlagScale #1247](https://github.com/flagos-ai/FlagScale/pull/1247) contains related
middle-stage handling for the older API; the current path also covers last-stage
TP receivers without modifying Megatron source files.

## CPU regression checks

With PyTorch, pytest and Megatron-LM-FL available in the environment:

```bash
python -m pytest -q tests/unit_tests/train/megatron/test_qwen35_pipeline_positions.py
```

The checks execute the actual FlagScale batch functions and installed Megatron TP
helper with CPU tensors. A broadcast replay replaces distributed transport and
checks matching shapes, dtypes and collective order on TP source/receiver ranks.
Coverage includes PP1/2/4, TP2/4, multiple DP streams and microbatches, native
document-reset positions, attention-mask presence, dataset initialization before
configuration conversion, and the unchanged default GPT path.

These checks do not replace a distributed GPU training or checkpoint-resume run.

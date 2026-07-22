# Qwen3.5 Checkpoint Converter

This directory provides a unified entry point for bidirectional Qwen3.5 checkpoint conversion between HuggingFace and Megatron.

## Main Files

| File/Directory | Description |
| --- | --- |
| [convert_qwen35.py](convert_qwen35.py) | Unified conversion entry point, supports `hf2meg` and `meg2hf` |
| [run.sh](run.sh) | Shell wrapper for `convert_qwen35.py` |
| [qwen35/](qwen35/) | Shared conversion library (Config, GDN/Attention/Vision/MTP/MLP, TP/PP/EP sharding, IO, validation) |

## Usage

### HF → Megatron

```bash
./run.sh hf2meg \
    --yaml /workspace/FlagScale/examples/qwen35/conf/train/4b_nv_baseline.yaml \
    --hf-path Qwen/Qwen3.5-4B \
    --meg-path /path/to/output \
    [--ref-path /path/to/ref/megatron]
```

For `hf2meg`, `--hf-path` can be either a local HF checkpoint directory or a ModelScope model ID (e.g. `Qwen/Qwen3.5-4B`). When the path is not found locally, the converter automatically downloads it from ModelScope.

### Megatron → HF

```bash
./run.sh meg2hf \
    --yaml /workspace/FlagScale/examples/qwen35/conf/train/4b_nv_baseline.yaml \
    --meg-path /path/to/megatron/checkpoint \
    --hf-path /path/to/output \
    [--ref-path /path/to/ref/hf]
```

### Direct Python Invocation

```bash
python convert_qwen35.py --direction hf2meg --hf-path ... --meg-path ... --yaml ...
python convert_qwen35.py --direction meg2hf --meg-path ... --hf-path ... --yaml ...
```

## Parameters

- `--direction {hf2meg,meg2hf}`: **required**, conversion direction
- `--hf-path PATH|ID`: **required**, HF checkpoint directory (input for hf2meg, output for meg2hf). For `hf2meg`, a ModelScope model ID such as `Qwen/Qwen3.5-4B` is also accepted; the checkpoint will be downloaded automatically if it is not found locally.
- `--meg-path PATH`: **required**, Megatron checkpoint directory (output for hf2meg, input for meg2hf)
- `--yaml PATH`: **required**, training config yaml (provides TP/PP/EP and model structure)
- `--ref-path PATH`: optional, reference checkpoint path for post-conversion comparison
- `--tp N` / `--pp N` / `--ep N`: optional, override parallel sizes from yaml
- `--adjust-embedding`: during hf2meg, adjust vocab size to match the reference checkpoint
- `--adjust-ln`: enable legacy layer norm adjustment (off by default; usually not needed for Qwen3.5)

## Output Format

- **Megatron**: `{save_dir}/release/mp_rank_*/model_optim_rng.pt`, plus `latest_checkpointed_iteration.txt`
- **HF**: `{save_dir}/model.safetensors`

## Validation

- If `--ref-path` is provided, the converter automatically compares keys and shapes with the reference checkpoint
- Use `compare_two_ckpts.py` for stricter per-tensor value comparison:

```bash
python compare_two_ckpts.py --ref /path/to/ref/release --gen /path/to/gen/release --tp 2
```

## WARNING

If torch version < 2.9, maybe need modify (tools/checkpoint/qwen35/qwen35/config.py)

```python
self.use_linear_proj = cfg.get("vision_patch_embed_linear", True)
```

to

```python
self.use_linear_proj = cfg.get("vision_patch_embed_linear", False)
```

because qwen_vl will use linear instead of conv3d when torch version >= 2.9.

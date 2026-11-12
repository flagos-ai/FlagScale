# HF ↔ MCore DCP Conversion

Convert model weights between HF safetensors and MCore DCP. Run from the repository root in a single process (world size 1, rank 0).

The bundled model profile is `tools/checkpoint/deepseek_v4/mapping.yaml` for DSv4.
It explicitly composes shared structures and their argument groups; unselected
structures contribute no fields or rules. Additional model profiles, including
Qwen integrations, are separate changes and are not bundled with this release.

The shared catalog includes GQA, GDN, dense/MoE, vision and MTP layouts for custom
profiles. GPT/GQA YaRN uses the `yarn_gqa` group and native `yarn_*` parameters;
the separate `yarn` group describes MLA parameters. A model adapter must supply
any configuration conventions required by its selected groups.

## HF → MCore

The source needs `config.json` and safetensors weights, with an index for sharded weights.

```bash
python -m tools.checkpoint.hf_mcore_dcp.convert hf-to-mcore \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --source /models/dsv4-hf \
  --output /outputs/dsv4-mcore \
  --dtype fp32
```

HF imports default to **FP32**: dense floating weights are promoted without loss,
and FP8 E4M3 / MXFP4 inputs are decoded using their source scales in FP32.
Integer tables retain their dtype. This avoids an intermediate BF16/FP16 rounding
step; it cannot recover information already lost in the HF quantization.
`--dtype auto` retains dense source dtypes and still decodes quantized inputs to
FP32. Use `--dtype bf16` or `fp16` only when an explicit precision reduction is intended.

Outputs contain ordinary dense DCP tensors and empty `_extra_state` placeholders
in MCore's standard ShardedObject format. HF FP8/MXFP4 payloads are decoded using
their source scales; packed TE payloads and source scales are not stored.

Training selects the low-precision recipe and parameter type. MCore's native
`force_all_tensors_to_non_fp8` load staging reads weights into dense tensors;
TE's `load_state_dict` / `copy_` then quantizes them into the initialized target
parameters. This converter has no TE dependency, recipe/version options, or
custom training loading adapters. Delayed-scaling recipe state and calibration
history are not supported or generated.

Saving FP32 weights does not force an FP32 destination model or optimizer
initialization. GPU numerical correctness and TP/EP resharding must be validated
with the target training configuration.

## MCore → HF

The source can be a checkpoint root or a completed iteration directory.
**This release requires `--hf-reference` for every MCore → HF export, including dry runs.**

```bash
python -m tools.checkpoint.hf_mcore_dcp.convert mcore-to-hf \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --source /checkpoints/dsv4/iter_0000100 \
  --hf-reference /models/dsv4-hf \
  --output /outputs/dsv4-export
```

`--hf-reference` supplies the original HF `config.json` and tokenizer assets;
reference weights are not required or read. Its optional index supplies the adapter
with source key inventory. Exported weight names always come from the mapping,
including when the reference contains no index. This release copies root-level
configuration/tokenizer files, including the default `chat_template.jinja`, unchanged;
nested assets such as named templates in `chat_templates/` are outside its initial scope.

HF import also saves the root-level non-weight assets in `iter_0000000/hf_assets`.
You may explicitly pass that directory as `--hf-reference`. A saved snapshot or asset
directory does not waive the reference requirement. Reference-free export is deferred
to a later revision after automatic configuration and asset recovery is supported.

If the source retains the converter's `hf_export.json`, `--mapping` can be omitted. Training saves may not retain this file.

Vocabulary conversion preserves the complete embedding and output-head row counts,
including padding. The shared embedding rule derives HF `vocab_size` and MCore
`vocab_size`, `padded_vocab_size`, and `actual_vocab_size` from tensor metadata;
saved args, reference config, and tokenizer length do not override these dimensions.
The output head must agree with the embedding vocabulary dimension. DSv4 hash-router
tables also retain all rows. There is no `--vocab-size` override or automatic cropping.
An existing `make_vocab_size_divisible_by` is copied through the configuration map;
the converter does not calculate or invent one. Training users choose TP/alignment
settings compatible with the explicitly saved `padded_vocab_size`.

Export preserves actual source dtypes by default. Use `--dtype fp32` to promote dense floating weights.

## Options

| Option | Usage |
| --- | --- |
| `--dry-run` | Print the metadata-based plan without writing weights. Value checks run during conversion. |
| `--dtype` | `fp32` by default for HF import; `auto` by default for export. Auto preserves dense dtypes and decodes quantized imports to FP32. Explicit `bf16`/`fp16` casts retain existing FP32 and integer tensors. |
| `--device` | Decode on `cpu` (default) or `cuda[:N]`. |
| `--max-shard-size 4GB` | Target output shard size; individual regions may exceed it. |
| `--overwrite` | Replace an existing output only after conversion succeeds. Input/output directories must not overlap. |
| `--mapping-override custom.yaml` | Apply YAML/JSON overrides; repeat to apply multiple files in order. |

Overrides replace complete rules by `id`; new IDs add rules, and `{id: name, disabled: true}` removes a rule. Unmapped source tensors require explicit `ignore` entries. Rule `hf` names must be the intended serialized HF output names. Optional input-name normalization maps source aliases onto these names; no output naming snapshot or inverse alias inference is used.

To export a self-contained, resolved mapping from a profile:

```bash
python -m tools.checkpoint.hf_mcore_dcp.convert export-mapping \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --output /tmp/dsv4-mapping.json
```

Mappings use **schema_version: 2**. The previous schema, `extends`, and Python mapping-dictionary imports are not supported. A checkpoint whose saved mapping uses an earlier schema requires an explicit current `--mapping`. Conversion executes declared layouts; it does not verify model equivalence.

Adapters may implement `tensor_transforms(configs, hf_names, forward)` and return
a dictionary of logical HF names to named, non-mutating tensor functions.
These functions change values without changing shape or dtype; the common
executor calls them before import layouts or after export layouts. Their names
and affected tensors appear in `tensor_value_transforms` in the conversion
report, including dry runs. They run after an explicit dense dtype cast (FP32 inputs remain FP32). With
`--dtype auto`, arithmetic retains each tensor's dense dtype; norm offsets can
therefore round in BF16/FP16 and do not guarantee bit-exact roundtrips.
`--dtype fp32` increases arithmetic and output precision but does not recover
rounding already present in a source checkpoint.

## Shared YAML catalogs

See the [schema reference (中文)](schema/README.md) for field formats, expressions,
scope expansion, override semantics, adapter hooks, and a complete model example.

- [parameters.yaml](schema/parameters.yaml) defines named weight structures: embeddings,
  output heads, norms, GQA, gated GQA, dense SwiGLU, MLA projections, MoE/shared
  experts, GDN, compressed attention, sparse indexers, hyper-connections and MTP
  streams, vision blocks and patch mergers. Each structure declares its `rules`
  and required `args` groups.
- [args.yaml](schema/args.yaml) defines configuration field mappings, defaults and simple
  derived values. Only groups required by selected structures, plus the profile's
  explicit `args` groups, are applied.
- A model `mapping.yaml` selects structures, binds their names and scopes, and
  declares model identifiers or exceptional defaults. It does not copy complete
  weight inventories.
- An optional `adapter` module handles conventions that are not a tensor layout
  or a configuration expression. All hooks are optional: `normalize_config`,
  `finish_config`, `prepare_mapping`, `tensor_transforms`, and `export_config`.
  `export_config(config)` restores the serialized HF configuration envelope
  after compilation and output precision selection, without changing tensors.

This follows HybridModel's separation of precise component definitions from
model composition. It does not translate GPTModel layer numbering into
HybridModel numbering: each scope must name the actual source/target DCP keys.

For example, a dense decoder profile can be expressed as:

```yaml
schema_version: 2
profile: decoder
parameters: [embedding, output, final_norm, gqa, qk_norm, dense_swiglu]
args: [rope]
bindings:
  embedding: model.embed_tokens
  output: lm_head
  final_norm: model.norm
  attention: self_attn
  attention_output: o_proj
  attention_norm: input_layernorm
  mlp_norm: post_attention_layernorm
  mlp: mlp
  gate_proj: gate_proj
  up_proj: up_proj
  down_proj: down_proj
scopes:
  layer:
    - hf: model.layers.{layer}.
      mcore: decoder.layers.{layer}.
```

A heterogeneous profile can select precise structures on separate scopes:

```yaml
parameters:
  - use: gdn
    scope: linear_layers
  - use: gated_gqa
    scope: attention_layers
```

The profile or its `prepare_mapping` hook supplies those scopes. Each scope is
a list of HF/MCore prefix pairs; literal layer indices can select
an arbitrary layer arrangement. `{layer}` and `{expert}` remain numeric axes
expanded by the common compiler. No model is instantiated.

Structural name bindings are substituted before numeric axes are expanded:

```yaml
bindings:
  attention: self_attn
  attention_norm: input_layernorm
  mlp: mlp
  gate_proj: gate_proj
  up_proj: up_proj
  down_proj: down_proj
```

A `use` entry may override `scope`, `optional`, and `bindings` for that instance.
Defaults are declared once at the top of `parameters.yaml`. All supplied model
profiles declare serialized HF names directly. Optional `naming.strip_prefix`,
`naming.segments`, and `naming.replace` affect HF inputs only; export follows the
declared names regardless of snapshots or reference files.

Composition order is: selected structures and their argument groups, explicit
profile argument groups, profile overrides, then each `--mapping-override` in
order. `config_map`, `mcore_args`, and `hf_config` override by field; `rules`
override by complete rule `id`. Structure rule IDs must be unique within one
composition. Use one scope containing several prefix pairs to apply a structure
at multiple locations. Existing `generate`, `extra_state`, `collect` and
`zero_checks` remain generic operations; a model declares where they apply.

`export-mapping` and newly written `hf_export.json` contain expanded rules and
arguments. They can be used without reselecting structures. The source model's
complete HF config is retained for export, while translating an existing MCore
checkpoint only contributes fields from the selected argument groups.

To add a model built from existing structures, add a short profile and only the
necessary adapter hooks. Add a reusable layout or argument relationship to the
shared YAML catalogs when it is genuinely new. The DCP/HF storage code does not
need a model-specific branch.

## Source layout

```text
hf_mcore_dcp/
├── convert.py              # CLI, prepare/execute orchestration and adapter hooks
├── utils.py                # Naming, sizes, region checks, single-process checks, output commit
├── schema/
│   ├── parameters.yaml    # Reusable weight structures
│   ├── args.yaml          # Reusable configuration relationships
│   ├── mapping.py         # Structure composition and numeric scope expansion
│   └── config.py          # Configuration translation and expression evaluation
├── layout/
│   ├── compiler.py        # Shape inference, Piece/Profile definitions, region compilation
│   └── operators.py       # Reversible tensor layout operations
└── storage/
    ├── metadata.py        # HF index/header and MCore metadata inspection
    ├── hf.py              # HF tensor reads and sharded safetensors writes
    ├── dcp.py             # Streaming DCP reads/writes and MCore checkpoint packaging
    ├── precision.py       # Tensor descriptors, precision policy and HF scale handling
    └── codecs.py          # FP32 software decoding of HF FP8/MXFP4
```

Model profiles and adapters remain in their model directories. `convert.py`
composes schema rules, compiles layouts, and dispatches storage operations.
`utils.py` uses only the standard library; HF metadata inspection does not
import PyTorch. Package `__init__.py` files contain no dispatch logic; the root
package defines the shared `MappingError` exception.

FP8/MXFP4 decoding is confined to `storage/codecs.py`. The converter does not
import training adapters or TE, and writes no custom tensor subclasses.

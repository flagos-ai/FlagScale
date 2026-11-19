# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Rule-driven HF / MCore conversion with ordinary tensor checkpoint storage."""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import sys
from copy import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.checkpoint.hf_mcore_dcp import MappingError
from tools.checkpoint.hf_mcore_dcp.schema.config import translate_config
from tools.checkpoint.hf_mcore_dcp.schema.mapping import compose_mapping, load_mapping
from tools.checkpoint.hf_mcore_dcp.utils import require_world_one


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MappingError(f"Expected JSON object: {path}")
    return value


def _transform_tensor(name, tensor, transforms, policy):
    """Apply non-mutating model value transforms; mappings own shape changes."""
    if name not in transforms:
        return tensor
    import torch

    from tools.checkpoint.hf_mcore_dcp.storage.precision import DENSE_DTYPES

    # Cast before arithmetic so --dtype fp32 also increases transform precision.
    if not policy.preserve_source_dtypes and tensor.dtype != torch.float32:
        tensor = tensor.to(DENSE_DTYPES[policy.dtype])
    return transforms[name](tensor)


def prepare(args):
    if args.command == "mcore-to-hf" and not args.hf_reference:
        raise MappingError(
            "mcore-to-hf requires --hf-reference in this release; "
            "provide HF config and tokenizer assets (no weights required)"
        )

    from tools.checkpoint.hf_mcore_dcp.layout.compiler import (
        DTYPE_CODES,
        compile_mapping,
        compile_regions,
        plan_report,
    )
    from tools.checkpoint.hf_mcore_dcp.storage.dcp import (
        StreamingCheckpointReader,
        _groups,
        _storage_dtype,
    )
    from tools.checkpoint.hf_mcore_dcp.storage.hf import SafetensorReader
    from tools.checkpoint.hf_mcore_dcp.storage.metadata import (
        read_mcore_artifacts,
        resolve_iteration_dir,
    )
    from tools.checkpoint.hf_mcore_dcp.storage.precision import (
        DENSE_DTYPES,
        PrecisionSource,
        TensorSpec,
        resolve_precision,
    )
    from tools.checkpoint.hf_mcore_dcp.utils import normalize_name, parse_size

    require_world_one()
    forward = args.command == "hf-to-mcore"
    if args.dtype is None:
        args.dtype = "fp32" if forward else "auto"
    saved = {}
    if not forward:
        directory = resolve_iteration_dir(args.source)
        if (directory / "hf_export.json").is_file():
            saved = _json(directory / "hf_export.json")
    if args.mapping:
        mapping = load_mapping(args.mapping, args.mapping_override)
    elif saved.get("mapping"):
        mapping = compose_mapping(saved["mapping"])
        for path in args.mapping_override:
            mapping = load_mapping(path, _base=mapping)
    else:
        raise MappingError("Provide --mapping or a checkpoint containing hf_export.json")
    shard_size = parse_size(args.max_shard_size)
    if shard_size <= 0:
        raise MappingError("max-shard-size must be positive")
    adapter = importlib.import_module(mapping["adapter"]) if mapping.get("adapter") else None
    naming = mapping.get("naming", {})
    canonical = lambda name: normalize_name(name, naming)
    patterns = mapping.get("quantization", {}).get("mxfp4", [])
    if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
        raise MappingError("quantization.mxfp4 must be a list of tensor patterns")
    reader = None
    if forward:
        hf = _json(Path(args.source) / "config.json")
        ignored = mapping.get("ignore", {}).get("hf", [])
        original = PrecisionSource(
            SafetensorReader(args.source),
            hf,
            normalize=canonical,
            skip=lambda n: any(fnmatch.fnmatchcase(canonical(n), p) for p in ignored),
            mxfp4=lambda n: any(fnmatch.fnmatchcase(canonical(n), p) for p in patterns),
        )
        if adapter and hasattr(adapter, "normalize_config"):
            hf = adapter.normalize_config(hf, True, original.keys)
        mc = translate_config(mapping, hf, True)
        if adapter and hasattr(adapter, "finish_config"):
            mc = adapter.finish_config(mc, True)
        policy = resolve_precision(
            original, hf, args.dtype, args.device, to_mcore=True
        )
        original.dtype, original.device = DENSE_DTYPES[policy.dtype], policy.device
        specs = {
            key: replace(
                spec,
                dtype=DTYPE_CODES[str(original.dtype)],
                kind="dense",
                scale=None,
                scale_dtype=None,
            )
            if spec.kind != "dense"
            else spec
            for key, spec in original.specs.items()
        }
    else:
        artifacts = read_mcore_artifacts(args.source)
        mc = artifacts["args"]
        if adapter and hasattr(adapter, "normalize_config"):
            mc = adapter.normalize_config(mc, False, artifacts["model_tensors"])
        hf = dict(saved.get("hf_config", {}))
        reference = _json(Path(args.hf_reference) / "config.json")
        index = Path(args.hf_reference) / "model.safetensors.index.json"
        reference_names = {}
        if index.is_file():
            reference_names = {
                canonical(n): TensorSpec(n, (), "BF16") for n in _json(index)["weight_map"]
            }
        if adapter and hasattr(adapter, "normalize_config"):
            reference = adapter.normalize_config(reference, True, reference_names)
        hf.update(reference)
        hf.update(translate_config(mapping, mc, False))
        if adapter and hasattr(adapter, "finish_config"):
            hf = adapter.finish_config(hf, False)
        specs = {
            key: TensorSpec(
                key, tuple(item["shape"]), DTYPE_CODES.get(item["dtype"], item["dtype"])
            )
            for key, item in artifacts["model_tensors"].items()
        }
    effective = (
        adapter.prepare_mapping(mapping, {"hf": hf, "mcore": mc}, specs, forward)
        if adapter and hasattr(adapter, "prepare_mapping")
        else mapping
    )
    # Shape inference uses metadata only; storage contains ordinary tensors.
    plan = compile_mapping(
        effective, specs, {"hf": hf, "mcore": mc}, forward=forward, dtype=args.dtype
    )
    profile = compile_regions(plan, mapping.get("profile", "mapping"))
    if forward:
        source = copy(original)
        source.specs = {key: original.specs[key] for key in profile.shapes}
        source.physical_keys = {
            n for spec in source.specs.values() for n in (spec.name, spec.scale) if n
        }
        source.consumed = set()
        for pieces in _groups(profile).values():
            _storage_dtype(pieces, source, policy)
        zero_getter = original.get_tensor
    else:
        reader = StreamingCheckpointReader(args.source, profile, ignore=plan["ignored"])
        dtypes = reader.tensor_dtypes()
        output_specs = {
            name: replace(spec, dtype=DTYPE_CODES[str(dtypes[name])])
            for name, spec in plan["outputs"].items()
            if name in profile.shapes
        }
        output_specs.update(
            {name: spec for name, spec in plan["outputs"].items() if name not in profile.shapes}
        )
        source = SimpleNamespace(
            specs=output_specs,
            keys=set(output_specs),
            low_precision_kinds={s.kind for s in output_specs.values() if s.kind != "dense"},
        )
        policy = resolve_precision(
            source, hf, args.dtype, args.device, to_mcore=False
        )
        zero_getter = None
    transforms = (
        adapter.tensor_transforms({"hf": hf, "mcore": mc}, profile.shapes, forward)
        if adapter and hasattr(adapter, "tensor_transforms")
        else {}
    )
    if forward and transforms:
        getter = source.get_tensor
        source.get_tensor = lambda name: _transform_tensor(name, getter(name), transforms, policy)
    report = {
        "schema_version": 2,
        "direction": args.command,
        "profile": profile.name,
        "checkpoint_content": "model_only",
        "model_semantics_validated": False,
        "mapping": mapping,
        "configs": {"hf": hf, "mcore": mc},
        "plan": plan_report(plan),
        "regions": profile.report(),
        "precision": policy.report(),
        "value_checks_pending": plan["zero_checks"],
        "tensor_value_transforms": [
            {"tensor": name, "operation": fn.__name__} for name, fn in sorted(transforms.items())
        ],
    }
    return SimpleNamespace(
        args=args,
        forward=forward,
        mapping=mapping,
        hf=hf,
        mc=mc,
        plan=plan,
        profile=profile,
        source=source,
        reader=reader,
        policy=policy,
        report=report,
        shard_size=shard_size,
        zero_getter=zero_getter,
        transforms=transforms,
        adapter=adapter,
    )


def execute(context):
    import torch

    from tools.checkpoint.hf_mcore_dcp.storage.dcp import save_streaming_checkpoint
    from tools.checkpoint.hf_mcore_dcp.storage.hf import ShardedSafetensorWriter, copy_hf_assets
    from tools.checkpoint.hf_mcore_dcp.storage.precision import (
        encode_hf_tensors,
        output_config,
    )
    from tools.checkpoint.hf_mcore_dcp.utils import staged_output

    c, args = context, context.args
    inputs = [args.source] + ([args.hf_reference] if args.hf_reference else [])
    inputs += [p for p in [args.mapping, *args.mapping_override] if p]
    for name in c.plan["zero_checks"]:
        if c.zero_getter is None or bool((c.zero_getter(name) != 0).any()):
            raise MappingError(f"Cannot omit a nonzero tensor: {name}")
    with staged_output(args.output, args.overwrite, inputs) as output:
        if c.forward:
            details = save_streaming_checkpoint(
                c.source,
                c.profile,
                c.policy,
                c.hf,
                output,
                model_args=c.mc,
                max_shard_size=c.shard_size,
            )
            details["iteration_dir"] = str(Path(args.output).resolve() / "iter_0000000")
            (output / "iter_0000000" / "hf_export.json").write_text(
                json.dumps({"schema_version": 2, "mapping": c.mapping, "hf_config": c.hf}, indent=2)
                + "\n"
            )
            copy_hf_assets(args.source, output / "iter_0000000" / "hf_assets")
            c.report["storage"] = details
        else:
            copy_hf_assets(args.hf_reference, output)
            config = output_config(c.hf, c.policy)
            if c.adapter and hasattr(c.adapter, "export_config"):
                config = c.adapter.export_config(config)
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

            def tensors():
                for name, tensor in c.reader.tensors():
                    yield name, _transform_tensor(name, tensor, c.transforms, c.policy)
                for node in c.plan["nodes"]:
                    if node.get("generate"):
                        from tools.checkpoint.hf_mcore_dcp.layout.compiler import STORAGE_DTYPES

                        item = node["generate"]
                        for name in node["outputs"]:
                            yield (
                                name,
                                torch.full(
                                    item["shape"],
                                    item["value"],
                                    dtype=STORAGE_DTYPES[item["dtype"]],
                                ),
                            )

            writer = ShardedSafetensorWriter(output, c.shard_size)
            for name, tensor in encode_hf_tensors(tensors(), c.source, c.policy):
                writer.add(name, tensor)
            writer.finish()
        c.report["value_checks_pending"] = []
        (output / "conversion_report.json").write_text(json.dumps(c.report, indent=2) + "\n")
    print(f"{args.command} completed: {Path(args.output).resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("hf-to-mcore", "mcore-to-hf", "export-mapping"))
    parser.add_argument(
        "--mapping", help="YAML/JSON structure composition; required unless saved with source"
    )
    parser.add_argument("--mapping-override", action="append", default=[])
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--hf-reference",
        help="Required for mcore-to-hf in this release: HF config and tokenizer/assets; no weights required",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        help="Default: fp32 for HF import, auto for MCore export",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export-mapping":
            if not args.mapping:
                raise MappingError("export-mapping requires --mapping")
            mapping = load_mapping(args.mapping, args.mapping_override)
            if args.dry_run:
                print(json.dumps(mapping, indent=2))
            else:
                with Path(args.output).open(
                    "w" if args.overwrite else "x", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(mapping, indent=2) + "\n")
        else:
            if not args.source:
                raise MappingError("Conversion requires --source")
            context = prepare(args)
            if args.dry_run:
                print(json.dumps(context.report, indent=2))
            else:
                execute(context)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

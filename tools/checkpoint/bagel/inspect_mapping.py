#!/usr/bin/env python3
"""Inspect BAGEL checkpoint coverage without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open

try:
    from .mapping import MappingKind, build_bagel_registry
except ImportError:  # Support direct execution from the repository root.
    from mapping import MappingKind, build_bagel_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Directory containing ema.safetensors and ae.safetensors")
    parser.add_argument("--json-output", type=Path, help="Optional path for the complete JSON report")
    parser.add_argument("--show-unmapped", action="store_true", help="Print every unmapped key")
    return parser.parse_args()


def tensor_inventory(path: Path) -> list[dict[str, Any]]:
    inventory = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_slice(key)
            inventory.append({"key": key, "shape": list(tensor.get_shape())})
    return inventory


def inspect(checkpoint: Path) -> dict[str, Any]:
    registry = build_bagel_registry()
    files = ("ema.safetensors", "ae.safetensors")
    records: list[dict[str, Any]] = []
    for filename in files:
        path = checkpoint / filename
        if not path.is_file():
            raise FileNotFoundError(f"required checkpoint file does not exist: {path}")
        for tensor in tensor_inventory(path):
            resolved = registry.resolve(tensor["key"])
            records.append(
                {
                    "file": filename,
                    **tensor,
                    "status": "mapped" if resolved else "unmapped",
                    "rule": resolved.rule if resolved else None,
                    "kind": resolved.kind.value if resolved else None,
                    "target": resolved.target if resolved else None,
                    "group": resolved.group if resolved else None,
                    "note": resolved.note if resolved else "",
                }
            )

    kind_counts = Counter(record["kind"] or "unmapped" for record in records)
    rule_counts = Counter(record["rule"] or "unmapped" for record in records)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["group"]:
            groups[record["group"]].append(record)

    incomplete_groups = []
    for name, members in sorted(groups.items()):
        expected = 3 if ".qkv" in name else 2
        if len(members) != expected:
            incomplete_groups.append(
                {"group": name, "expected": expected, "actual": len(members), "keys": [m["key"] for m in members]}
            )

    layers = sorted(
        {
            int(part)
            for record in records
            if record["key"].startswith("language_model.model.layers.")
            for part in [record["key"].split(".")[3]]
        }
    )
    vit_layers = sorted(
        {
            int(part)
            for record in records
            if record["key"].startswith("vit_model.vision_model.encoder.layers.")
            for part in [record["key"].split(".")[4]]
        }
    )
    unmapped = [record for record in records if record["status"] == "unmapped"]
    return {
        "checkpoint": str(checkpoint.resolve()),
        "summary": {
            "tensor_count": len(records),
            "mapped_count": len(records) - len(unmapped),
            "unmapped_count": len(unmapped),
            "language_layers": layers,
            "vision_layers": vit_layers,
            "kind_counts": dict(sorted(kind_counts.items())),
            "rule_counts": dict(sorted(rule_counts.items())),
            "incomplete_fusion_group_count": len(incomplete_groups),
        },
        "incomplete_fusion_groups": incomplete_groups,
        "records": records,
    }


def main() -> int:
    args = parse_args()
    report = inspect(args.checkpoint)
    summary = report["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.show_unmapped:
        for record in report["records"]:
            if record["status"] == "unmapped":
                print(f"UNMAPPED {record['file']} {record['key']} {record['shape']}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return int(summary["unmapped_count"] != 0 or summary["incomplete_fusion_group_count"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())

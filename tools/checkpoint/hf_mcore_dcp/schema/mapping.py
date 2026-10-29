# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compose named weight structures and argument groups from the shared YAML catalogs."""

from __future__ import annotations

import itertools
import json
from copy import deepcopy
from pathlib import Path

from .. import MappingError
from ..utils import numeric_pattern
from .config import resolve

SCHEMA_VERSION = 2
_DICT_FIELDS = (
    "config_map",
    "hf_config",
    "mcore_args",
    "ignore",
    "quantization",
    "naming",
    "scopes",
    "zero_checks",
)


def merge_rules(base, overrides):
    result = {row["id"]: deepcopy(row) for row in base}
    seen = set()
    for row in overrides:
        key = row["id"]
        if key in seen:
            raise MappingError(f"Duplicate rule id: {key}")
        seen.add(key)
        if row.get("disabled"):
            del result[key]
        else:
            result[key] = deepcopy(row)
    return list(result.values())


def _read(path):
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise MappingError(f"Expected mapping schema_version: {SCHEMA_VERSION}: {path}")
    return value


def overlay(base, patch):
    result = deepcopy(base)
    result["rules"] = merge_rules(base.get("rules", []), patch.get("rules", []))
    for key in _DICT_FIELDS:
        result[key] = {**base.get(key, {}), **deepcopy(patch.get(key, {}))}
    for key in ("profile", "adapter"):
        if key in patch:
            result[key] = patch[key]
    return result


def _bind(value, bindings):
    """Bind structural names, leaving numeric layer/expert axes for expansion."""
    if isinstance(value, str):
        for name, replacement in bindings.items():
            value = value.replace("{" + name + "}", replacement)
        return value
    if isinstance(value, list):
        return [_bind(item, bindings) for item in value]
    if isinstance(value, dict):
        return {key: _bind(item, bindings) for key, item in value.items()}
    return value


def compose_mapping(document, base=None):
    if document.get("schema_version") != SCHEMA_VERSION:
        raise MappingError(
            f"Expected mapping schema_version: {SCHEMA_VERSION}; select a current --mapping"
        )
    allowed = {
        "schema_version",
        "profile",
        "adapter",
        "parameters",
        "args",
        "bindings",
        "rules",
        *_DICT_FIELDS,
    }
    if document.keys() - allowed:
        raise MappingError(f"Unknown mapping fields: {sorted(document.keys() - allowed)}")
    if document.get("naming", {}).keys() - {"strip_prefix", "segments", "replace"}:
        raise MappingError("naming only supports input strip_prefix, segments and replace")
    root = Path(__file__).parent
    parameters = _read(root / "parameters.yaml")
    arguments = _read(root / "args.yaml")["groups"]
    bindings = {**parameters["bindings"], **document.get("bindings", {})}
    rules, groups = [], []
    for selection in document.get("parameters", []):
        item = {"use": selection} if isinstance(selection, str) else selection
        template = parameters["structures"][item["use"]]
        groups.extend(template.get("args", []))
        for rule in _bind(template["rules"], {**bindings, **item.get("bindings", {})}):
            if "scope" in item:
                rule["scope"] = item["scope"]
            if "optional" in item:
                rule["optional"] = item["optional"]
            rules.append(rule)
    result = {"schema_version": SCHEMA_VERSION, "rules": merge_rules([], rules)}
    for group in dict.fromkeys([*groups, *document.get("args", [])]):
        result = overlay(result, arguments[group])
    if base is not None:
        result = overlay(base, result)
    return overlay(result, {**document, "rules": _bind(document.get("rules", []), bindings)})


def load_mapping(location, overrides=(), *, _base=None):
    result = compose_mapping(_read(location), _base)
    for path in overrides:
        result = compose_mapping(_read(path), result)
    return result


def expand_rules(mapping, keys, configs, forward):
    """Enumerate layer ids from actual keys; only collected axes need a count."""
    source_side = "hf" if forward else "mcore"
    rows = []
    for rule in mapping["rules"]:
        if rule.get("scope") is not None and rule["scope"] not in mapping.get("scopes", {}):
            raise MappingError(f"Unknown scope: {rule['scope']}")
        scopes = mapping.get("scopes", {}).get(rule.get("scope"), [None])
        for scope in scopes:
            row = {k: v for k, v in rule.items() if k != "scope"}
            if scope is not None:
                for side in ("hf", "mcore"):
                    row[side] = [scope[side] + t for t in row[side]]
                row["id"] = rule["id"] + ":" + scope["hf"]
                row["extra_state"] = [scope["mcore"] + t for t in row.get("extra_state", [])]
                row["optional"] = row.get("optional", False) or scope.get("optional", False)
            rows.append(row)
    for raw in rows:
        allowed = {
            "id",
            "hf",
            "mcore",
            "op",
            "params",
            "collect",
            "optional",
            "extra_state",
            "export_slice",
            "generate",
            "shape_config",
        }
        if raw.keys() - allowed:
            raise MappingError(
                f"Unknown fields in rule {raw['id']}: {sorted(raw.keys() - allowed)}"
            )
        if raw.get("generate"):
            row = {**raw, "params": {}, "generate": resolve(raw["generate"], configs)}
            yield row
            continue
        for side in ("hf", "mcore"):
            if (
                not isinstance(raw.get(side), list)
                or not raw[side]
                or any(not isinstance(x, str) or not x for x in raw[side])
            ):
                raise MappingError(f"Rule {raw['id']} requires nonempty {side} name templates")
        declared_collect = raw.get("collect", {})
        if not isinstance(declared_collect, dict):
            raise MappingError(f"Invalid collected axes: {raw['id']}")
        patterns = [numeric_pattern(t)[0] for t in raw[source_side]]
        contexts = set()
        for key in keys:
            for pattern in patterns:
                match = pattern.fullmatch(key)
                if match:
                    contexts.add(
                        tuple(
                            sorted(
                                (k, v)
                                for k, v in match.groupdict().items()
                                if k not in declared_collect
                            )
                        )
                    )
        if not contexts:
            if raw.get("optional", False):
                continue  # Do not require MoE/MLA configuration for absent modules.
            raise MappingError(f"Rule {raw['id']} matches no {source_side} tensors")
        collect = resolve(declared_collect, configs)
        if any(
            not isinstance(key, str) or not key.isidentifier() or type(n) is not int or n <= 0
            for key, n in collect.items()
        ):
            raise MappingError(f"Invalid collected axes: {raw['id']}")
        for ctx in sorted(contexts, key=lambda ctx: tuple(int(v) for _, v in ctx)):
            row = {**raw, "params": resolve(raw.get("params", {}), configs)}
            for side in ("hf", "mcore"):
                names = []
                # Collected names are expert-major, preserving the listed gate/up order.
                axes = {
                    k: n
                    for k, n in collect.items()
                    if any(k in numeric_pattern(t)[1] for t in raw[side])
                }
                for values in itertools.product(*(range(n) for n in axes.values())):
                    bindings = {**dict(ctx), **dict(zip(axes, values))}
                    try:
                        names.extend(t.format(**bindings) for t in raw[side])
                    except KeyError as error:
                        raise MappingError(
                            f"Unbound target placeholder in {raw['id']}: {error}"
                        ) from error
                if len(names) != len(set(names)):
                    raise MappingError(f"Duplicate expanded names: {raw['id']}")
                row[side] = names
            if "extra_state" in row:
                axes = {
                    k: n
                    for k, n in collect.items()
                    if any(k in numeric_pattern(t)[1] for t in row["extra_state"])
                }
                row["extra_state"] = [
                    t.format(
                        **{
                            **dict(ctx),
                            **dict(zip(axes, values)),
                            **{k + "_count": v for k, v in collect.items()},
                        }
                    )
                    for values in itertools.product(*(range(n) for n in axes.values()))
                    for t in row["extra_state"]
                ]
            if "export_slice" in row:
                row["export_slice"] = resolve(row["export_slice"], configs)
            yield row

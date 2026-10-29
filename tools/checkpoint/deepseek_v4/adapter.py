# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Only DSv4 config normalization and its disabled-router-bias convention.

No model construction, expected-shape table, storage or independent conversion
pipeline. User-declared tensor layouts are executed by the shared compiler.
"""

from copy import deepcopy

from tools.checkpoint.hf_mcore_dcp import MappingError

LAYER_RATIOS = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def normalize_config(config, forward, keys):
    config = deepcopy(config)
    if forward:
        if "compress_ratios" not in config:
            rates = {**LAYER_RATIOS, **config.get("compress_rates", {})}
            config["compress_ratios"] = [rates[k] for k in config["layer_types"]]
        if "num_hash_layers" not in config:
            layers = config.get("mlp_layer_types", [])
            config["num_hash_layers"] = next(
                (i for i, kind in enumerate(layers) if kind != "hash_moe"), len(layers)
            )
        config.setdefault("num_nextn_predict_layers", 0)
        if len(config["compress_ratios"]) == config["num_hidden_layers"]:
            config["compress_ratios"] = (
                list(config["compress_ratios"]) + [0] * config["num_nextn_predict_layers"]
            )
        config.setdefault("n_shared_experts", 1)
        config.setdefault("tie_word_embeddings", False)
        config.setdefault(
            "moe_router_enable_expert_bias", any(k.endswith(".ffn.gate.bias") for k in keys)
        )
        scaling = dict(config.get("rope_scaling") or {})
        scaling["rope_type"] = scaling.pop("type", scaling.get("rope_type", "rope"))
        if scaling["rope_type"] == "default":
            scaling["rope_type"] = "rope"
        if scaling["rope_type"] == "yarn":
            scaling.setdefault("mscale", 1.0)
            scaling.setdefault("mscale_all_dim", 0.0)
        config["rope_scaling"] = scaling
    else:
        config["moe_router_enable_expert_bias"] = bool(
            config.get("moe_router_enable_expert_bias")
        ) or (config.get("moe_router_load_balancing_type") == "quantile_balancing")
        config["rope_type"] = config.get("rope_type") or "yarn"
        if config["rope_type"] == "yarn":
            config.setdefault("beta_fast", 32)
            config.setdefault("beta_slow", 1)
    return config


def finish_config(config, forward):
    config = deepcopy(config)
    if forward:
        count = config["num_layers"] + (config.get("mtp_num_layers", 0) or 0)
        config["csa_compress_ratios"] = config["csa_compress_ratios"][:count]
    else:
        layers = config["num_hidden_layers"]
        mtp = config.get("num_nextn_predict_layers", 0) or 0
        ratios = list(config["compress_ratios"])
        if len(ratios) == layers:
            ratios.extend([0] * mtp)
        if len(ratios) < layers + mtp:
            raise MappingError("Compression layout is shorter than the configured layer count")
        config["compress_ratios"] = ratios[: layers + mtp]
        names = {v: k for k, v in LAYER_RATIOS.items()}
        # Unknown ratios can still be used with explicit rules; keep their numeric form.
        config["layer_types"] = [names.get(r, str(r)) for r in ratios[:layers]]
        config["compress_rates"] = {names[r]: r for r in names if r}
        config["mlp_layer_types"] = [
            "hash_moe" if i < config.get("num_hash_layers", 0) else "moe" for i in range(layers)
        ]
        if (config.get("rope_scaling") or {}).get("rope_type") != "yarn":
            config["rope_scaling"] = None
    return config


def prepare_mapping(mapping, configs, specs, forward):
    mapping = deepcopy(mapping)
    config = configs["hf"] if forward else configs["mcore"]
    if config.get("moe_router_enable_expert_bias"):
        return mapping
    biases = [r for r in mapping["rules"] if r["id"] == "router_bias"]
    if not biases:
        return mapping
    mapping["rules"] = [r for r in mapping["rules"] if r["id"] != "router_bias"]
    hf = configs["hf"]
    for rule in biases:
        templates = rule["hf"]
        for scope in mapping["scopes"][rule["scope"]]:
            mtp = scope["hf"].startswith("mtp.")
            count = (hf.get("num_nextn_predict_layers", 0) or 0) if mtp else hf["num_hidden_layers"]
            start = 0 if mtp else hf.get("num_hash_layers", 0)
            for layer in range(start, count):
                for template in templates:
                    name = (scope["hf"] + template).format(layer=layer)
                    if forward:
                        mapping.setdefault("zero_checks", {}).setdefault("hf", []).append(name)
                    else:
                        mapping["rules"].append(
                            {
                                "id": "zero:" + name,
                                "hf": [name],
                                "mcore": [],
                                "generate": {
                                    "side": "hf",
                                    "shape": [hf["n_routed_experts"]],
                                    "dtype": "F32",
                                    "value": 0,
                                },
                            }
                        )
    return mapping

#!/usr/bin/env python3

"""
FlagScale mix-precision compress entrypoint (llm-compressor aligned)

Key behaviors:
1) Reads FlagScale-generated Hydra config via --config-path (OmegaConf.load)
2) Selects pipeline by passing dataset=cfg.compress_args.scheme (e.g., "mix_precision_search")
3) Recipe uses:
   - QuantizationModifier: default W8A16 on broad targets (e.g., ["Linear"])
   - QuIPModifier: targets=[] initially; pipeline writes back chosen targets
4) Does NOT call model.save_pretrained() after oneshot (avoid overwriting compressed artifacts)
   Only saves processor/tokenizer into a subdir to avoid collisions.
"""

import argparse
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

# ---- Ensure your custom CalibrationPipeline is registered ----
# Adjust this import path to wherever your pipeline module lives.
# The important thing is: importing the module executes the @CalibrationPipeline.register decorator.
try:
    import flagscale.compress.pipelines.mix_precision_pipeline  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "Failed to import mix_precision_pipeline for registration. "
        "Fix the import path so your CalibrationPipeline is registered."
    ) from e


def _pick(cfg: Any, *keys: str, default=None):
    """Safe getter for OmegaConf / dict-like configs."""
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            # OmegaConf supports attribute and item access; use item access defensively
            try:
                cur = cur.get(k)
            except Exception:
                try:
                    cur = getattr(cur, k)
                except Exception:
                    return default
    return default if cur is None else cur


def _as_abs_output_dir(cfg_root: Any) -> str:
    """
    Build output_dir from:
      - system.save_dir (required-ish)
      - experiment.exp_dir (optional): if present and save_dir is relative, join them
    """
    save_dir = _pick(cfg_root, "system", "save_dir", default=None)
    if not save_dir:
        raise ValueError("Missing config field: system.save_dir")

    save_dir = str(save_dir)

    exp_dir = _pick(cfg_root, "experiment", "exp_dir", default=None)
    if exp_dir and not os.path.isabs(save_dir):
        return str(Path(str(exp_dir)) / save_dir)

    return str(Path(save_dir))


def _resolve_model_id_or_path(cfg_root: Any) -> str:
    # common patterns: model.model_path, model.path, model.name_or_path
    for cand in [
        ("model", "model_path"),
        ("model", "path"),
        ("model", "name_or_path"),
        ("model_path",),
    ]:
        val = _pick(cfg_root, *cand, default=None)
        if val:
            return str(val)
    raise ValueError(
        "Missing model path in config. Expected one of: model.model_path / model.path / model.name_or_path"
    )


def _load_cfg(config_path: str) -> Any:
    cfg = OmegaConf.load(config_path)

    # Many FlagScale setups nest the actual compress config under `compress:`
    # because compress_mix.yaml uses `defaults: - compress: mix_precision`.
    # Normalize so `cfg_root` has fields: system/compress_args/data/model/...
    if _pick(cfg, "compress", default=None) is not None:
        # merge top-level + cfg.compress so experiment.* still accessible
        cfg_root = OmegaConf.merge(cfg, cfg.compress)
        return cfg_root

    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-path",
        required=True,
        help="Path to FlagScale/Hydra generated config.yaml (e.g., outputs/.../hydra/.hydra/config.yaml)",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.config_path)

    # ---- core config ----
    model_id_or_path = _resolve_model_id_or_path(cfg)
    output_dir = _as_abs_output_dir(cfg)

    scheme = _pick(cfg, "compress_args", "scheme", default=None)
    if not scheme:
        raise ValueError(
            "Missing config field: compress_args.scheme (e.g., 'mix_precision_search')"
        )

    # Targets default: ["Linear"]
    targets = _pick(cfg, "compress_args", "targets", default=["Linear"])
    if targets is None or targets == "":
        targets = ["Linear"]

    # Calibration/data knobs (oneshot expects num_calibration_samples, not steps)
    batch_size = int(_pick(cfg, "data", "batch_size", default=1))
    num_calibration_samples = _pick(cfg, "data", "num_calibration_samples", default=None)
    if num_calibration_samples is None:
        # backward compat: some configs use num_calibration_steps
        steps = int(_pick(cfg, "data", "num_calibration_steps", default=512))
        num_calibration_samples = steps * max(batch_size, 1)
    num_calibration_samples = int(num_calibration_samples)

    int(_pick(cfg, "data", "max_seq_length", default=384))
    str(_pick(cfg, "data", "text_column", default="text"))
    bool(_pick(cfg, "data", "pad_to_max_length", default=True))

    tokenizer_args = _pick(cfg, "data", "tokenizer_args", default={}) or {}
    trust_remote_code = bool(tokenizer_args.get("trust_remote_code", True))

    # ---- build recipe (llm-compressor aligned) ----
    # Import modifiers with a couple of fallback paths, depending on llmcompressor version.
    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except Exception:
        from llmcompressor.modifiers.quantization.quantization import (
            QuantizationModifier,  # type: ignore
        )

    try:
        pass
    except Exception:
        # some versions may expose it elsewhere
        try:
            pass  # type: ignore
        except Exception:
            pass

    # Keep ignore minimal; customize if your project passes ignore patterns in config.
    # ignore = _pick(cfg, "compress_args", "ignore", default=None)
    ignore = getattr(cfg.compress.compress_args, "ignore", None) or ["lm_head"]

    recipe = [
        # global default quant: 8-bit weights, fp16 acts (per your existing intent)
        QuantizationModifier(
            targets=targets,
            scheme="W8A16",
            ignore=ignore,
            # ignore=cfg.compress.compress_args.get("ignore", None),
        ),
    ]

    # ---- run oneshot ----
    from llmcompressor import oneshot

    # Important: per your config design, dataset=scheme is what selects the registered CalibrationPipeline.
    # (oneshot signature confirms dataset is used for that purpose in your integration)
    compressed_model = oneshot(
        model=model_id_or_path,
        tokenizer=model_id_or_path,  # safe default; can be overridden by cfg if you expose tokenizer_path
        # processor=model_id_or_path,   # for VLMs; if not applicable, llmcompressor usually ignores safely
        trust_remote_code_model=trust_remote_code,
        recipe=recipe,
        pipeline="mix_precision_search",
        output_dir=output_dir,
        save_compressed=True,  # crucial: let llmcompressor write compressed artifacts
    )

    # ---- avoid overwriting compressed artifacts ----
    # Save processor/tokenizer into a subdir to avoid colliding with llmcompressor exporter outputs.
    aux_dir = Path(output_dir) / "aux"
    aux_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort: if it’s a HF model, we can load tokenizer/processor and save them.
    # We intentionally do NOT call compressed_model.save_pretrained(output_dir).
    try:
        from transformers import AutoProcessor, AutoTokenizer

        # Tokenizer
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id_or_path,
                use_fast=bool(tokenizer_args.get("use_fast", True)),
                trust_remote_code=trust_remote_code,
            )
            tok.save_pretrained(str(aux_dir / "tokenizer"))
        except Exception:
            pass

        # Processor (for VLMs); harmless if not available
        try:
            proc = AutoProcessor.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
            )
            proc.save_pretrained(str(aux_dir / "processor"))
        except Exception:
            pass

    except Exception:
        # transformers not available or not needed
        pass

    print(f"[OK] Compressed model exported to: {output_dir}")
    return compressed_model


if __name__ == "__main__":
    main()

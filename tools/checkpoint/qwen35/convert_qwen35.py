#!/usr/bin/env python3
"""Unified Qwen3.5 HF <-> Megatron checkpoint converter.

Supports both dense and MoE models. Model type is auto-detected from the
input checkpoint / YAML; users do not need to specify it.

Examples:
    # HF -> Megatron
    python convert_qwen35.py --direction hf2meg \
        --hf-path /path/to/hf \
        --meg-path /path/to/output \
        --yaml /path/to/train.yaml \
        [--ref-path /path/to/ref]

    # Megatron -> HF
    python convert_qwen35.py --direction meg2hf \
        --meg-path /path/to/meg/checkpoint \
        --hf-path /path/to/output \
        --yaml /path/to/train.yaml \
        [--ref-path /path/to/hf/ref]
"""

import argparse
import sys

from qwen35.config import Config, detect_model_type
from qwen35.constants import LN_ADJUSTMENT
from qwen35.converter import DenseConverter, MoEConverter


def parse_args():
    p = argparse.ArgumentParser(description="Unified Qwen3.5 checkpoint converter")
    p.add_argument(
        "--direction",
        required=True,
        choices=["hf2meg", "meg2hf"],
        help="Conversion direction",
    )
    p.add_argument("--hf-path", required=True, help="Path to HF checkpoint directory")
    p.add_argument(
        "--meg-path",
        required=True,
        help="For hf2meg: output Megatron checkpoint directory. "
        "For meg2hf: input Megatron checkpoint directory.",
    )
    p.add_argument(
        "--yaml",
        required=True,
        help="Path to training YAML config (provides TP/PP/EP and model shapes)",
    )
    p.add_argument(
        "--ref-path",
        default=None,
        help="Reference checkpoint path for validation (Megatron ref for hf2meg, "
        "HF ref for meg2hf).",
    )
    p.add_argument(
        "--adjust-embedding",
        action="store_true",
        help="Adjust embedding vocab size to match reference checkpoint (hf2meg only)",
    )
    p.add_argument(
        "--adjust-ln",
        action="store_true",
        help="Enable legacy layer norm adjustment (add/subtract 1.0). "
        "Only use this if the model stores raw gamma values instead of "
        "zero-centered weights (default: disabled for Qwen3.5).",
    )
    p.add_argument(
        "--no-adjust-ln",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Override tensor model parallel size from YAML",
    )
    p.add_argument(
        "--pp",
        type=int,
        default=None,
        help="Override pipeline model parallel size from YAML",
    )
    p.add_argument(
        "--ep",
        type=int,
        default=None,
        help="Override expert model parallel size from YAML (MoE only)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI override for layer norm adjustment
    if args.adjust_ln:
        import qwen35.constants

        qwen35.constants.LN_ADJUSTMENT = True
    if args.no_adjust_ln:
        import qwen35.constants

        qwen35.constants.LN_ADJUSTMENT = False

    cfg = Config(args.yaml)
    if args.tp is not None:
        cfg.tp = args.tp
    if args.pp is not None:
        cfg.pp = args.pp
    if args.ep is not None:
        cfg.ep = args.ep

    # Auto-detect model type from whichever input is available
    hf_input = args.hf_path if args.direction == "hf2meg" else None
    meg_input = args.meg_path if args.direction == "meg2hf" else None
    model_type = detect_model_type(
        hf_dir=hf_input,
        meg_dir=meg_input,
        yaml_path=args.yaml,
    )
    converter_cls = MoEConverter if model_type == "moe" else DenseConverter
    converter = converter_cls(cfg, adjust_embedding=args.adjust_embedding)

    print(f"Direction: {args.direction}")
    print(f"Model type: {model_type}")
    print(f"TP={cfg.tp}, PP={cfg.pp}, EP={cfg.ep}")
    print(f"Layers={cfg.num_layers}, hidden={cfg.hidden_size}")
    print(f"LN adjustment: {LN_ADJUSTMENT}")

    if args.direction == "hf2meg":
        success = converter.run_hf2meg(args.hf_path, args.meg_path, args.ref_path)
    else:
        success = converter.run_meg2hf(args.meg_path, args.hf_path, args.ref_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

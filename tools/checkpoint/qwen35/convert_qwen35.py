#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import shutil
import sys
import tempfile

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
    p.add_argument(
        "--hf-path",
        default=None,
        help="For hf2meg: input HF checkpoint directory or ModelScope model ID. "
        "For meg2hf: output HF checkpoint directory (optional if --ref-path is given).",
    )
    p.add_argument(
        "--meg-path",
        default=None,
        help="For hf2meg: output Megatron checkpoint directory (optional if --ref-path is given). "
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
        "HF ref for meg2hf). When provided, the corresponding output path may be omitted; "
        "output will be written to a temporary directory, validated, and cleaned up.",
    )
    p.add_argument(
        "--ref-skip-value",
        action="store_true",
        help="When validating against --ref-path, skip numerical value comparison "
        "and only compare structure, keys, and shapes.",
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
        "--adjust-embedding",
        action="store_true",
        help="During hf2meg, adjust vocab size to match the reference checkpoint.",
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
    return p.parse_args(), p


def main():
    args, parser = parse_args()

    # Validate required paths based on direction
    if args.direction == "hf2meg":
        if not args.hf_path:
            parser.error("--hf-path is required for hf2meg")
        if not args.meg_path and not args.ref_path:
            parser.error("--meg-path is required for hf2meg when --ref-path is not provided")
    else:
        if not args.meg_path:
            parser.error("--meg-path is required for meg2hf")
        if not args.hf_path and not args.ref_path:
            parser.error("--hf-path is required for meg2hf when --ref-path is not provided")

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
    print(f"Ref skip value: {args.ref_skip_value}")

    temp_dir = None
    try:
        if args.direction == "hf2meg":
            out_dir = args.meg_path
            if not out_dir:
                temp_dir = tempfile.mkdtemp(prefix="qwen35_hf2meg_")
                out_dir = temp_dir
                print(f"No --meg-path provided; using temporary output: {out_dir}")
            success = converter.run_hf2meg(
                args.hf_path, out_dir, args.ref_path, skip_value=args.ref_skip_value
            )
        else:
            out_dir = args.hf_path
            if not out_dir:
                temp_dir = tempfile.mkdtemp(prefix="qwen35_meg2hf_")
                out_dir = temp_dir
                print(f"No --hf-path provided; using temporary output: {out_dir}")
            success = converter.run_meg2hf(
                args.meg_path, out_dir, args.ref_path, skip_value=args.ref_skip_value
            )
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temporary output directory: {temp_dir}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

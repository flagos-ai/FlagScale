"""Validation helpers for generated checkpoints."""

import os

import torch
from safetensors import safe_open

from qwen35.io import find_megatron_shard


def validate_hf2meg_against_ref(shards_dict, cfg, ref_dir, use_ep=False):
    """Compare generated Megatron shards with a reference checkpoint."""
    if ref_dir is None:
        return True

    print("\n" + "=" * 80)
    print("Validation: Comparing generated Megatron checkpoint with reference")
    print("=" * 80)

    all_ok = True
    for rank_tuple in sorted(shards_dict.keys()):
        if use_ep:
            pp_rank, tp_rank, _ep_rank = rank_tuple
        else:
            pp_rank, tp_rank = rank_tuple

        ref_path = find_megatron_shard(ref_dir, tp_rank, pp_rank)
        if ref_path is None:
            print(f"  Skip: reference not found for PP={pp_rank}, TP={tp_rank}")
            continue

        ref_sd = torch.load(ref_path, map_location="cpu", weights_only=False)["model"]
        gen_sd = shards_dict[rank_tuple]

        ref_keys = set(k for k in ref_sd.keys() if "_extra_state" not in k)
        gen_keys = set(k for k in gen_sd.keys() if "_extra_state" not in k)

        if not cfg.untie:
            ref_keys.discard("language_model.output_layer.weight")
            gen_keys.discard("language_model.output_layer.weight")

        missing = ref_keys - gen_keys
        extra = gen_keys - ref_keys

        if missing:
            print(f"  PP={pp_rank}, TP={tp_rank}: Missing keys ({len(missing)}):")
            for k in sorted(missing)[:5]:
                print(f"    {k}")
            all_ok = False
        if extra:
            print(f"  PP={pp_rank}, TP={tp_rank}: Extra keys ({len(extra)}):")
            for k in sorted(extra)[:5]:
                print(f"    {k}")
            all_ok = False

        mismatches = 0
        for k in ref_keys & gen_keys:
            if isinstance(ref_sd[k], torch.Tensor) and isinstance(gen_sd[k], torch.Tensor):
                if ref_sd[k].shape != gen_sd[k].shape:
                    if "embedding.word_embeddings" in k:
                        print(
                            "  Embedding shape differs (expected if not using --adjust-embedding):"
                        )
                        print(f"    ref: {tuple(ref_sd[k].shape)}, gen: {tuple(gen_sd[k].shape)}")
                    else:
                        mismatches += 1
                        if mismatches <= 3:
                            print(f"  Shape mismatch: {k}")
                            print(
                                f"    ref: {tuple(ref_sd[k].shape)}, gen: {tuple(gen_sd[k].shape)}"
                            )

        if mismatches > 0:
            print(f"  Total shape mismatches: {mismatches}")
            all_ok = False

        if not missing and not extra and mismatches == 0:
            print(f"  PP={pp_rank}, TP={tp_rank}: OK")

    print("=" * 80)
    print("Validation PASSED" if all_ok else "Validation FAILED")
    print("=" * 80)
    return all_ok


def load_hf_shapes(hf_dir):
    """Load shape dict from a reference HF checkpoint."""
    if hf_dir is None:
        return None
    shapes = {}
    for st_file in sorted(os.listdir(hf_dir)):
        if not st_file.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(hf_dir, st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                shapes[key] = list(f.get_tensor(key).shape)
    return shapes


def validate_meg2hf_against_ref(hf_sd, cfg, ref_dir):
    """Compare generated HF checkpoint with a reference HF model."""
    if ref_dir is None:
        return True

    print("\n" + "=" * 100)
    print("Shape Comparison: Converted vs Reference HF Model")
    print("=" * 100)

    ref_shapes = load_hf_shapes(ref_dir)
    if ref_shapes is None or len(ref_shapes) == 0:
        print("No reference shapes provided, skipping comparison.")
        return True

    import math
    import re

    expected_missing = set()
    for k in list(ref_shapes.keys()):
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx >= cfg.num_layers:
                expected_missing.add(k)

    all_keys = sorted(set(list(hf_sd.keys()) + list(ref_shapes.keys())))
    mismatches, missing, extra, matched = [], [], [], 0

    for k in all_keys:
        in_conv = k in hf_sd
        in_ref = k in ref_shapes
        if in_conv and in_ref:
            cs = list(hf_sd[k].shape)
            rs = ref_shapes[k]
            if cs == rs:
                matched += 1
            else:
                mismatches.append((k, cs, rs))
        elif in_conv:
            extra.append(k)
        elif k not in expected_missing:
            missing.append(k)

    print(f"\nMatched: {matched}/{len(all_keys)}")
    if mismatches:
        print(f"\nShape mismatches ({len(mismatches)}):")
        for k, cs, rs in mismatches:
            print(f"  {k:80s} converted={cs} ref={rs}")
    if missing:
        print(f"\nMissing in converted ({len(missing)}):")
        for k in missing:
            print(f"  {k:80s} ref_shape={ref_shapes[k]}")
    if extra:
        print(f"\nExtra in converted ({len(extra)}):")
        for k in extra:
            print(f"  {k:80s} shape={list(hf_sd[k].shape)}")
    if expected_missing:
        print(f"\nExpected missing (Megatron has no equivalent): {len(expected_missing)}")

    conv_total = sum(t.numel() for t in hf_sd.values())
    ref_total = sum(math.prod(shape) if shape else 1 for shape in ref_shapes.values())
    print(f"\nConverted total params: {conv_total:>15,}")
    print(f"Reference total params: {ref_total:>15,}")
    print(f"Difference:             {conv_total - ref_total:>15,}")

    success = not (mismatches or missing or extra)
    print("\n" + "=" * 100)
    print("VALIDATION PASSED" if success else "VALIDATION FAILED")
    print("=" * 100)
    return success

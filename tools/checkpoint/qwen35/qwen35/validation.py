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

"""Validation helpers for generated checkpoints."""

import os

import torch

from qwen35.io import find_megatron_shard, load_hf_weights

# Tolerance for floating-point checkpoint comparison.
_RTOL = 1e-5
_ATOL = 1e-6


def _list_megatron_ranks(ref_dir, cfg):
    """List all (pp_rank, tp_rank, ep_rank) tuples found in a reference Megatron checkpoint.

    Scans ``ref_dir`` as well as ``release/`` and ``iter_*/`` subdirectories.
    Directory names are parsed according to ``cfg.pp`` and ``cfg.ep``.
    """
    candidates = [ref_dir]
    release_dir = os.path.join(ref_dir, "release")
    if os.path.isdir(release_dir):
        candidates.append(release_dir)

    if os.path.isdir(ref_dir):
        for d in sorted(os.listdir(ref_dir)):
            if d.startswith("iter_") and os.path.isdir(os.path.join(ref_dir, d)):
                candidates.append(os.path.join(ref_dir, d))
                iter_release = os.path.join(ref_dir, d, "release")
                if os.path.isdir(iter_release):
                    candidates.append(iter_release)

    ranks = set()
    for base in candidates:
        for entry in os.listdir(base):
            if not entry.startswith("mp_rank_"):
                continue
            parts = entry.split("_")[2:]  # strip "mp_rank" prefix
            if not parts:
                continue
            try:
                nums = [int(p) for p in parts]
            except ValueError:
                continue

            tp_rank = nums[0]
            pp_rank = 0
            ep_rank = 0
            if len(nums) == 2:
                if cfg.pp > 1:
                    pp_rank = nums[1]
                elif cfg.ep > 1:
                    ep_rank = nums[1]
            elif len(nums) >= 3:
                pp_rank, ep_rank = nums[1], nums[2]

            ranks.add((pp_rank, tp_rank, ep_rank))

    return ranks


def validate_hf2meg_against_ref(shards_dict, cfg, ref_dir, use_ep=False, skip_value=False):
    """Compare generated Megatron shards with a reference checkpoint.

    ``shards_dict`` keys are either ``(pp_rank, tp_rank)`` or
    ``(pp_rank, tp_rank, ep_rank)`` tuples.  The reference checkpoint is
    expected to contain matching shards.
    """
    if ref_dir is None:
        return True

    print("\n" + "=" * 80)
    print("Validation: Comparing generated Megatron checkpoint with reference")
    if skip_value:
        print("(value comparison disabled, checking structure/keys/shapes only)")
    print("=" * 80)

    all_ok = True
    gen_ranks = set()
    for rank_tuple in sorted(shards_dict.keys()):
        if len(rank_tuple) == 3:
            pp_rank, tp_rank, ep_rank = rank_tuple
        else:
            pp_rank, tp_rank = rank_tuple
            ep_rank = 0
        gen_ranks.add((pp_rank, tp_rank, ep_rank))

        ref_path = find_megatron_shard(ref_dir, tp_rank, pp_rank, ep_rank)
        if ref_path is None:
            rank_label = f"PP={pp_rank}, TP={tp_rank}"
            if cfg.ep > 1:
                rank_label = f"PP={pp_rank}, TP={tp_rank}, EP={ep_rank}"
            print(f"  FAIL: reference not found for {rank_label}")
            all_ok = False
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

        rank_label = f"PP={pp_rank}, TP={tp_rank}"
        if cfg.ep > 1:
            rank_label = f"PP={pp_rank}, TP={tp_rank}, EP={ep_rank}"

        if missing:
            print(f"  {rank_label}: Missing keys ({len(missing)}):")
            for k in sorted(missing)[:5]:
                print(f"    {k}")
            all_ok = False
        if extra:
            print(f"  {rank_label}: Extra keys ({len(extra)}):")
            for k in sorted(extra)[:5]:
                print(f"    {k}")
            all_ok = False

        mismatches = 0
        max_diff_info = None
        for k in ref_keys & gen_keys:
            if not isinstance(ref_sd[k], torch.Tensor) or not isinstance(gen_sd[k], torch.Tensor):
                continue
            if ref_sd[k].shape != gen_sd[k].shape:
                if "embedding.word_embeddings" in k:
                    print("  Embedding shape differs (expected if not using --adjust-embedding):")
                    print(f"    ref: {tuple(ref_sd[k].shape)}, gen: {tuple(gen_sd[k].shape)}")
                else:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"  Shape mismatch: {k}")
                        print(f"    ref: {tuple(ref_sd[k].shape)}, gen: {tuple(gen_sd[k].shape)}")
            elif not skip_value and not torch.allclose(
                ref_sd[k], gen_sd[k], rtol=_RTOL, atol=_ATOL
            ):
                mismatches += 1
                diff = (ref_sd[k] - gen_sd[k]).abs()
                info = (k, diff.max().item(), diff.mean().item())
                if max_diff_info is None or info[1] > max_diff_info[1]:
                    max_diff_info = info
                if mismatches <= 3:
                    print(f"  Value mismatch: {k}")
                    print(
                        f"    max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}"
                    )

        if mismatches > 0:
            print(f"  Total mismatches: {mismatches}")
            if max_diff_info:
                print(
                    f"  Largest difference: {max_diff_info[0]} "
                    f"max_diff={max_diff_info[1]:.6e}, mean_diff={max_diff_info[2]:.6e}"
                )
            all_ok = False

        if not missing and not extra and mismatches == 0:
            print(f"  {rank_label}: OK")

    ref_ranks = _list_megatron_ranks(ref_dir, cfg)
    missing_in_ref = gen_ranks - ref_ranks
    extra_in_ref = ref_ranks - gen_ranks
    if missing_in_ref:
        print(
            f"  FAIL: ranks present in generated checkpoint but missing in reference: "
            f"{sorted(missing_in_ref)}"
        )
        all_ok = False
    if extra_in_ref:
        print(
            f"  FAIL: ranks present in reference but missing in generated checkpoint: "
            f"{sorted(extra_in_ref)}"
        )
        all_ok = False

    print("=" * 80)
    print("Validation PASSED" if all_ok else "Validation FAILED")
    print("=" * 80)
    return all_ok


def validate_meg2hf_against_ref(hf_sd, cfg, ref_dir, skip_value=False):
    """Compare generated HF checkpoint with a reference HF model."""
    if ref_dir is None:
        return True

    print("\n" + "=" * 100)
    print("Value Comparison: Converted vs Reference HF Model")
    if skip_value:
        print("(value comparison disabled, checking structure/keys/shapes only)")
    print("=" * 100)

    ref_sd = load_hf_weights(ref_dir)
    if not ref_sd:
        print("No reference weights found, skipping comparison.")
        return True

    import re

    expected_missing = set()
    for k in list(ref_sd.keys()):
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx >= cfg.num_layers:
                expected_missing.add(k)

    ref_keys = set(k for k in ref_sd.keys() if "_extra_state" not in k)
    gen_keys = set(k for k in hf_sd.keys() if "_extra_state" not in k)

    missing = [k for k in (ref_keys - gen_keys) if k not in expected_missing]
    extra = list(gen_keys - ref_keys)

    mismatches = []
    matched = 0
    for k in sorted(ref_keys & gen_keys):
        if k in expected_missing:
            continue
        ref_t = ref_sd[k]
        gen_t = hf_sd[k]
        if not isinstance(ref_t, torch.Tensor) or not isinstance(gen_t, torch.Tensor):
            continue
        if ref_t.shape != gen_t.shape:
            mismatches.append((k, list(gen_t.shape), list(ref_t.shape), None, None))
        elif skip_value or torch.allclose(ref_t, gen_t, rtol=_RTOL, atol=_ATOL):
            matched += 1
        else:
            diff = (ref_t - gen_t).abs()
            mismatches.append(
                (k, list(gen_t.shape), list(ref_t.shape), diff.max().item(), diff.mean().item())
            )

    print(f"\nMatched: {matched}/{len(ref_keys & gen_keys)}")
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for k, cs, rs, max_d, mean_d in mismatches:
            if max_d is None:
                print(f"  {k:80s} converted={cs} ref={rs}")
            else:
                print(
                    f"  {k:80s} converted={cs} ref={rs} "
                    f"max_diff={max_d:.6e}, mean_diff={mean_d:.6e}"
                )
    if missing:
        print(f"\nMissing in converted ({len(missing)}):")
        for k in missing:
            print(f"  {k:80s} ref_shape={list(ref_sd[k].shape)}")
    if extra:
        print(f"\nExtra in converted ({len(extra)}):")
        for k in extra:
            print(f"  {k:80s} shape={list(hf_sd[k].shape)}")
    if expected_missing:
        print(f"\nExpected missing (Megatron has no equivalent): {len(expected_missing)}")

    conv_total = sum(t.numel() for t in hf_sd.values() if isinstance(t, torch.Tensor))
    ref_total = sum(t.numel() for t in ref_sd.values() if isinstance(t, torch.Tensor))
    print(f"\nConverted total params: {conv_total:>15,}")
    print(f"Reference total params: {ref_total:>15,}")
    print(f"Difference:             {conv_total - ref_total:>15,}")

    success = not (mismatches or missing or extra)
    print("\n" + "=" * 100)
    print("VALIDATION PASSED" if success else "VALIDATION FAILED")
    print("=" * 100)
    return success

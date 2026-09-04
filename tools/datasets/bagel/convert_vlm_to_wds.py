# Copyright (c) 2025, BAAI. All rights reserved.
# Convert Bagel VLM jsonl+images to WebDataset tar format for Energon.

"""
Input format:
  - jsonl files: each line is a JSON object with "conversations" field
  - image directories: referenced by <image> tags in conversations

Output format:
  WebDataset tar shards, each sample contains:
    - __key__: unique sample id
    - json: conversation data + metadata (encoded as UTF-8 JSON)
    - 000.jpg, 001.jpg, ...: image files referenced in conversations

Usage:
  python convert_vlm_to_wds.py \
    --jsonl_paths /data/vlm_sft/train.jsonl \
    --image_dirs /data/vlm_sft/images \
    --output_dir /data/vlm_sft/wds \
    --max_count 5000
"""

import argparse
import json
import os
from pathlib import Path

import webdataset as wds
import yaml
from tqdm import tqdm

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory


def parse_image_paths_from_conversations(conversations, image_dir):
    """Extract image file paths from conversation data."""
    for conv in conversations:
        value = conv.get("value", "")
        count = value.count("<image>")
        image_paths_needed = count
    # Images are typically listed in a separate field or inferred from conversation
    return image_paths_needed


def generate_configs(path: EPath, split, num_workers=1):
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]
    split_parts_ratio = [("train", split[0]), ("val", split[1]), ("test", split[2])]
    split_parts_patterns = None

    BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=split_parts_ratio,
        split_parts_patterns=split_parts_patterns,
        shuffle_seed=42,
        workers=num_workers,
    )

    # Write dataset.yaml for CrudeWebdataset
    metadata = {
        "__class__": "CrudeWebdataset",
        "__module__": "megatron.energon.flavors.crude",
        "subflavors": {"task": "vlm"},
    }
    meta_dir = os.path.join(path.url, ".nv-meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "dataset.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)


def convert_vlm_to_wds(
    jsonl_paths: list,
    image_dirs: list,
    output_dir: str,
    max_count: int = 5000,
    max_size: float = 1e9,
):
    """Convert VLM jsonl data to WebDataset tar format.

    Each sample in the tar contains:
      - __key__: "{dataset_idx}_{sample_idx}"
      - json: UTF-8 encoded JSON with conversations + metadata
      - 000.jpg, 001.jpg, ...: raw image bytes
    """
    os.makedirs(output_dir, exist_ok=True)

    global_idx = 0
    shard_pattern = os.path.join(output_dir, "vlm-%06d.tar")

    with wds.ShardWriter(shard_pattern, maxcount=max_count, maxsize=max_size) as sink:
        for jsonl_path, image_dir in zip(jsonl_paths, image_dirs):
            if not os.path.exists(jsonl_path):
                print(f"Warning: {jsonl_path} not found, skipping")
                continue

            print(f"Processing {jsonl_path} with image_dir={image_dir}")
            with open(jsonl_path, "r") as f:
                for line_idx, line in enumerate(
                    tqdm(f, desc=f"Converting {Path(jsonl_path).name}")
                ):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Warning: invalid JSON at line {line_idx}, skipping")
                        continue

                    conversations = data.get("conversations", [])
                    image_files = data.get("image", data.get("images", []))
                    if isinstance(image_files, str):
                        image_files = [image_files]

                    sample = {
                        "__key__": f"{global_idx:09d}",
                        "json": json.dumps(
                            {
                                "conversations": conversations,
                                "metadata": {
                                    k: v
                                    for k, v in data.items()
                                    if k not in ("conversations", "image", "images")
                                },
                            }
                        ).encode("utf-8"),
                    }

                    # Add images as numbered files
                    for img_idx, img_file in enumerate(image_files):
                        img_path = os.path.join(image_dir, img_file)
                        if os.path.exists(img_path):
                            with open(img_path, "rb") as img_f:
                                ext = Path(img_path).suffix.lstrip(".")
                                if ext not in ("jpg", "jpeg", "png", "webp"):
                                    ext = "jpg"
                                sample[f"{img_idx:03d}.{ext}"] = img_f.read()
                        else:
                            print(f"Warning: image not found: {img_path}")

                    sink.write(sample)
                    global_idx += 1

    print(f"Done. Wrote {global_idx} samples to {output_dir}")
    # print(f"Run 'energon prepare {output_dir} --sample-type CrudeWebdataset' to finalize.")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert Bagel VLM data to WebDataset tar format")
    parser.add_argument("--jsonl_paths", nargs="+", required=True, help="Paths to jsonl files")
    parser.add_argument(
        "--image_dirs", nargs="+", required=True, help="Image directories (one per jsonl)"
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for tar shards")
    parser.add_argument("--max_count", type=int, default=5000, help="Max samples per shard")
    parser.add_argument("--max_size", type=float, default=9e12, help="Max shard size in bytes")
    parser.add_argument("--train-split", default=1, type=float)
    parser.add_argument("--val-split", default=0, type=float)
    parser.add_argument("--test-split", default=0, type=float)
    parser.add_argument("--num-workers", default=1, type=int)
    args = parser.parse_args()

    assert len(args.jsonl_paths) == len(args.image_dirs), (
        "Must provide same number of jsonl_paths and image_dirs"
    )

    output_dir = convert_vlm_to_wds(
        jsonl_paths=args.jsonl_paths,
        image_dirs=args.image_dirs,
        output_dir=args.output_dir,
        max_count=args.max_count,
        max_size=args.max_size,
    )
    print("Generating Configurations")
    split = [args.train_split, args.val_split, args.test_split]
    generate_configs(EPath(output_dir), split, num_workers=args.num_workers)
    print("Configurations Generated")


if __name__ == "__main__":
    main()


"""
python tools/datasets/bagel/convert_vlm_to_wds.py \
    --jsonl_paths /share/project/zhaoyingli/dataset/bagel_example/vlm/llava_ov_si.jsonl \
    --image_dirs /share/project/zhaoyingli/dataset/bagel_example/vlm/images \
    --output_dir /share/project/zhaoyingli/dataset/bagel_example/vlm/wds/ \
    --max_count 5000
"""

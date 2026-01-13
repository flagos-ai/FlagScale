#!/usr/bin/env python
"""
Extract inference inputs (images, state, task) from a LeRobotDataset.

This script extracts the required inputs from a dataset sample and saves them
in a format that can be used by the inference script.

Usage:
    # Extract from a specific frame index
    python dump_dataset_inputs.py \
        --dataset_root /path/to/dataset \
        --output_dir ./inference_inputs \
        --frame_index 100

    # Extract from a specific episode and frame
    python dump_dataset_inputs.py \
        --dataset_root /path/to/dataset \
        --output_dir ./inference_inputs \
        --episode_index 0 \
        --frame_in_episode 50

    # Extract multiple samples
    python dump_dataset_inputs.py \
        --dataset_root /path/to/dataset \
        --output_dir ./inference_inputs \
        --frame_indices 100 200 300
"""

import argparse
import json
import os
import sys

# Add FlagScale root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from flagscale.train.datasets.lerobot_dataset import LeRobotDataset


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.

    Handles different tensor formats:
    - (C, H, W) - single image
    - (H, W, C) - single image (channel last)
    - (B, C, H, W) - batch, takes first
    - (B, H, W, C) - batch, takes first
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Handle channel-first vs channel-last
    if tensor.dim() == 3:
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:
            # (C, H, W) -> (H, W, C)
            tensor = tensor.permute(1, 2, 0)
        # Now should be (H, W, C)

    # Clamp values to [0, 1] if needed
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    # Convert to [0, 255] uint8
    if tensor.dtype != torch.uint8:
        tensor = (tensor.clamp(0, 1) * 255).byte()

    # Handle grayscale
    if tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)

    # Convert to PIL Image
    to_pil = ToPILImage()
    if tensor.shape[2] == 3:
        # RGB
        img = to_pil(tensor.permute(2, 0, 1))
    else:
        # Grayscale
        img = Image.fromarray(tensor.numpy(), mode="L")

    return img


def extract_sample(
    dataset: LeRobotDataset,
    frame_index: int | None = None,
    episode_index: int | None = None,
    frame_in_episode: int | None = None,
) -> dict:
    """Extract a sample from the dataset.

    Args:
        dataset: LeRobotDataset instance
        frame_index: Global frame index (takes precedence)
        episode_index: Episode index (requires frame_in_episode)
        frame_in_episode: Frame index within episode

    Returns:
        Dictionary with sample data
    """
    if frame_index is not None:
        idx = frame_index
    elif episode_index is not None and frame_in_episode is not None:
        # Find the global index from episode and frame
        episode_info = dataset.meta.episodes.iloc[episode_index]
        idx = episode_info["dataset_from_index"] + frame_in_episode
    else:
        raise ValueError("Must provide either frame_index or (episode_index, frame_in_episode)")

    if idx >= len(dataset):
        raise ValueError(f"Index {idx} out of range (dataset has {len(dataset)} frames)")

    sample = dataset[idx]
    return sample


def dump_sample(
    sample: dict,
    output_dir: Path,
    sample_name: str = "sample",
    image_format: str = "jpg",
    dataset=None,
) -> dict:
    """Save sample data to files.

    Args:
        sample: Sample dictionary from dataset
        output_dir: Directory to save files
        sample_name: Base name for output files
        image_format: Image format ('jpg' or 'png')

    Returns:
        Dictionary with paths to saved files
    """
    saved_paths = {"images": {}, "state": None, "task": None}

    # TODO: A little bit hacky
    image_keys = [k for k in sample.keys() if "images" in k]
    print(f"Found {len(image_keys)} image key(s): {image_keys}")

    for img_key in image_keys:
        img_tensor = sample[img_key]
        img = tensor_to_image(img_tensor)

        filename = img_key.replace(".", "_")
        img_path = output_dir / f"{sample_name}_{filename}.{image_format}"

        img.save(img_path)
        print(f"Saved image: {img_path}")
        saved_paths["images"][img_key] = str(img_path)

    # Extract and save state
    state_keys = [k for k in sample.keys() if "state" in k and "images" not in k]
    if state_keys:
        state_key = state_keys[0]  # Use first state key
        state_tensor = sample[state_key]

        # Ensure it's 2D (batch, dim)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        state_path = output_dir / f"{sample_name}_state.pt"
        torch.save(state_tensor, state_path)
        print(f"Saved state: {state_path} (shape: {state_tensor.shape})")
        saved_paths["state"] = str(state_path)
    else:
        print("Warning: No state found in sample")

    # Extract and save task
    if "task" in sample:
        task = sample["task"]
        if isinstance(task, torch.Tensor):
            task = task.item() if task.numel() == 1 else str(task.tolist())
        elif isinstance(task, list) and len(task) > 0:
            task = task[0] if isinstance(task[0], str) else str(task[0])

        task_path = output_dir / f"{sample_name}_task.txt"
        with open(task_path, "w", encoding="utf-8") as f:
            f.write(str(task))
        print(f"Saved task: {task_path} (content: '{task}')")
        saved_paths["task"] = str(task_path)
    elif "task_index" in sample:
        # Try to get task from task_index
        task_idx = sample["task_index"]
        if isinstance(task_idx, torch.Tensor):
            task_idx = task_idx.item()

        # Get task from dataset metadata
        if dataset is not None and hasattr(dataset, "meta") and hasattr(dataset.meta, "tasks"):
            tasks_df = dataset.meta.tasks
            if task_idx < len(tasks_df):
                task = tasks_df.iloc[task_idx]["task"]
                task_path = output_dir / f"{sample_name}_task.txt"
                with open(task_path, "w", encoding="utf-8") as f:
                    f.write(str(task))
                print(f"Saved task: {task_path} (content: '{task}')")
                saved_paths["task"] = str(task_path)
    else:
        print("Warning: No task found in sample")

    return saved_paths


def get_args():
    parser = argparse.ArgumentParser(description="Extract inference inputs from LeRobotDataset")
    parser.add_argument(
        "--dataset_root", type=str, default=None, help="Local dataset root directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory to save extracted files"
    )
    parser.add_argument(
        "--frame_index", type=int, default=None, help="Global frame index to extract"
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=None,
        help="Episode index (requires --frame_in_episode)",
    )
    parser.add_argument(
        "--frame_in_episode",
        type=int,
        default=None,
        help="Frame index within episode (requires --episode_index)",
    )
    parser.add_argument(
        "--frame_indices",
        type=int,
        nargs="+",
        default=None,
        help="Multiple frame indices to extract",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Image format to save",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help="Video backend to use (default: pyav, more reliable than torchcodec)",
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # Create output directory early
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset_root}")
    dataset = LeRobotDataset(root=args.dataset_root, video_backend=args.video_backend)
    print(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")

    # Determine which samples to extract
    if args.frame_indices:
        indices = args.frame_indices
        sample_names = [f"frame_{idx}" for idx in indices]
    elif args.frame_index is not None:
        indices = [args.frame_index]
        sample_names = [f"frame_{args.frame_index}"]
    elif args.episode_index is not None and args.frame_in_episode is not None:
        # Calculate global index
        episode_info = dataset.meta.episodes[args.episode_index]
        global_idx = episode_info["dataset_from_index"] + args.frame_in_episode
        indices = [global_idx]
        sample_names = [f"episode_{args.episode_index}_frame_{args.frame_in_episode}"]
    else:
        raise ValueError(
            "Must provide --frame_index, --frame_indices, or (--episode_index + --frame_in_episode)"
        )

    # Extract and save samples
    all_paths = []

    for idx, sample_name in zip(indices, sample_names, strict=False):
        print(f"\n{'=' * 60}")
        print(f"Extracting sample {idx} ({sample_name})")
        print(f"{'=' * 60}")

        sample = extract_sample(dataset, frame_index=idx)
        paths = dump_sample(sample, output_dir, sample_name, args.image_format, dataset=dataset)
        all_paths.append({"index": idx, "sample_name": sample_name, "paths": paths})

    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_paths, f, indent=2)
    print(f"Extraction complete! Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

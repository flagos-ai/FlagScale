# Adopted from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/toolkits/multimodal_data_preprocessing/convert_custom_dataset_to_wds_chatml.py
# We must store the path of vision data, not the real data.

import json
import os
import pickle
from argparse import ArgumentParser
from collections.abc import Sequence
from itertools import zip_longest

import numpy as np
import webdataset as wds
import yaml
from tqdm import tqdm
from webdataset.writer import add_handlers, default_handlers

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory


def _generalized_bit_reversal(length_or_indices: int | Sequence[int]) -> Sequence[int]:
    """This function creates a permutation of given length.
    The sequence is created by a recursive divide and interleave algorithm
    to ensure a balanced distribution across ranks.
    It corresponds to a generalized bit reversal permutation, which - for lengths
    of power of two - is the reversed binary representation of the original indices.

    For example for 16 indices, the sequence is:
        [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]

    Visual illustration:
        Step|0|1|2|3|4|5|6|7|8|9|A|B|C|D|E|F|
            |-------------------------------|
           0|X| | | | | | | | | | | | | | | |
           1|X| | | | | | | |X| | | | | | | |
           2|X| | | |X| | | |X| | | | | | | |
           3|X| | | |X| | | |X| | | |X| | | |
           4|X| |X| |X| | | |X| | | |X| | | |
           5|X| |X| |X| | | |X| |X| |X| | | |
           6|X| |X| |X| |X| |X| |X| |X| | | |
           7|X| |X| |X| |X| |X| |X| |X| |X| |
           8|X|X|X| |X| |X| |X| |X| |X| |X| |
           9|X|X|X| |X| |X| |X|X|X| |X| |X| |
          10|X|X|X| |X|X|X| |X|X|X| |X| |X| |
          11|X|X|X| |X|X|X| |X|X|X| |X|X|X| |
          12|X|X|X|X|X|X|X| |X|X|X| |X|X|X| |
          13|X|X|X|X|X|X|X| |X|X|X|X|X|X|X| |
          14|X|X|X|X|X|X|X|X|X|X|X|X|X|X|X| |
          15|X|X|X|X|X|X|X|X|X|X|X|X|X|X|X|X|
    """

    if isinstance(length_or_indices, int):
        indices = list(range(length_or_indices))
    else:
        indices = length_or_indices

    if len(indices) <= 2:
        return indices
    mid = len(indices) // 2
    left = indices[:mid]
    right = indices[mid:]

    left_result = _generalized_bit_reversal(left)
    right_result = _generalized_bit_reversal(right)

    # Interleave the results
    zipped = zip_longest(left_result, right_result)
    result = [item for sublist in zipped for item in sublist if item is not None]
    return result


# align https://github.com/NVIDIA/Megatron-Energon/blob/aa681fdd6f5dc3b2e6ca1354fb0d0c9b08307d65/src/megatron/energon/flavors/webdataset/sharder.py#L191
def split_samples_to_workers(
    total_samples: int, world_size: int, *, rotation_offset: int = 0
) -> Sequence[int]:
    # We split the total number of samples into the number of global workers across all ranks.
    # Note that the global number of workers intentionally stays the same if you
    # divide the number of ranks by N, and multiply the number of workers per rank by N.
    # This allows to reproduce the same global batches with a different number of ranks.

    num_workers = 1

    global_workers = num_workers * world_size

    min_samples_per_worker = int(total_samples / global_workers)
    num_workers_with_more_samples = total_samples % global_workers

    # We are going to compute the samples assigned to each worker on the current rank.
    # This is done in multiple steps.
    # Some of these steps could be collapsed into one, but we keep them separate for clarity:
    # 1. Compute the number of samples per global worker (rotated by rotation_offset,
    #    typically given by previous datasets).
    # 2. Permute the number of samples per global worker by a generalized bit reversal sequence
    # 3. Given the sample counts, compute the start and end indices for each global worker
    # 4. Extract the local worker sample assignments for the current rank.
    # 5. Split the shards based on the start and end indices.

    # 1. Let's compute it globally for all workers first
    num_samples_per_global_worker = []
    for global_worker_idx in range(global_workers):
        if (
            global_worker_idx - rotation_offset + global_workers
        ) % global_workers < num_workers_with_more_samples:
            # This worker gets one more sample
            num_samples_per_global_worker.append(min_samples_per_worker + 1)
        else:
            # This worker gets the minimum number of samples
            num_samples_per_global_worker.append(min_samples_per_worker)

    # 2. Permute the number of samples per global worker
    worker_bitrev_seq = _generalized_bit_reversal(global_workers)

    # The worker_bitrev_seq is the order in which any remainder samples shall
    # be assigned to workers.
    # That means, the x-axis (array index) is the remainder sample index
    # and the y-axis (value) is the global worker index.
    # So we map the y (value) to the old global worker index from the linear sequence.

    new_num_samples_per_global_worker = [-1] * global_workers
    for old_worker_idx, new_worker_idx in enumerate(worker_bitrev_seq):
        new_num_samples_per_global_worker[new_worker_idx] = num_samples_per_global_worker[
            old_worker_idx
        ]
    return new_num_samples_per_global_worker


def convert(
    dataset_dir,
    output_dir,
    json_name,
    sort_function=sorted,
    max_count=10000,
    image_key="images",
    video_key="videos",
    vision_dir=None,
    dp_size=1,
    drop_last=False,
):
    """
    Here we provide an example to convert llava-pretrain dataset to ChatMLSample
    """
    if vision_dir is None:
        vision_dir = dataset_dir
    # Paths to the dataset files
    json_file = os.path.join(dataset_dir, json_name)
    output = os.path.join(output_dir, f"wds-{dp_size}")
    os.makedirs(output, exist_ok=True)

    # support both json and jsonl
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except:
        with open(json_file, "r") as f:
            data = [json.loads(l) for l in f.readlines()]
    data_len = len(data)
    print(f"Loaded {data_len} entries")

    print(f"The first entry in the dataset is {data[0]}")
    if image_key not in data[0]:
        print(f"Warning: {image_key} not found in the first entry")
    if video_key not in data[0]:
        print(f"Warning: {video_key} not found in the first entry")

    # custom webdataset ShardWriter Encoder
    # "jpgs": the key when saving the image, see line 93
    # "videos": the key when saving the video, see line 92
    add_handlers(default_handlers, "jpgs", lambda data: pickle.dumps(data))
    add_handlers(default_handlers, "videos", lambda data: pickle.dumps(data))

    def write_sample(entry, vision_dir, has_idx=None, idx=0):
        # NOTE: read a dataset in sharegpt format
        images_data: list[str] = []
        # NOTE: we support both list and str for image path.
        image_paths = entry.get(image_key, [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        images_data = image_paths

        videoes_data: list[list[str]] = []
        second_per_grid_ts = []

        for video in entry.pop(video_key, []):
            video_noext, _ = os.path.splitext(video)
            frame_folder = os.path.join(vision_dir, video_noext)
            # NOTE: we implicitly require a `${frame_folder}.json`` file containing fps rates of each video
            # otherwise fps will be regarded as `1` by default.
            if os.path.exists(frame_folder + ".json"):
                with open(frame_folder + ".json", "r") as f:
                    fps = float(json.load(f)["fps"])
            else:
                fps = 2.0

            frames: list[str] = []
            for frame in sort_function(os.listdir(frame_folder)):
                # get relative path（remove "vision_dir"）
                relative_path = os.path.relpath(os.path.join(frame_folder, frame), start=vision_dir)
                frames.append(relative_path)

            if len(frames) % 2 == 1:
                frames = frames[:-1]
            videoes_data.append(frames)
            second_per_grid_ts.append(1 / fps)

        if has_idx is None:
            has_idx = "id" in entry
        assert has_idx == ("id" in entry), "All entries should either all contain idx or not."

        sample = {
            "__key__": entry.pop("id", str(idx)),
            "jpgs": images_data,
            "videos": videoes_data,
            "json": json.dumps(
                {"conversations": entry["conversations"], "second_per_grid_ts": second_per_grid_ts}
            ).encode("utf-8"),
        }
        shard_writer.write(sample)

    has_idx = None

    num_per_rank = data_len // dp_size
    left_data_count = data_len % dp_size
    align_data_count = data_len - left_data_count

    real_num_per_rank = split_samples_to_workers(total_samples=data_len, world_size=dp_size)
    print(f"the real_num_per_rank is {real_num_per_rank}")

    left_num_per_rank = [real_num - num_per_rank for real_num in real_num_per_rank]
    cu_left_num_per_rank = np.cumsum(left_num_per_rank)

    origin_index_list = list(range(data_len))
    deal_index_list = []
    with wds.ShardWriter(
        os.path.join(output, "pretrain-%d.tar"), maxcount=max_count, maxsize=9e12
    ) as shard_writer:
        for rank in tqdm(range(dp_size)):
            current_count = 0
            for id in tqdm(range(num_per_rank)):
                current_count += 1
                data_id = id * dp_size + rank
                entry = data[data_id]
                write_sample(entry, vision_dir, has_idx=has_idx, idx=data_id)
                deal_index_list.append(data_id)
            if left_data_count > 0 and left_num_per_rank[rank] > 0:
                current_count += 1
                idx = align_data_count + cu_left_num_per_rank[rank] - 1
                entry = data[idx]
                write_sample(entry, vision_dir, has_idx=has_idx, idx=idx)
                deal_index_list.append(idx)
            assert current_count == real_num_per_rank[rank], (
                f"current count [{current_count}] of data is not equal to real_num_per_rank: {real_num_per_rank[rank]}"
            )
        # Add the assertion to check all indices are covered
        assert sorted(deal_index_list) == origin_index_list, (
            f"deal_index_list: {sorted(deal_index_list)} is not equal to origin_index_list: {origin_index_list}"
        )
    print("Dataset successfully converted to wds")
    return output


def generate_configs(path: EPath, split, shuffle_tars=True, num_workers=1):
    # path = path.absolute()
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]
    split_parts_ratio = [("train", split[0]), ("val", split[1]), ("test", split[2])]
    split_parts_patterns = None

    # NOTE: generate .info.yaml and split.yaml
    _ = BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=split_parts_ratio,
        split_parts_patterns=split_parts_patterns,
        tar_index_only=False,
        shuffle_seed=42 if shuffle_tars else None,
        workers=num_workers,
    )

    # NOTE: dump dataset.yaml
    metadata = {
        "__class__": "ChatMLWebdataset",
        "__module__": "tools.datasets.qwenvl.data.energon.chatml",
        "field_map": {"imgs": "jpgs", "videos": "videos", "conversation": "json"},
    }
    with open(os.path.join(path.url, ".nv-meta", "dataset.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--dataset-root", required=True, type=str)
    argparser.add_argument("--output-root", required=True, type=str)
    argparser.add_argument("--vision-root", default=None, type=str)
    argparser.add_argument("--json", default="dataset.json", type=str)
    argparser.add_argument(
        "--images-key", default="images", type=str, help="The key for images in json"
    )
    argparser.add_argument(
        "--videos-key", default="videos", type=str, help="The key for videos in json"
    )
    argparser.add_argument("--max-samples-per-tar", default=10000, type=float)
    argparser.add_argument("--train-split", default=1, type=float)
    argparser.add_argument("--val-split", default=0, type=float)
    argparser.add_argument("--test-split", default=0, type=float)
    argparser.add_argument("--shuffle-tars", action="store_true")
    argparser.add_argument("--num-workers", default=1, type=int)
    argparser.add_argument("--dp-size", default=1, type=int)
    argparser.add_argument(
        "--drop-last", action="store_true", help="This option is not used currently."
    )
    args = argparser.parse_args()
    print(f"=======input args=======\n{args}\n=======input args=======\n")
    output_dir = convert(
        args.dataset_root,
        args.output_root,
        args.json,
        max_count=args.max_samples_per_tar,
        image_key=args.images_key,
        video_key=args.videos_key,
        vision_dir=args.vision_root,
        dp_size=args.dp_size,
        drop_last=args.drop_last,
    )
    print("Generating Configurations")
    # NOTE: split_ratio: train/val/test
    split = [args.train_split, args.val_split, args.test_split]
    generate_configs(
        EPath(output_dir), split, shuffle_tars=args.shuffle_tars, num_workers=args.num_workers
    )
    print("Configurations Generated")

# Adopted from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/toolkits/multimodal_data_preprocessing/convert_custom_dataset_to_wds_chatml.py
# We must store the path of vision data, not the real data.

import json
import math
import os
import pickle
from argparse import ArgumentParser

import webdataset as wds
import yaml
from tqdm import tqdm
from webdataset.writer import add_handlers, default_handlers

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory


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
        image_data: list[str] = []
        # NOTE: we support both list and str for image path.
        image_paths = entry.get(image_key, [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        image_data = image_paths

        video_data: list[list[str]] = []
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
            video_data.append(frames)
            second_per_grid_ts.append(1 / fps)

        if has_idx is None:
            has_idx = "id" in entry
        assert has_idx == ("id" in entry), "All entries should either all contain idx or not."

        sample = {
            "__key__": entry.pop("id", str(idx)),
            "jpgs": image_data,
            "videos": video_data,
            "json": json.dumps(
                {"conversations": entry["conversations"], "second_per_grid_ts": second_per_grid_ts}
            ).encode("utf-8"),
        }
        shard_writer.write(sample)

    has_idx = None
    if drop_last:
        num_per_rank = data_len // dp_size
        left_data_count = data_len % dp_size
        with wds.ShardWriter(
            os.path.join(output, "pretrain-%d.tar"), maxcount=max_count, maxsize=9e9
        ) as shard_writer:
            for rank in tqdm(range(dp_size)):
                for id in tqdm(range(num_per_rank)):
                    data_id = id * dp_size + rank
                    entry = data[data_id]
                    write_sample(entry, vision_dir, has_idx=has_idx, idx=data_id)
            if left_data_count > 0:
                for idx, entry in enumerate(data[data_len - left_data_count :]):
                    write_sample(
                        entry, vision_dir, has_idx=has_idx, idx=data_len - left_data_count + idx
                    )
    else:
        num_per_rank = math.ceil(data_len / dp_size)
        with wds.ShardWriter(
            os.path.join(output, "pretrain-%d.tar"), maxcount=max_count, maxsize=9e9
        ) as shard_writer:
            for rank in tqdm(range(dp_size)):
                for id in tqdm(range(num_per_rank)):
                    data_id = id * dp_size + rank
                    if data_id >= data_len:
                        break
                    entry = data[data_id]
                    write_sample(entry, vision_dir, has_idx=has_idx, idx=data_id)

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
    argparser.add_argument("--drop-last", action="store_true")
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

import argparse
import ast
import copy
import json
import os
import random
import re
import warnings

import torch
import transformers
from llava.model.builder import load_pretrained_model
from llava.train.train import DataArguments, make_supervised_data_module
from torch.utils.data import Dataset

from flagscale.compress.compressor import compress, prepare_config

warnings.filterwarnings("ignore")


class CusDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.indices = list(range(len(ds)))
        self.column_names = self
        self.calibration = self

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        cur_ds = self.ds[index]
        tmp_ = cur_ds.pop("image")
        # cur_ds["modalities"] = [tmp_[0][2]]
        cur_ds["image_sizes"] = [torch.tensor(tmp_[0][1])]
        cur_ds["images"] = [tmp_[0][0].unsqueeze(0).to(torch.float16)]
        del tmp_
        cur_ds.pop("id")
        cur_ds["input_ids"] = cur_ds["input_ids"].unsqueeze(0)
        cur_ds["labels"] = cur_ds["labels"].unsqueeze(0)
        return cur_ds

    def shuffle(self):
        random.shuffle(self.indices)
        return self

    def select(self, samples_idx):
        indices = []
        for idx in samples_idx:
            indices.append(self.indices[idx])
        self.indices = indices
        return self


def prepare_model(cfg):
    origin_config = json.load(open(os.path.join(cfg.model["model_path"], "config.json"), "r"))
    origin_vocab_size = origin_config["vocab_size"]
    tokenizer, model, _, _ = load_pretrained_model(
        cfg.model["model_path"],
        None,
        cfg.model["model_name"],
        device_map=cfg.model["device_map"],
        attn_implementation="sdpa",
        multimodal=True,
    )
    model.resize_token_embeddings(origin_vocab_size)
    return model, tokenizer


def prepare_dataset(cfg, model, tokenizer):
    if cfg.data.data_path is None:
        return None
    new_data_args = copy.deepcopy(cfg.data)
    new_data_args.pop("num_calibration_samples")
    new_data_args.pop("max_seq_length")
    new_data_args.pop("tokenzier_args")

    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_dict(new_data_args)[0]
    vision_tower = model.get_vision_tower()

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    data_args.image_folder = "/"
    data_args.mm_use_im_start_end = model.config.mm_use_im_start_end

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    if data_args.image_grid_pinpoints is not None:
        if (
            isinstance(data_args.image_grid_pinpoints, str)
            and "x" in data_args.image_grid_pinpoints
        ):
            try:
                patch_size = data_args.image_processor.size[0]
            except Exception:
                patch_size = data_args.image_processor.size["shortest_edge"]

            assert patch_size in [
                224,
                336,
                384,
                448,
                512,
            ], "patch_size should be in [224, 336, 384, 448, 512]"
            # Use regex to extract the range from the input string
            matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
            range_start = tuple(map(int, matches[0]))
            range_end = tuple(map(int, matches[-1]))
            # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
            grid_pinpoints = [
                (i, j)
                for i in range(range_start[0], range_end[0] + 1)
                for j in range(range_start[1], range_end[1] + 1)
            ]
            # Multiply all elements by patch_size
            data_args.image_grid_pinpoints = [
                [dim * patch_size for dim in pair] for pair in grid_pinpoints
            ]
        elif isinstance(data_args.image_grid_pinpoints, str):
            data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
    dataset = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    ds = CusDataset(dataset["train_dataset"])
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()
    cfg = prepare_config(args.config_path)
    model, tokenizer = prepare_model(cfg)
    dataset = prepare_dataset(cfg, model, tokenizer)
    compress(cfg, dataset=dataset, model=model)

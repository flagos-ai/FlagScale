import argparse
import random

import torch
from torch.utils.data import Dataset

from megatron.core.datasets.indexed_dataset import IndexedDataset

from flagscale.compress.compressor import compress, prepare_config


class CusDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.indices = list(range(len(ds)))
        self.column_names = {"input_ids": self}
        self.calibration = self

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        cur_ds = self.ds[index]
        slice_idx = cur_ds.tolist().index(151643)  ### mask padding
        return {
            "input_ids": torch.tensor(cur_ds[:slice_idx]).unsqueeze(0),
            "attention_mask": torch.ones(slice_idx, dtype=torch.int64),
        }

    def shuffle(self):
        random.shuffle(self.indices)
        return self

    def select(self, samples_idx):
        indices = []
        for idx in samples_idx:
            indices.append(self.indices[idx])
        self.indices = indices
        return self


def prepare_dataset(cfg):
    print(cfg)
    if cfg.data.data_path is None:
        return None
    ds = IndexedDataset(cfg.data.data_path, mmap=True)
    dataset = CusDataset(ds)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()
    cfg = prepare_config(args.config_path)
    dataset = prepare_dataset(cfg)
    compress(cfg, dataset=dataset)

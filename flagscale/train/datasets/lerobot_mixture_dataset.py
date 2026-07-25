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

import hashlib
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from flagscale.logger import logger
from flagscale.train.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def safe_hash(input_tuple) -> int:
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)
    seed = int(sha256.hexdigest(), 16)
    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class LeRobotMixtureDataset(Dataset):
    """Weighted mixture of multiple LeRobotDatasets.

    Samples a dataset based on weights, then samples a step within that dataset,
    using deterministic per-index RNG for DataLoader safety.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotDataset, float]],
        mode: str = "train",
        balance_dataset_weights: bool = True,
        seed: int = 42,
    ):
        datasets: list[LeRobotDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            if len(dataset) == 0:
                logger.warning("Skipping empty dataset")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture.")

        self.datasets = datasets
        self.seed = seed
        self.mode = mode

        self._dataset_lengths = np.array([len(ds) for ds in self.datasets])
        logger.info(f"Dataset lengths: {self._dataset_lengths}")

        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        if balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths

        self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
        else:
            self._dataset_sampling_weights /= weights_sum

        logger.info(f"Normalized sampling weights: {self._dataset_sampling_weights}")

        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
        if not np.any(self._primary_dataset_indices):
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        self.set_epoch(0)

        self._meta = self.datasets[0].meta
        self._merged_stats = self._merge_stats()
        self._len = self._compute_len()

    @property
    def meta(self) -> LeRobotDatasetMetadata:
        return self._meta

    @property
    def merged_stats(self) -> dict:
        return self._merged_stats

    @property
    def num_frames(self) -> int:
        return sum(ds.num_frames for ds in self.datasets)

    @property
    def num_episodes(self) -> int:
        return sum(ds.num_episodes for ds in self.datasets)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def sample_step(self, index: int) -> tuple[LeRobotDataset, int]:
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        dataset_index = rng.choice(len(self.datasets), p=self._dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        sample_index = int(rng.integers(0, len(dataset)))
        return dataset, sample_index

    def __getitem__(self, index: int) -> dict:
        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                dataset, sample_index = self.sample_step(index)
                item = dataset[sample_index]
                item["_dataset_name"] = dataset.root.name
                item["_image_feature_keys"] = list(
                    getattr(dataset, "image_feature_keys", dataset.meta.camera_keys)
                )
                # Check for None values that would crash default_collate
                # none_keys = [k for k, v in item.items() if v is None]
                # if none_keys:
                #     raise ValueError(
                #         f"None values in dataset={dataset.root.name} "
                #         f"episode={item.get('episode_index', '?')} "
                #         f"sample_index={sample_index} keys={none_keys}"
                #     )
                return item
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}"
                    )
                    index = random.randint(0, len(self) - 1)
                else:
                    logger.error(f"All {max_retries} attempts failed for index {index}")
                    raise last_exception

    def __len__(self) -> int:
        return self._len

    def _compute_len(self) -> int:
        if len(self.datasets) == 0:
            return 0

        valid_indices = (self._dataset_lengths > 0) & (~np.isnan(self._dataset_lengths))
        valid_weights = (self._dataset_sampling_weights > 0) & (
            ~np.isnan(self._dataset_sampling_weights)
        )
        valid_indices = valid_indices & valid_weights
        if not np.any(valid_indices):
            return 0

        primary_and_valid = self._primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            if np.any(valid_indices):
                return int(self._dataset_lengths[valid_indices].max())
            return 0

        ratios = (self._dataset_lengths / self._dataset_sampling_weights)[primary_and_valid]
        valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
        if len(valid_ratios) == 0:
            return 0
        return int(valid_ratios.max())

    def _merge_stats(self) -> dict:
        weights = self._dataset_sampling_weights
        all_stats = [ds.meta.stats for ds in self.datasets]

        all_keys = set()
        for stats in all_stats:
            all_keys.update(stats.keys())

        merged = {}
        for key in all_keys:
            per_dataset = []
            per_dataset_weights = []
            for i, stats in enumerate(all_stats):
                if key in stats:
                    per_dataset.append(stats[key])
                    per_dataset_weights.append(weights[i])

            if not per_dataset:
                continue

            w = np.array(per_dataset_weights)
            w = w / w.sum()

            merged[key] = self._merge_single_key_stats(per_dataset, w)

        return merged

    @staticmethod
    def _merge_single_key_stats(
        per_dataset: list[dict[str, torch.Tensor]],
        weights: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        merged = {}

        stacked_means = torch.stack([s["mean"] for s in per_dataset], dim=0)
        sample_w = torch.tensor(weights, dtype=stacked_means.dtype)
        sample_w = sample_w.reshape(-1, *([1] * (stacked_means.dim() - 1)))
        weighted_mean = (sample_w * stacked_means).sum(dim=0)

        for sk in per_dataset[0].keys():
            tensors = [s[sk] for s in per_dataset]
            stacked = torch.stack(tensors, dim=0)
            w = (
                sample_w
                if stacked.shape == stacked_means.shape
                else torch.tensor(weights, dtype=stacked.dtype).reshape(
                    -1, *([1] * (stacked.dim() - 1))
                )
            )

            if sk == "mean":
                merged[sk] = weighted_mean
            elif sk == "std":
                variances = stacked**2 + stacked_means**2
                weighted_var = (w * variances).sum(dim=0) - weighted_mean**2
                merged[sk] = weighted_var.clamp(min=0).sqrt()
            elif sk == "min" or sk.startswith("q0"):
                merged[sk] = stacked.min(dim=0).values
            elif sk == "max" or sk.startswith("q9"):
                merged[sk] = stacked.max(dim=0).values
            else:
                merged[sk] = (w * stacked).sum(dim=0)

        return merged

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

from typing import Optional, Tuple, Union

import numpy

from megatron.core.datasets.indexed_dataset import IndexedDataset


class ConcatedIndexedDataset(IndexedDataset):
    def __init__(self, datasets) -> None:
        self.path_prefix = datasets[-1].path_prefix
        self.datasets = datasets
        self.offsets = [0]
        for dataset in datasets:
            self.offsets.append(self.offsets[-1] + len(dataset))

    def __del__(self) -> None:
        for dataset in self.datasets:
            del dataset

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        for i, size in enumerate(self.offsets[1:]):
            if idx < size:
                break
        return self.datasets[i][idx - self.offsets[i]]

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        for i, size in enumerate(self.offsets[1:]):
            if idx < size:
                break

        return self.datasets[i].get(idx - self.offsets[i], offset, length)

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        return numpy.concatenate([dataset.sequence_lengths for dataset in self.datasets])

    @property
    def document_indices(self) -> numpy.ndarray:
        return numpy.concatenate(
            [
                dataset.document_indices[:-1] + offset
                for dataset, offset in zip(self.datasets, self.offsets)
            ]
            + [numpy.array([len(self)])]
        )

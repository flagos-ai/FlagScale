# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


import numpy as np
import torch

from megatron.energon import (
    Batch,
    DefaultTaskEncoder,
    InterleavedSample
)
from megatron.training import get_args


def print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()


@dataclass
class AnyResTaskSample:
    __key__: str
    __subflavors__: Dict
    input_ids: torch.Tensor
    input_ids_shape: torch.Tensor
    labels: torch.Tensor
    labels_shape: torch.Tensor
    images: List[torch.Tensor]
    image_sizes: List[torch.Tensor]
    modalities: List[torch.Tensor]

# Typing for the resulting batch data after encode_batch()
@dataclass
class AnyResTaskBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    input_ids: torch.Tensor
    input_ids_shape: torch.Tensor
    labels: torch.Tensor
    labels_shape: torch.Tensor
    images: torch.Tensor
    image_sizes: torch.Tensor
    split_image_sizes: torch.Tensor
    modalities: torch.Tensor


class AnyResTaskEncoder(DefaultTaskEncoder[InterleavedSample, InterleavedSample, AnyResTaskBatch, dict]):
    """
    A task encoder for anyres.
    This encoder is just a wrapper around data that has already been made.
    Production data can be referenced to LLaVA-NeXT and can be input into vision tower.
    """

    def __init__(
        self
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by
        # overwriting the `batch` method)
        super().__init__()

        self.args = get_args()

    def encode_sample(self, sample: InterleavedSample):
        if not isinstance(sample, InterleavedSample):
            raise ValueError(f"This encoder only supports InterleavedSample, but got {type(sample)}.")
        yield self.encode_interleaved(sample)

    def encode_interleaved(self, sample: InterleavedSample):
        modalities = None
        # The sequence is #[input_ids, labels, images, image_sizes]
        if len(sample.sequence) == 4:
            input_ids, labels, images, image_sizes = sample.sequence
        # The sequence is #[input_ids, labels, images, image_sizes, modalities]
        elif len(sample.sequence) == 5:
            input_ids, labels, images, image_sizes, modalities = sample.sequence
        else:
            assert ValueError("The sequence must have 4 or 5 elements, but got {len(sample.sequence)}.")

        # process modalities to tensor
        modalities_list = []
        for modality in modalities:
            # image, video, text to 0, 1, 2
            if modality == "image":
                modalities_list.append(torch.tensor([0]))
            elif modality == "video":
                modalities_list.append(torch.tensor([1]))
            elif modality == "text":
                modalities_list.append(torch.tensor([2]))
            else:
                raise ValueError(f"Unsupported modality: {modalities}")

        modalities = modalities_list
        return AnyResTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            input_ids=input_ids,
            input_ids_shape=torch.tensor(input_ids.shape),
            labels=labels,
            labels_shape=torch.tensor(labels.shape),
            images=images,
            image_sizes=image_sizes,
            modalities=modalities
        )

    def batch(self, samples: List[AnyResTaskSample]) -> AnyResTaskBatch:
        input_ids = torch.cat([s.input_ids.flatten() for s in samples], dim=0)
        input_ids_shape = torch.stack([s.input_ids_shape for s in samples], dim=0)
        labels = torch.cat([s.labels.flatten() for s in samples], dim=0)
        labels_shape = torch.stack([s.labels_shape for s in samples], dim=0)
        # Double loop
        images = torch.cat([image.flatten() for s in samples for image in s.images], dim=0)
        split_image_sizes = torch.stack([torch.tensor(image.shape) for s in samples for image in s.images], dim=0)
        # Adapt video data by decord
        image_sizes = torch.stack([image_sizes if len(image_sizes.shape) == 1 else torch.tensor((1, image_sizes.item())) for s in samples for image_sizes in s.image_sizes], dim=0)
        modalities = torch.stack([modalities for s in samples for modalities in s.modalities], dim=0)

        batch = AnyResTaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            input_ids=input_ids,
            input_ids_shape=input_ids_shape,
            labels=labels,
            labels_shape=labels_shape,
            images=images,
            image_sizes=image_sizes,
            split_image_sizes=split_image_sizes,
            modalities=modalities
        )

        return batch

    def encode_batch(self, batch: AnyResTaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw

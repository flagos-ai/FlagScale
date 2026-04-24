# Copyright (c) 2024, BAAI. All rights reserved.

import logging
import os
import time

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy
import torch

from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    _get_ltor_masks_and_position_ids,
)
from megatron.core.datasets.indexed_dataset import IndexedDataset, get_bin_path, get_idx_path
from megatron.core.datasets.utils import Split

logger = logging.getLogger(__name__)


@dataclass
class SFTDatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core SFT datasets"""

    apply_sft_dataset_separated_loss_mask_if_existed: bool = None
    """Option to apply separated loss mask files"""


class SFTDataset(GPTDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: SFTDatasetConfig,
    ) -> None:
        self.config = config
        self.apply_sft_dataset_separated_loss_mask_if_existed = (
            config.apply_sft_dataset_separated_loss_mask_if_existed
        )
        self.loss_mask_dataset = None

        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

        self._build_loss_mask_dataset()

    def _build_loss_mask_dataset(self) -> None:
        """
        Load Loss Mask IndexedDataset
        """
        path_prefix = None
        base_prefix = "_text_document"
        loss_mask_prefix = "_loss_mask_document"
        if self.dataset_path.endswith(base_prefix):
            path_prefix = self.dataset_path[: -len(base_prefix)] + loss_mask_prefix
        if self.apply_sft_dataset_separated_loss_mask_if_existed and path_prefix:
            idx_path = get_idx_path(path_prefix)
            bin_path = get_bin_path(path_prefix)
            if os.path.exists(idx_path) and os.path.exists(bin_path):
                self.loss_mask_dataset = IndexedDataset(
                    path_prefix, multimodal=False, mmap=self.config.mmap_bin_files
                )

                print(f"> Used Dataset: aux_loss_mask ...")
                if self.loss_mask_dataset is not None:
                    assert len(self.dataset) == len(
                        self.loss_mask_dataset
                    ), f"Samples are not equal, ({len(self.dataset)} != {len(self.loss_mask_dataset)})"

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        # aux dataset
        aux_loss_mask, _ = self._query_document_sample_shuffle_indices_aux_dataset(
            self.loss_mask_dataset, idx
        )
        if aux_loss_mask is not None:
            if idx % 100 == 0:
                print(f"> Used aux_loss_mask at current sample={idx} ...")
            loss_mask = torch.from_numpy(aux_loss_mask).float()[1:].contiguous()

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    def _query_document_sample_shuffle_indices_aux_dataset(
        self, aux_dataset, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the aux ids and document ids for a given index

        Args:
            aux_dataset (int): The aux dataset
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        if aux_dataset is None:
            return (None, None)

        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                aux_dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + 1,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + 1
                sample_parts.append(
                    aux_dataset.get(self.document_index[i], offset=offset, length=length)
                )

        return (
            numpy.array(numpy.concatenate(sample_parts), dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )

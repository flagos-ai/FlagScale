# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict

import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset

IGNORE_INDEX = -100


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
            A jsonl line can contain multiple conversations packed together into on list. Each
            conversation starts with the system role, and conversations can have multiple turns
            of the user and assistant roles.
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list:
        return self.dataset[idx]["messages"]


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _split_conversations(merged_conversations):
        conversations = []
        current = []
        for message in merged_conversations:
            if current and message["role"] == "system":
                conversations.append(current)
                current = []
            current.append(message)
        if current:
            conversations.append(current)
        return conversations

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length

        merged_conversations = self.dataset[int(self.indices[idx % len(self.indices)])]
        split_conversations = self._split_conversations(merged_conversations)

        pack_tokens = []
        pack_targets = []
        cu_seqlens = [0]
        pad = tokenizer.pad
        assert not self.config.reset_position_ids
        # TODO(duncan): Track number of convs dropped and/or truncated and amount of end-padding
        for conversation in split_conversations:

            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False
            )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()

            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)

            if self.config.context_parallel_size > 1:
                pad_granularity = self.config.context_parallel_size * 2
                mod_token_count = len(pack_tokens) % pad_granularity
                if mod_token_count != 0:
                    pad_len = pad_granularity - mod_token_count
                    pack_tokens.extend([pad] * pad_len)
                    pack_targets.extend([pad] * pad_len)

            # TODO(duncan): Consider also padding to multiple of number of tokens here. This might
            # be needed for efficiency (and potentially set via command-line argument).

            cu_seqlens.append(len(pack_tokens))

            # Handle any necessary truncation
            if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
                # Truncate on the right
                max_body = pack_length
                pack_tokens = pack_tokens[:max_body]
                pack_targets = pack_targets[:max_body]
                pack_tokens.append(pad)
                pack_targets.append(pad)
                cu_seqlens[-1] = len(pack_tokens) - 1
                break

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            pack_tokens.extend([pad] * pad_len)
            pack_targets.extend([pad] * pad_len)
            cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1

        # Align and convert to tensors
        input_ids    = torch.tensor(pack_tokens[:-1],  dtype=torch.int64)
        labels       = torch.tensor(pack_targets[1:], dtype=torch.int64)

        assert not self.config.create_attention_mask and not self.config.reset_attention_mask
        assert len(cu_seqlens) >= 2
        return sft_batch(input_ids, labels, cu_seqlens, pad)


def packed_position_ids(cu_seqlens, length=None):
    """Restart at zero at each THD boundary, including on later pipeline stages."""
    boundaries = cu_seqlens.to(torch.long)
    if length is None:
        length = int(boundaries[-1].item())
    starts = boundaries[:-1].repeat_interleave(boundaries.diff(), output_size=length)
    return torch.arange(length, device=boundaries.device) - starts


def sft_batch(tokens, labels, boundaries, pad):
    """Build the common THD training sample from aligned next-token targets."""
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": ((labels != IGNORE_INDEX) & (labels != pad)).float(),
        "position_ids": packed_position_ids(cu_seqlens, tokens.numel()),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": cu_seqlens.diff().max(),
    }

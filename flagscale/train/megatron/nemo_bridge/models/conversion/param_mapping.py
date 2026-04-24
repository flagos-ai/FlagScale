# Copyright (c) 2025, BAAI. All rights reserved.

import torch
import torch.nn as nn
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.models.conversion.param_mapping import ColumnParallelMapping as OriginalColumnParallelMapping
from megatron.bridge.models.conversion.param_mapping import AutoMapping as OriginalAutoMapping
from megatron.bridge.models.conversion.param_mapping import QKVMapping as OriginalQKVMapping
from megatron.bridge.models.conversion.param_mapping import (
    MegatronParamMapping,
    RowParallelMapping,
    ReplicatedMapping,
    GatedMLPMapping,
)


import logging
logger = logging.getLogger(__name__)

def col_padding_size(hf_weight: torch.Tensor, mcore_weight: torch.Tensor, tp_size: int):
    hf_size = hf_weight.shape[0]
    mcore_size = mcore_weight.shape[0] * tp_size
    full_word = {}
    is_rank0 = torch.distributed.get_rank() == 0
    # Cut out extra padding we don't need
    if hf_size > mcore_size:
        full_word = hf_weight[0:mcore_size, :]
        if is_rank0:
            print(f"> padding TP-ColumnParallelfrom {hf_size} to {mcore_size}")

    # Expanding embedding to larger size by replicating final entry
    elif hf_size < mcore_size:
        padding_size = mcore_size - hf_size

        full_word = torch.cat((hf_weight, hf_weight[-1].unsqueeze(0).expand(padding_size, -1)))
        if is_rank0:
            print(f"> padding TP-ColumnParallelfrom {hf_size} to {mcore_size}")
    # Same size!
    else:
        full_word = hf_weight
    return full_word


class ColumnParallelMapping(OriginalColumnParallelMapping):
    """
    Mapping for column-parallel linear and embedding weights.

    """
    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Split weight along dim 0 and distribute to TP ranks."""
        # Some parameters are named with global expert number, e.g. experts.weight15,
        # normalize it to experts.weight0, note we are only use the shape, dtype, device info,
        # not the actual value, so it is safe to do this.
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)

        if self.tp_size == 1:
            full_weight = col_padding_size(hf_weights, target_param, self.tp_size)
            return full_weight

        # On rank 0, check for divisibility and split
        if self.tp_rank == 0:
            if hf_weights is None:
                raise ValueError("hf_weights should not be None on rank 0")

            if hf_weights.dtype != target_param.dtype:
                logger.warning(
                    f"WARNING: Dtype mismatch between HuggingFace weights and Megatron module. "
                    f"HF dtype: {hf_weights.dtype}. Megatron dtype: {target_param.dtype}. "
                    f"Casting HF weights to Megatron dtype. THIS MAY RESULT IN A LOSS OF PRECISION. "
                )
                hf_weights = hf_weights.to(target_param.dtype)

            full_weight = col_padding_size(hf_weights, target_param, self.tp_size)
            full_size = full_weight.shape[0]
            if full_size % self.tp_size != 0:
                raise ValueError(
                    f"Cannot evenly split dimension 0 size {full_size} across {self.tp_size} TP ranks"
                )
            splits = torch.chunk(full_weight, self.tp_size, dim=0)
        else:
            splits = None

        # Scatter to all ranks. Each rank gets its sharded shape from its module.
        return self.scatter_to_tp_ranks(
            splits, target_param.shape, target_param.dtype, target_param.device
        )

class AutoMapping(OriginalAutoMapping):
    def _get_or_create_mapping(self, parallelism_type: str) -> MegatronParamMapping[torch.Tensor]:
        """Get or create the appropriate mapping for the given type."""
        if parallelism_type == "column":
            return ColumnParallelMapping(self.megatron_param, self.hf_param)
        elif parallelism_type == "row":
            return RowParallelMapping(self.megatron_param, self.hf_param)
        elif parallelism_type == "replicated":
            return ReplicatedMapping(self.megatron_param, self.hf_param)
        else:
            raise ValueError(f"Unknown parallelism type: {parallelism_type}")

class QKVMapping(OriginalQKVMapping):
    def __init__(self, megatron_param: str, q: str, k: str, v: str):
        super().__init__(megatron_param, q, k, v)
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)


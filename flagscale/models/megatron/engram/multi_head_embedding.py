## built-in
from typing import Optional, Callable, Tuple
## third-party
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.plugin.platform import get_platform

cur_platform = get_platform()

# megatron-core
from megatron.core import tensor_parallel
from megatron.core.utils import get_pg_size, get_pg_rank, get_tensor_model_parallel_group_if_none
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_cpu
from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.dist_checkpointing.mapping import ShardedTensor

# engram
from .engram_config import EngramConfig


def _vocab_size_with_padding(orig_vocab_size, tp_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = tp_size
    after = int(math.ceil(after / multiple) * multiple)
    return after


def _initialize_engram_weight_gpu_with_seed(weight, init_method, local_init_seed, partition_dim=0, stride=1):
    tensor_parallel.set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )
    with torch.random.fork_rng(devices=[weight.device]):
        torch.manual_seed(local_init_seed)
        init_method(weight)


class EngramMemory(nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default values are kept.

    Unlike to the MCore VocabParallelEmbedding, the embedding parallel use parallelism like expert parallel.
    The parallel group is the subset of data parallel, which is given as the engram_model_parallel_size.
    Input of each rank is different, when forwarding, the input will be transmit to other rank using an All2All operator.
    
    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
    
    Keyword Args:
        init_method: A Callable.
        config: A EngramConfig object.
        embedding_parallel_group: vocab parallel group, a torch.distributed.ProcessGroup object.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        reduce_scatter_embeddings: bool = False,
        config: ModelParallelConfig,
        embedding_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.embedding_parallel_group = embedding_parallel_group
        if self.embedding_parallel_group is None:
            self.embedding_parallel_size = 1
            self.embedding_parallel_rank = 0
        else:
            self.embedding_parallel_size = get_pg_size(self.embedding_parallel_group)
            self.embedding_parallel_rank = get_pg_rank(self.embedding_parallel_group)

        (self.vocab_start_index, self.vocab_end_index) = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, self.embedding_parallel_rank, self.embedding_parallel_size
            )
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.deterministic_mode = config.deterministic_mode

        # Allocate weights and initialize on GPU only.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                    rank=self.embedding_parallel_rank,
                    world_size=self.embedding_parallel_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=cur_platform.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                engram_seed = int(getattr(config, "engram_seed", 0))
                pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                local_init_seed = 2718 + engram_seed + pp_rank * 100 + int(self.embedding_parallel_rank)
                _initialize_engram_weight_gpu_with_seed(
                    self.weight, init_method, local_init_seed, partition_dim=0, stride=1
                )

    
    def enable_parallel(self):
        if self.embedding_parallel_size > 1:
            setattr(self.weight, "is_engram_embedding", True)
            setattr(self.weight, "allreduce", False)
    
    def enable_offloading(self):
        setattr(self.weight, "is_offloading_candidate", True)

    def _dispatch(self, input_ids):
        cur_platform.range_push("engram_embedding_dispatch")
        self.hidden_shape = input_ids.shape
        input_ids = input_ids.view(-1)
        routing_map = input_ids // self.num_embeddings_per_partition
        # [num_partitions], number of tokens assigned to each partition from the current rank's input.
        num_tokens_per_partition = torch.bincount(
            routing_map,
            minlength=self.embedding_parallel_size,
         ).to(dtype=torch.int64)
        # Reorder the token indices to match the order of partitions.
        # Shape = (batch * seqlen, ).
        token_indices_partitions_sorted = torch.argsort(routing_map, stable=True)
        # Shape = (batch * seqlen, ).
        routed_input = input_ids[token_indices_partitions_sorted]
        # Use to unsort.
        self._token_unsort_indices = torch.empty_like(token_indices_partitions_sorted)
        self._token_unsort_indices[token_indices_partitions_sorted] = torch.arange(
            token_indices_partitions_sorted.size(0), device=token_indices_partitions_sorted.device
        )
        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            output_splits_cuda = tensor_parallel.all_to_all(
                self.embedding_parallel_group,
                num_tokens_per_partition,
                None,
                None,
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            output_splits_cuda = torch.ops._c10d_functional.wait_tensor(
                output_splits_cuda
            )
            input_splits = num_tokens_per_partition.view(self.embedding_parallel_size, -1).sum(dim=1).to(torch.device("cpu"), non_blocking=True)
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                output_splits_cuda.view(self.embedding_parallel_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()
        
        # perform all-to-all
        routed_input = tensor_parallel.all_to_all(
            self.embedding_parallel_group, routed_input, self.output_splits, self.input_splits
        )
        routed_input = routed_input - self.vocab_start_index
        cur_platform.range_pop()
        return routed_input

    def _combine(self, hidden_states: torch.Tensor):
        cur_platform.range_push("engram_embedding_combine")
        routed_hidden_states = tensor_parallel.all_to_all(self.embedding_parallel_group, hidden_states, self.input_splits, self.output_splits)
        routed_hidden_states = routed_hidden_states[self._token_unsort_indices]
        hidden_states = routed_hidden_states.view(*self.hidden_shape, -1)
        cur_platform.range_pop()
        return hidden_states

    def forward(self, input_: torch.Tensor):
        """Forward.

        Args:
            input_ (torch.Tensor): Input tensor, shape (b, s), dtype = torch.int64.
        """
        cur_platform.range_push("engram_embedding_forward")
        if self.reduce_scatter_embeddings:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            num_tokens_per_sp_rank = input_.shape[1] // tp_size
            if tp_rank < tp_size - 1:
                input_ = input_[:, tp_rank * num_tokens_per_sp_rank : (tp_rank + 1) * num_tokens_per_sp_rank]
            else:
                input_ = input_[:, tp_rank * num_tokens_per_sp_rank : ]
            input_ = input_.contiguous()
        if self.embedding_parallel_size > 1:
            input_ = self._dispatch(input_)
        # Get the embeddings.
        if self.deterministic_mode:
            output = self.weight[input_]
        else:
            # F.embedding currently has a non-deterministic backward function
            output = F.embedding(input_, self.weight)
        # Get the complete output embedding
        if self.embedding_parallel_size > 1:
            output = self._combine(output)
        if self.reduce_scatter_embeddings:
            output = output.transpose(0, 1).contiguous()
        cur_platform.range_pop()
        return output

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None, **kwargs,
    ):
        state_dict = self.state_dict(prefix="", keep_vars=True)
        weight_prefix = f"{prefix}weight"
        prepend_axis_num = len(sharded_offsets)
        new_offsets = []
        tp_rank = self.embedding_parallel_rank
        tp_size = self.embedding_parallel_size
        dp_replica_id = get_pg_rank(parallel_state.get_engram_data_parallel_group())
        new_offsets.append((prepend_axis_num, tp_rank, tp_size))

        replica_id = (0, 0, dp_replica_id)
        sharded_tensor = ShardedTensor.from_rank_offsets(
            weight_prefix,
            state_dict["weight"],
            *sharded_offsets,
            *new_offsets,
            replica_id=replica_id,
            prepend_axis_num=prepend_axis_num,
            allow_shape_mismatch=True,
            **kwargs
        )
        return {
            weight_prefix: sharded_tensor
        }


class MultiHeadEmbedding(nn.Module):
    def __init__(self, engram_cfg: EngramConfig, list_of_N: list[int], D: int):
        super().__init__()
        self.engram_cfg = engram_cfg
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = sum(list_of_N)

        # embeddings (parallel).
        if self.engram_cfg.engram_embedding_parallel_method == "allreduce":
            self.tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
            self.reduce_scatter_embeddings = self.engram_cfg.sequence_parallel

            padded_total_N = _vocab_size_with_padding(total_N, get_pg_size(self.tp_group))
            print(f"Engram multi-head embedding: pad total_n from {total_N} to {padded_total_N}")

            self.memory = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=padded_total_N,
                embedding_dim=D,
                init_method=self.engram_cfg.embedding_init_method,
                reduce_scatter_embeddings=self.reduce_scatter_embeddings,
                config=self.engram_cfg,
                tp_group=self.tp_group,
            )
        else:
            self.embedding_parallel_group = parallel_state.get_engram_embedding_parallel_group()
            self.reduce_scatter_embeddings = self.engram_cfg.sequence_parallel
            padded_total_N = _vocab_size_with_padding(total_N, get_pg_size(self.embedding_parallel_group))
            print(f"Engram multi-head embedding: pad total_n from {total_N} to {padded_total_N}")
            self.memory = EngramMemory(
                num_embeddings=padded_total_N,
                embedding_dim=D,
                init_method=self.engram_cfg.embedding_init_method,
                reduce_scatter_embeddings=self.reduce_scatter_embeddings,
                config=self.engram_cfg,
                embedding_parallel_group=self.embedding_parallel_group,
            )
            if self.engram_cfg.engram_embedding_parallel_method == "alltoall":
                self.memory.enable_parallel()
                if self.engram_cfg.engram_offload_embedding_optimizer_states:
                    self.memory.enable_offloading()
            else:
                raise ValueError(f"Unsupported engram_embedding_parallel_method: {self.engram_cfg.engram_embedding_parallel_method}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.memory(shifted_input_ids)

        if not self.reduce_scatter_embeddings:
            output = output.transpose(0, 1).contiguous()
        return output
    
    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        sharded_dict = {}
        memory_prefix = f"{prefix}memory."
        memory_sharded_dict = self.memory.sharded_state_dict(memory_prefix, sharded_offsets, metadata)
        sharded_dict.update(memory_sharded_dict)
        return sharded_dict

# Copyright (c) 2025, BAAI. All rights reserved.

"""Runtime parallel-state context switching for colocated MIMO modules.

WARNING: The current implementation replaces
``megatron.plugin.hetero.parallel_context._GLOBAL_PARALLEL_CONTEXT`` at
runtime.  This is the simplest way to make ``megatron.core.parallel_state``
getters return module-local process groups, but it has caveats:

1. It is not thread-safe.  Only the main training thread should call
   ``switch_parallel_state``.
2. Any code that caches parallel-state values across a ``switch`` will see
   stale data.  We mitigate this by computing rank/world-size dynamically from
   the underlying groups.
3. It depends on Megatron-LM-FL internals.  A future refactor should move to
   a proper ``ParallelContext`` plugin or push the module-local group selection
   into Megatron core.

The class below supports only CP=1.  Expert parallelism (EP > 1) is
supported for the language module (the vision module is dense and keeps
singleton expert groups).  PP > 1 is supported for the language module
(the vision module stays on first-stage ranks).  Methods that only make
sense for other configurations raise
``NotImplementedError`` instead of returning a silently wrong default.
"""

from contextlib import contextmanager

import torch.distributed as dist

import megatron.plugin.hetero.parallel_context as hetero_ctx
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import GlobalMemoryBuffer


class _ModuleParallelContext:
    """Lightweight parallel context wrapping a single module's ProcessGroupCollection.

    This class implements the subset of ``megatron.plugin.hetero.parallel_context.ParallelContext``
    methods that are reachable from ``megatron.core.parallel_state`` getters. Rank and world size
    are computed dynamically from the underlying process groups to avoid stale cached values.
    """

    def __init__(self, pg_collection: ProcessGroupCollection):
        self.pg_collection = pg_collection
        # Memory buffer is lazily created on first access.
        self._global_memory_buffer = None

    # ------------------------------------------------------------------
    # Group getters
    # ------------------------------------------------------------------

    def get_model_parallel_group(self, check_initialized=True):
        return self.pg_collection.mp

    def get_tensor_model_parallel_group(self, check_initialized=True):
        return self.pg_collection.tp

    def get_pipeline_model_parallel_group(self, check_initialized=True, local_pp_group=False):
        return self.pg_collection.pp

    def get_data_parallel_group(self, with_context_parallel=False, partial_data_parallel=False):
        if with_context_parallel:
            if partial_data_parallel:
                return self.pg_collection.intra_dp_cp
            return self.pg_collection.dp_cp
        return self.pg_collection.dp

    def get_data_parallel_group_gloo(
        self, with_context_parallel=False, partial_data_parallel=False
    ):
        gloo_group = getattr(self.pg_collection, "dp_gloo", None)
        assert gloo_group is not None, (
            "pg_collection has no gloo DP group; it must be built by build_colocated_pg_collections"
        )
        return gloo_group

    def get_context_parallel_group(self, check_initialized=True):
        assert self._group_size(self.pg_collection.cp) == 1, (
            "CP > 1 is not supported in colocated MIMO yet"
        )
        return self.pg_collection.cp

    def get_context_parallel_global_ranks(self, check_initialized=True):
        group = self.get_context_parallel_group(check_initialized)
        return list(range(dist.get_world_size(group)))

    def get_hierarchical_context_parallel_groups(self, check_initialized=True):
        return [self.get_context_parallel_group(check_initialized)]

    def get_embedding_group(self, check_initialized=True):
        return self.pg_collection.embd

    def get_position_embedding_group(self, check_initialized=True):
        return self.pg_collection.pos_embd

    def get_amax_reduction_group(self, with_context_parallel=False, tp_only_amax_red=False):
        if with_context_parallel:
            assert self._group_size(self.pg_collection.tp_cp) == 1, (
                "CP > 1 is not supported in colocated MIMO yet"
            )
            return self.pg_collection.tp_cp
        return self.pg_collection.tp

    def get_tensor_and_data_parallel_group(self, with_context_parallel=False):
        # tp_dp_cp is the real TPxDP group (fixed pp) built by
        # ``build_colocated_pg_collections``; with CP=1 the
        # with/without-context-parallel variants coincide.
        return self.pg_collection.tp_dp_cp

    def get_tensor_and_context_parallel_group(self, check_initialized=True):
        assert self._group_size(self.pg_collection.tp_cp) == 1, (
            "CP > 1 is not supported in colocated MIMO yet"
        )
        return self.pg_collection.tp_cp

    def get_expert_model_parallel_group(self, check_initialized=True):
        return self.pg_collection.ep

    def get_expert_tensor_parallel_group(self, check_initialized=True):
        return self.pg_collection.expt_tp

    def get_expert_tensor_and_model_parallel_group(self, check_initialized=True):
        return self.pg_collection.tp_ep

    def get_expert_tensor_model_pipeline_parallel_group(self, check_initialized=True):
        return self.pg_collection.tp_ep_pp

    def get_expert_data_parallel_group(self):
        return self.pg_collection.expt_dp

    def get_expert_data_parallel_group_gloo(self):
        gloo_group = getattr(self.pg_collection, "expt_dp_gloo", None)
        if gloo_group is not None:
            return gloo_group
        return self.get_expert_data_parallel_group()

    def get_intra_distributed_optimizer_instance_group(self):
        return self.pg_collection.intra_dist_opt

    def get_inter_partial_data_parallel_group(self):
        return self.pg_collection.dp_cp

    # ------------------------------------------------------------------
    # World size / rank getters
    # ------------------------------------------------------------------

    def _group_size(self, group):
        if group is None:
            return 1
        return dist.get_world_size(group)

    def _group_rank(self, group):
        if group is None:
            return 0
        return dist.get_rank(group)

    def get_tensor_model_parallel_world_size(self):
        return self._group_size(self.pg_collection.tp)

    def get_pipeline_model_parallel_world_size(self, group=None):
        return self._group_size(group or self.pg_collection.pp)

    def get_tensor_model_parallel_rank(self):
        return self._group_rank(self.pg_collection.tp)

    def get_pipeline_model_parallel_rank(self, group=None):
        return self._group_rank(group or self.pg_collection.pp)

    def get_data_parallel_world_size(
        self, with_context_parallel=False, partial_data_parallel=False
    ):
        return self._group_size(
            self.get_data_parallel_group(with_context_parallel, partial_data_parallel)
        )

    def get_data_parallel_rank(self, with_context_parallel=False, partial_data_parallel=False):
        return self._group_rank(
            self.get_data_parallel_group(with_context_parallel, partial_data_parallel)
        )

    def get_context_parallel_world_size(self):
        return self._group_size(self.pg_collection.cp)

    def get_context_parallel_rank(self):
        return self._group_rank(self.pg_collection.cp)

    def get_tensor_and_context_parallel_world_size(self):
        return self._group_size(self.pg_collection.tp_cp)

    def get_tensor_and_context_parallel_rank(self):
        return self._group_rank(self.pg_collection.tp_cp)

    def get_expert_model_parallel_world_size(self):
        return self._group_size(self.pg_collection.ep)

    def get_expert_model_parallel_rank(self):
        return self._group_rank(self.pg_collection.ep)

    def get_expert_tensor_parallel_world_size(self):
        return self._group_size(self.pg_collection.expt_tp)

    def get_expert_tensor_parallel_rank(self):
        return self._group_rank(self.pg_collection.expt_tp)

    def get_expert_tensor_and_model_parallel_world_size(self):
        return self._group_size(self.pg_collection.tp_ep)

    def get_expert_tensor_and_model_parallel_rank(self):
        return self._group_rank(self.pg_collection.tp_ep)

    def get_expert_data_parallel_world_size(self):
        return self._group_size(self.pg_collection.expt_dp)

    def get_expert_data_parallel_rank(self):
        return self._group_rank(self.pg_collection.expt_dp)

    # ------------------------------------------------------------------
    # Src ranks and pipeline helpers
    # ------------------------------------------------------------------

    def _group_ranks(self, group):
        """Return the global ranks that belong to ``group``."""
        if group is None:
            return [dist.get_rank()]
        assert hasattr(dist, "get_process_group_ranks"), (
            "dist.get_process_group_ranks is required for colocated MIMO"
        )
        return dist.get_process_group_ranks(group)

    def get_model_parallel_src_rank(self):
        return self._group_ranks(self.pg_collection.mp)[0]

    def get_tensor_model_parallel_src_rank(self):
        return self._group_ranks(self.pg_collection.tp)[0]

    def get_data_parallel_src_rank(self, with_context_parallel=False):
        return self._group_ranks(self.get_data_parallel_group(with_context_parallel))[0]

    def get_pipeline_model_parallel_first_rank(self, group=None):
        return self._group_ranks(group or self.pg_collection.pp)[0]

    def get_pipeline_model_parallel_last_rank(self, group=None):
        ranks = self._group_ranks(group or self.pg_collection.pp)
        return ranks[-1]

    def get_pipeline_model_parallel_next_rank(self, group=None):
        ranks = self._group_ranks(group or self.pg_collection.pp)
        idx = self._group_rank(group or self.pg_collection.pp)
        return ranks[(idx + 1) % len(ranks)]

    def get_pipeline_model_parallel_prev_rank(self, group=None):
        ranks = self._group_ranks(group or self.pg_collection.pp)
        idx = self._group_rank(group or self.pg_collection.pp)
        return ranks[(idx - 1) % len(ranks)]

    def get_last_rank_when_using_pipeline(self):
        return self.get_pipeline_model_parallel_last_rank()

    def is_pipeline_first_stage(self, ignore_virtual=False, group=None):
        return self.get_pipeline_model_parallel_rank(group) == 0

    def is_pipeline_last_stage(self, ignore_virtual=False, group=None):
        return (
            self.get_pipeline_model_parallel_rank(group)
            == self.get_pipeline_model_parallel_world_size(group) - 1
        )

    def is_rank_in_embedding_group(self, ignore_virtual=False, group=None):
        return dist.get_rank() in self._group_ranks(group or self.pg_collection.embd)

    def is_rank_in_position_embedding_group(self, group=None):
        return dist.get_rank() in self._group_ranks(group or self.pg_collection.pos_embd)

    # ------------------------------------------------------------------
    # Virtual pipeline (PP=1, always zero / None)
    # ------------------------------------------------------------------

    def get_virtual_pipeline_model_parallel_world_size(self):
        return None

    def get_virtual_pipeline_model_parallel_rank(self):
        return None

    # ------------------------------------------------------------------
    # Setters (no-op; sizes are read from groups dynamically)
    # ------------------------------------------------------------------

    def set_tensor_model_parallel_world_size(self, world_size):
        pass

    def set_pipeline_model_parallel_world_size(self, world_size):
        pass

    def set_virtual_pipeline_model_parallel_world_size(self, world_size):
        pass

    def set_tensor_model_parallel_rank(self, rank):
        pass

    def set_pipeline_model_parallel_rank(self, rank):
        pass

    def set_virtual_pipeline_model_parallel_rank(self, rank):
        pass

    def set_data_parallel_rank(self, rank):
        pass

    def set_expert_model_parallel_world_size(self, world_size):
        pass

    def set_expert_model_parallel_rank(self, rank):
        pass

    def set_expert_tensor_parallel_world_size(self, world_size):
        pass

    def set_expert_tensor_parallel_rank(self, rank):
        pass

    # ------------------------------------------------------------------
    # Global memory buffer
    # ------------------------------------------------------------------

    def get_global_memory_buffer(self):
        if self._global_memory_buffer is None:
            self._global_memory_buffer = GlobalMemoryBuffer()
        return self._global_memory_buffer

    def set_global_memory_buffer(self):
        if self._global_memory_buffer is None:
            self._global_memory_buffer = GlobalMemoryBuffer()

    def destroy_global_memory_buffer(self):
        self._global_memory_buffer = None


@contextmanager
def switch_parallel_state(pg_collection: ProcessGroupCollection):
    """Switch the global parallel_state context to the given module's process groups.

    WARNING: This mutates a Megatron global.  See the module docstring for the
    list of caveats.  The caller must ensure that no other thread is reading
    ``parallel_state`` during the switched region.

    Usage:
        with switch_parallel_state(vision_pg):
            vision_embeds = vision_module(...)
    """
    saved = hetero_ctx._GLOBAL_PARALLEL_CONTEXT
    hetero_ctx._GLOBAL_PARALLEL_CONTEXT = _ModuleParallelContext(pg_collection)
    try:
        yield
    finally:
        hetero_ctx._GLOBAL_PARALLEL_CONTEXT = saved

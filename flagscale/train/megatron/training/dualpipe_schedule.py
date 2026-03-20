# Copyright (c) 2024, BAAI. All rights reserved.
"""DualPipe pipeline parallel schedule for FlagScale.

Implements the bidirectional DualPipe algorithm introduced in the
DeepSeek-V3 Technical Report (https://arxiv.org/abs/2412.19437).
Reference implementation: https://github.com/deepseek-ai/DualPipe

DualPipe achieves full overlap of forward and backward computation-communication
phases in both directions, reducing pipeline bubble size compared to 1F1B.

Architecture
------------
- N physical pipeline-parallel ranks (N must be even).
- Each rank holds TWO model chunks:
    * model[0] – forward-direction chunk (pipeline position = rank_id)
    * model[1] – mirror chunk (pipeline position = N-1 - rank_id)
- Micro-batches are split into two halves and flow simultaneously in both
  directions through the pipeline.
- The schedule has 8 distinct phases that overlap forward and backward
  computation from the two directions, minimising idle ("bubble") time.

Usage
-----
Enable with ``--use-dualpipe`` in your training configuration.  The schedule
is automatically selected by FlagScale's training loop when that flag is set.

Constraints
-----------
- ``pipeline_model_parallel_size`` must be even and > 1.
- ``num_microbatches`` must be even and >= ``pipeline_model_parallel_size * 2``.
- Two model chunks per rank are required (model is a list of length 2).
- Two data iterators per rank are required (data_iterator is a list of length 2),
  one for each direction of data flow.
"""

import contextlib
import queue
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model, get_model_config


# ---------------------------------------------------------------------------
# WeightGradStore
# ---------------------------------------------------------------------------

class WeightGradStore:
    """Defers weight-gradient computations to reduce pipeline bubble size.

    When ``enabled`` is True, backward passes compute only input gradients
    (not weight gradients).  Weight-gradient functions are accumulated here
    and executed later during the dedicated "W" steps of the DualPipe
    schedule, shortening the critical synchronisation path.

    This class is a FlagScale re-implementation of the same-named class from
    https://github.com/deepseek-ai/DualPipe/blob/main/dualpipe/utils.py.
    """

    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue: queue.Queue = queue.Queue()

    # Note: WeightGradStore uses class-level mutable state (not thread-safe).
    # This matches the DualPipe reference implementation which assumes
    # single-threaded execution within each training process.  Each rank
    # runs its own copy of the class independently.

    @classmethod
    def put(cls, func: Callable) -> None:
        """Queue a weight-gradient computation function."""
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        """Move currently cached functions into the persistent FIFO queue."""
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        """Execute all functions from the front of the queue (FIFO order)."""
        assert not cls.funcs_queue.empty(), "Weight gradient queue is empty."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        """Reset all state (call between training steps)."""
        cls.cache = []
        cls.funcs_queue = queue.Queue()


# ---------------------------------------------------------------------------
# P2P communication helpers
# ---------------------------------------------------------------------------

def _allocate_tensor(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate a CUDA tensor suitable for p2p receive."""
    return torch.empty(shape, dtype=dtype, device=torch.cuda.current_device(),
                       requires_grad=True)


def _commit_and_wait(ops: List[dist.P2POp]) -> None:
    """Execute a batch of P2P ops and wait for completion."""
    if not ops:
        return
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()


def _rank_in_group(group: dist.ProcessGroup, rank: int) -> int:
    """Convert a group-local rank to a global rank."""
    return dist.get_global_rank(group, rank)


def _append_recv(
    ops: List[dist.P2POp],
    src_local_rank: int,
    group: dist.ProcessGroup,
    tensor_shape: Tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a receive buffer and append an irecv op; return the buffer."""
    buf = _allocate_tensor(tensor_shape, dtype)
    src_global = _rank_in_group(group, src_local_rank)
    ops.append(dist.P2POp(dist.irecv, buf, src_global, group))
    return buf


def _append_send(
    ops: List[dist.P2POp],
    tensor: torch.Tensor,
    dst_local_rank: int,
    group: dist.ProcessGroup,
) -> None:
    """Append an isend op for *tensor*."""
    dst_global = _rank_in_group(group, dst_local_rank)
    ops.append(dist.P2POp(dist.isend, tensor, dst_global, group))


# ---------------------------------------------------------------------------
# Megatron-compatible forward / backward helpers
# ---------------------------------------------------------------------------

def _set_input(model_chunk: torch.nn.Module, input_tensor: Optional[torch.Tensor]) -> None:
    """Inject *input_tensor* into *model_chunk* via Megatron's set_input_tensor."""
    if input_tensor is None:
        return
    setter = get_attr_wrapped_model(model_chunk, "set_input_tensor")
    setter([input_tensor])


def _run_forward(
    forward_step_func: Callable,
    data_iter: Optional[Iterator],
    model_chunk: torch.nn.Module,
    input_tensor: Optional[torch.Tensor],
    config,
    num_microbatches: int,
    forward_data_store: List,
    collect_non_loss_data: bool,
    is_last_stage: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one micro-batch forward pass using Megatron's schedule helpers.

    Returns
    -------
    output_tensor : torch.Tensor
        Hidden-state output (or scalar loss on the last stage).
    num_tokens : torch.Tensor
        Token count tensor (for per-token loss averaging).
    """
    # Import here to avoid circular dependencies at module load time.
    from megatron.core.pipeline_parallel.schedules import (
        forward_step as megatron_forward_step,
    )
    from megatron.core.parallel_state import get_context_parallel_world_size

    _set_input(model_chunk, input_tensor)

    output_tensor, num_tokens = megatron_forward_step(
        forward_step_func=forward_step_func,
        data_iterator=data_iter,
        model=model_chunk,
        num_microbatches=num_microbatches,
        input_tensor=input_tensor,
        forward_data_store=forward_data_store,
        config=config,
        cp_group_size=get_context_parallel_world_size(),
        collect_non_loss_data=collect_non_loss_data,
        is_last_stage=is_last_stage,
    )
    return output_tensor, num_tokens


def _run_backward(
    input_tensor: Optional[torch.Tensor],
    output_tensor: torch.Tensor,
    output_grad: Optional[torch.Tensor],
    config,
    enable_zb: bool = False,
) -> Optional[torch.Tensor]:
    """Run one micro-batch backward pass using Megatron's backward_step helper.

    When *enable_zb* is True, weight gradients are deferred into
    :class:`WeightGradStore` (zero-bubble mode).

    Returns the gradient w.r.t. *input_tensor* (None for first stage).
    """
    from megatron.core.pipeline_parallel.schedules import backward_step as megatron_backward_step

    WeightGradStore.enabled = enable_zb
    input_grad = megatron_backward_step(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        output_tensor_grad=output_grad,
        config=config,
    )
    WeightGradStore.enabled = False
    if enable_zb:
        WeightGradStore.flush()
    return input_grad


# ---------------------------------------------------------------------------
# DualPipe schedule state
# ---------------------------------------------------------------------------

class _DualPipeState:
    """Mutable state threaded through the DualPipe schedule."""

    def __init__(self) -> None:
        # input_chunks[phase] = list of tensors received from upstream stage
        self.input_chunks: List[List[Optional[torch.Tensor]]] = [[], []]
        # output_chunks[phase] = list of output tensors produced by forward
        self.output_chunks: List[List[Optional[torch.Tensor]]] = [[], []]
        # input_grad_chunks[phase] = list of input-tensor grads (for sending backward)
        self.input_grad_chunks: List[List[Optional[torch.Tensor]]] = [[], []]
        # output_grad_chunks[phase] = list of grads received from downstream stage
        self.output_grad_chunks: List[List[Optional[torch.Tensor]]] = [[], []]

        # Counters: how many chunks of each type have been processed per phase
        self.f_cnt: List[int] = [0, 0]    # forward compute
        self.b_cnt: List[int] = [0, 0]    # backward compute
        self.sf_cnt: List[int] = [0, 0]   # send forward
        self.sb_cnt: List[int] = [0, 0]   # send backward
        self.rf_cnt: List[int] = [0, 0]   # recv forward
        self.rb_cnt: List[int] = [0, 0]   # recv backward

        # Pending communication ops and tensors to free after comm
        self.pending_ops: List[dist.P2POp] = []
        self.to_free: List[torch.Tensor] = []

        # Loss and forward data accumulators
        self.forward_data_store: List = []

    def reset(self) -> None:
        """Clear all state (call at the start of each step)."""
        self.input_chunks = [[], []]
        self.output_chunks = [[], []]
        self.input_grad_chunks = [[], []]
        self.output_grad_chunks = [[], []]
        self.f_cnt = [0, 0]
        self.b_cnt = [0, 0]
        self.sf_cnt = [0, 0]
        self.sb_cnt = [0, 0]
        self.rf_cnt = [0, 0]
        self.rb_cnt = [0, 0]
        self.pending_ops = []
        self.to_free = []
        self.forward_data_store = []
        WeightGradStore.clear()

    def commit_and_wait(self) -> None:
        """Execute all pending p2p ops and free marked tensors."""
        _commit_and_wait(self.pending_ops)
        self.pending_ops = []
        for t in self.to_free:
            t.data = torch.empty(0, device=t.device, dtype=t.dtype)
        self.to_free = []


# ---------------------------------------------------------------------------
# Data iterator utilities
# ---------------------------------------------------------------------------

class _SliceIterator:
    """A simple iterator over a pre-buffered list of micro-batches."""

    def __init__(self, items: list) -> None:
        self._items = items
        self._idx = 0

    def __iter__(self) -> "_SliceIterator":
        return self

    def __next__(self):
        if self._idx >= len(self._items):
            raise StopIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _split_data_iterator(
    data_iterator, num_microbatches: int
) -> List["_SliceIterator"]:
    """Pre-buffer *num_microbatches* items and split into two halves.

    This is used when a single data iterator is passed to
    :func:`forward_backward_dualpipe` instead of a pre-split list.

    Returns
    -------
    list of two :class:`_SliceIterator` objects:
        ``[iter_phase0, iter_phase1]``
    """
    half = num_microbatches // 2
    # Eagerly consume items so all pipeline ranks advance their iterator by the
    # same amount (as they would in a standard 1F1B schedule).
    if hasattr(data_iterator, '__next__') or hasattr(data_iterator, '__iter__') and not isinstance(data_iterator, (list, tuple)):
        items = [next(data_iterator) for _ in range(num_microbatches)]
    elif isinstance(data_iterator, (list, tuple)):
        # If it's already a list/tuple of iterators, use the first one.
        if not hasattr(data_iterator[0], '__next__'):
            raise TypeError(
                f"DualPipe: data_iterator is a list/tuple but its first element "
                f"({type(data_iterator[0])}) is not an iterator. "
                f"Pass either a single iterator or a list of two iterators."
            )
        items = [next(data_iterator[0]) for _ in range(num_microbatches)]
    else:
        raise TypeError(
            f"DualPipe: unsupported data_iterator type {type(data_iterator)}. "
            f"Expected a single iterator or a list of two iterators."
        )
    return [_SliceIterator(items[:half]), _SliceIterator(items[half:])]


# ---------------------------------------------------------------------------
# Main schedule function
# ---------------------------------------------------------------------------

def forward_backward_dualpipe(
    *,
    forward_step_func: Callable,
    data_iterator: List[Optional[Iterator]],
    model: List[torch.nn.Module],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    **kwargs,
) -> List[Dict]:
    """Bidirectional DualPipe pipeline-parallel forward and backward pass.

    This function has the same external interface as Megatron-Core's
    ``forward_backward_pipelining_without_interleaving`` so it can be used
    as a drop-in replacement when ``--use-dualpipe`` is enabled.

    Parameters
    ----------
    forward_step_func:
        User-provided forward step (same signature as Megatron schedules).
    data_iterator:
        A list of two data iterators: ``[iter_phase0, iter_phase1]``.
        ``iter_phase0`` feeds the forward-direction half of micro-batches;
        ``iter_phase1`` feeds the reverse-direction half.
    model:
        A list of two model chunks: ``[chunk_fwd, chunk_mirror]``.
    num_microbatches:
        Total micro-batches for this step.  Must be even and
        >= ``pipeline_model_parallel_size * 2``.
    seq_length:
        Sequence length (used to derive p2p tensor shape).
    micro_batch_size:
        Micro-batch size (used to derive p2p tensor shape).
    forward_only:
        When True only forward passes are executed (e.g. during evaluation).
    collect_non_loss_data:
        Forward non-loss outputs (e.g. for inference).

    Returns
    -------
    list of dicts
        Loss-reduced dictionaries, one per micro-batch on the last pipeline
        stage; empty on other stages (same convention as Megatron schedules).
    """
    # ------------------------------------------------------------------ setup
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    num_ranks = parallel_state.get_pipeline_model_parallel_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    assert num_ranks % 2 == 0, (
        f"DualPipe requires an even number of pipeline ranks, got {num_ranks}."
    )
    assert num_microbatches % 2 == 0, (
        f"DualPipe requires an even number of micro-batches, got {num_microbatches}."
    )
    assert num_microbatches >= num_ranks * 2, (
        f"DualPipe requires num_microbatches ({num_microbatches}) >= "
        f"num_ranks*2 ({num_ranks * 2})."
    )
    assert len(model) == 2, "DualPipe requires exactly 2 model chunks per rank."
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 2, (
            "DualPipe requires data_iterator to be a list of 2 iterators "
            "or a single iterator (which will be split automatically)."
        )
    else:
        # Auto-split the single iterator into two halves.
        data_iterator = _split_data_iterator(data_iterator, num_microbatches)

    # Rank topology
    is_first_rank = pp_rank == 0
    is_last_rank = pp_rank == num_ranks - 1
    is_in_second_half = pp_rank >= num_ranks // 2
    is_middle_rank = pp_rank in (num_ranks // 2 - 1, num_ranks // 2)

    prev_rank = pp_rank - 1  # group-local; -1 means no previous
    next_rank = pp_rank + 1  # group-local; num_ranks means no next

    half_rank = min(pp_rank, num_ranks - 1 - pp_rank)
    num_half_ranks = num_ranks // 2
    half_num_chunks = num_microbatches // 2

    # The "effective phase" for communication flips for the second half of ranks.
    # phase XOR is_in_second_half gives the canonical direction:
    #   0 → left-to-right  (data flows from rank 0 towards rank N-1)
    #   1 → right-to-left  (data flows from rank N-1 towards rank 0)

    config = get_model_config(model[0])

    # Determine tensor shape for p2p communication.
    # Megatron stores activations as (seq_len, micro_batch, hidden_size).
    # If sequence parallelism is on, seq_len is divided by TP world size.
    hidden_size = config.hidden_size
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    effective_seq_len = seq_length // tp_size if config.sequence_parallel else seq_length
    tensor_shape = (effective_seq_len, micro_batch_size, hidden_size)
    tensor_dtype = config.params_dtype

    # Which pipeline stages are "last" for each model chunk?
    # chunk[0] is at pipeline pos pp_rank   → last stage when pp_rank == num_ranks-1
    # chunk[1] is at pipeline pos N-1-pp_rank → last stage when pp_rank == 0
    chunk_is_last: List[bool] = [is_last_rank, is_first_rank]
    # chunk[0] is "first" stage when pp_rank == 0; chunk[1] when pp_rank == num_ranks-1
    chunk_is_first: List[bool] = [is_first_rank, is_last_rank]

    # Data iterators
    data_iter_phase = [data_iterator[0], data_iterator[1]]

    # ------------------------------------------------------------------ state
    state = _DualPipeState()

    # -------------------------------------------------------- inner helpers
    def _actual_phase(phase: int) -> int:
        """Translate schedule phase → canonical direction (XOR second-half)."""
        return phase ^ int(is_in_second_half)

    def _recv_fwd(phase: int) -> None:
        """Queue an irecv for an upstream forward tensor for *phase*."""
        canon = _actual_phase(phase)
        is_first = chunk_is_first[phase]
        if is_first:
            return  # No upstream stage; first stage creates its own input.
        state.rf_cnt[phase] += 1
        src = prev_rank if canon == 0 else next_rank
        buf = _append_recv(state.pending_ops, src, pp_group, tensor_shape, tensor_dtype)
        state.input_chunks[phase].append(buf)

    def _send_fwd(phase: int) -> None:
        """Queue an isend for the just-computed forward output for *phase*."""
        canon = _actual_phase(phase)
        is_last = chunk_is_last[phase]
        if is_last:
            return  # No downstream stage.
        idx = state.sf_cnt[phase]
        state.sf_cnt[phase] += 1
        tensor = state.output_chunks[phase][idx]
        dst = next_rank if canon == 0 else prev_rank
        _append_send(state.pending_ops, tensor, dst, pp_group)
        state.to_free.append(tensor)

    def _recv_bwd(phase: int) -> None:
        """Queue an irecv for a downstream backward gradient for *phase*."""
        if forward_only:
            return
        canon = _actual_phase(phase)
        is_last = chunk_is_last[phase]
        if is_last:
            return  # Last stage owns the loss; no gradient comes from downstream.
        state.rb_cnt[phase] += 1
        src = next_rank if canon == 0 else prev_rank
        buf = _append_recv(state.pending_ops, src, pp_group, tensor_shape, tensor_dtype)
        state.output_grad_chunks[phase].append(buf)

    def _send_bwd(phase: int) -> None:
        """Queue an isend for the computed input-gradient for *phase*."""
        if forward_only:
            return
        canon = _actual_phase(phase)
        is_first = chunk_is_first[phase]
        if is_first:
            return  # No upstream stage to send gradient to.
        idx = state.sb_cnt[phase]
        state.sb_cnt[phase] += 1
        grads = state.input_grad_chunks[phase][idx]
        dst = prev_rank if canon == 0 else next_rank
        for g in (grads if isinstance(grads, (list, tuple)) else [grads]):
            if g is not None:
                _append_send(state.pending_ops, g, dst, pp_group)

    def _fwd_compute(phase: int) -> None:
        """Compute one forward micro-batch for *phase*."""
        chunk_id = state.f_cnt[phase]
        state.f_cnt[phase] += 1

        input_tensor = (
            state.input_chunks[phase][chunk_id]
            if chunk_id < len(state.input_chunks[phase])
            else None
        )
        if forward_only and input_tensor is not None:
            # Clear stored input to save memory during inference.
            state.input_chunks[phase][chunk_id] = None

        model_chunk = model[phase]
        is_last = chunk_is_last[phase]

        output_tensor, _ = _run_forward(
            forward_step_func=forward_step_func,
            data_iter=data_iter_phase[phase],
            model_chunk=model_chunk,
            input_tensor=input_tensor,
            config=config,
            num_microbatches=num_microbatches,
            forward_data_store=state.forward_data_store,
            collect_non_loss_data=collect_non_loss_data,
            is_last_stage=is_last,
        )

        if not is_last:
            state.output_chunks[phase].append(output_tensor)
        else:
            # Store the loss tensor so backward can call .backward() on it.
            state.output_chunks[phase].append(output_tensor)

    def _bwd_compute(phase: int, enable_zb: bool = False) -> None:
        """Compute one backward micro-batch for *phase*."""
        if forward_only:
            return

        chunk_id = state.b_cnt[phase]
        state.b_cnt[phase] += 1

        is_last = chunk_is_last[phase]
        output_tensor = state.output_chunks[phase][chunk_id]
        if not is_last:
            state.output_chunks[phase][chunk_id] = None  # free after bwd

        input_tensor = (
            state.input_chunks[phase][chunk_id]
            if chunk_id < len(state.input_chunks[phase])
            else None
        )

        if is_last:
            output_grad = None  # Scalar loss; PyTorch fills grad_fn automatically.
        else:
            output_grad = state.output_grad_chunks[phase][chunk_id]
            state.output_grad_chunks[phase][chunk_id] = None

        input_grad = _run_backward(
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            output_grad=output_grad,
            config=config,
            enable_zb=enable_zb,
        )
        state.input_chunks[phase][chunk_id] = None
        state.input_grad_chunks[phase].append(input_grad)

    def _fwd_bwd_compute(phase0: int, phase1: int) -> None:
        """Overlapped forward (phase0) + backward (phase1)."""
        if forward_only:
            _fwd_compute(phase0)
            return
        # Non-overlapped fallback: run sequentially.
        # (A real implementation can add custom CUDA-stream overlap here.)
        _fwd_compute(phase0)
        _bwd_compute(phase1)

    def _weight_step() -> None:
        """Execute one deferred weight-gradient computation chunk."""
        if forward_only:
            return
        state.commit_and_wait()
        WeightGradStore.pop()

    # ------------------------------------------------------------------ schedule

    # Step 1: nF0 – warmup forward passes in direction 0
    step_1 = (num_half_ranks - half_rank - 1) * 2
    for _ in range(step_1):
        _recv_fwd(0)
        state.commit_and_wait()
        _fwd_compute(0)
        _send_fwd(0)

    # Step 2: nF0F1 – start feeding direction 1 alongside direction 0
    step_2 = half_rank + 1
    _recv_fwd(0)
    for i in range(step_2):
        state.commit_and_wait()
        _fwd_compute(0)
        if not is_middle_rank:
            _send_fwd(0)
        _recv_fwd(0)
        state.commit_and_wait()
        _fwd_compute(1)
        send_fwd1 = (not is_middle_rank) or (i < step_2 - 1)
        if send_fwd1:
            _send_fwd(1)
        if is_middle_rank:
            _send_fwd(0)

    # Step 3: nB1W1F1 – backward direction-1 with zero-bubble
    step_3 = num_half_ranks - half_rank - 1
    for _ in range(step_3):
        _recv_bwd(1)
        state.commit_and_wait()
        _bwd_compute(1, enable_zb=True)
        _send_bwd(1)
        _recv_fwd(1)
        _weight_step()
        state.commit_and_wait()
        _fwd_compute(1)
        _send_fwd(1)

    # Step 4: Main step – F0B1 / F1B0 overlapped pairs
    step_4 = half_num_chunks - num_ranks + half_rank + 1
    for i in range(step_4):
        if i == 0:
            if is_middle_rank:
                # Middle rank: avoid additional overlap to minimise bubble.
                state.commit_and_wait()
                _fwd_compute(0)
                _send_fwd(1)
                _recv_bwd(1)
                _recv_fwd(0)  # pre-recv for next F0
                state.commit_and_wait()
                _bwd_compute(1)
                _send_fwd(0)
                _send_bwd(1)
            else:
                _recv_bwd(1)
                state.commit_and_wait()
                _fwd_bwd_compute(0, 1)
                _send_fwd(0)
                _send_bwd(1)
        else:
            _recv_fwd(0)
            _recv_bwd(1)
            state.commit_and_wait()
            _fwd_bwd_compute(0, 1)
            _send_fwd(0)
            _send_bwd(1)

        _recv_fwd(1)
        _recv_bwd(0)
        state.commit_and_wait()
        _fwd_bwd_compute(1, 0)
        _send_fwd(1)
        _send_bwd(0)

    # Step 5: nB1F1B0 – finish direction-1 forward while doing direction-0 backward
    step_5 = num_half_ranks - half_rank - 1
    for _ in range(step_5):
        _recv_bwd(1)
        state.commit_and_wait()
        _bwd_compute(1)
        _send_bwd(1)

        _recv_fwd(1)
        _recv_bwd(0)
        state.commit_and_wait()
        _fwd_bwd_compute(1, 0)
        _send_fwd(1)
        _send_bwd(0)

    # Step 6: nB1B0 – pure backward (second half uses zero-bubble)
    step_6 = half_rank + 1
    enable_zb = False
    for i in range(step_6):
        if i == step_6 // 2 and half_rank % 2 == 1:
            enable_zb = True
        _recv_bwd(1)
        state.commit_and_wait()
        _bwd_compute(1, enable_zb=enable_zb)
        _send_bwd(1)

        if i == step_6 // 2 and half_rank % 2 == 0:
            enable_zb = True
        _recv_bwd(0)
        state.commit_and_wait()
        _bwd_compute(0, enable_zb=enable_zb)
        _send_bwd(0)

    # Step 7: nWB0 – weight-grad flush + direction-0 backward with zero-bubble
    step_7 = num_half_ranks - half_rank - 1
    for _ in range(step_7):
        _weight_step()
        _recv_bwd(0)
        state.commit_and_wait()
        _bwd_compute(0, enable_zb=True)
        _send_bwd(0)

    # Step 8: nW – flush remaining weight gradients
    step_8 = half_rank + 1
    for _ in range(step_8):
        _weight_step()

    assert WeightGradStore.funcs_queue.empty(), (
        "Weight gradient queue non-empty after DualPipe schedule; "
        "this indicates a scheduling bug."
    )

    state.commit_and_wait()
    state.reset()

    return state.forward_data_store


# ---------------------------------------------------------------------------
# Helper: get DualPipe forward-backward function
# ---------------------------------------------------------------------------

def get_dualpipe_forward_backward_func() -> Callable:
    """Return the DualPipe forward-backward schedule function.

    This is called instead of ``megatron.core.pipeline_parallel.get_forward_backward_func``
    when ``--use-dualpipe`` is enabled.
    """
    return forward_backward_dualpipe

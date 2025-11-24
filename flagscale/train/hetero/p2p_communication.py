# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
)
from megatron.core.model_parallel_config import ModelParallelConfig
# from megatron.core.pipeline_parallel.p2p_communication import is_single_shape

# Types
Shape = Union[List[int], torch.Size]

from flagscale.train import get_parallel_context
from flagscale.train.hetero.parallel_context import ParallelContext

def is_single_shape(x) -> bool:
    """Check if the input is a single shape."""
    if isinstance(x, torch.Size):
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(d, int) for d in x):
        return True
    return False

def get_device_type_for_comm(model_parallel_group=None):
    device = 'cuda'
    # "cpu:gloo": gloo only supports cpu tensor.
    # "gloo" & "cpu:gloo,cuda:gloo": gloo supports both cpu and cuda tensor.
    if isinstance(model_parallel_group, list):
        if 'cpu:gloo' == torch.distributed.get_backend(model_parallel_group[0]):
            device = 'cpu'
    else:
        if 'cpu:gloo' == torch.distributed.get_backend(model_parallel_group):
            device = 'cpu'
    return device

def is_inter_mesh_comm(para_ctx: ParallelContext, comm_with_front_layer: bool):
    """ Judge if the p2p communication across meshes.

    comm_with_front_layer: if this communication is established with front layer,
        including send_backward and recv_forward
    """

    assert para_ctx is not None, "Specify ParallelContext Necessary"
    assert comm_with_front_layer is not None, "Specify Communication Direction Necessary"
    if comm_with_front_layer:
        # To judge if current rank is in the first stage of current mesh.
        # In this condition, its pp rank should equal to
        # the sum of pp size of all previous meshes
        total_prev_pipeline_model_parallel_size = 0
        for i in range(0, para_ctx._current_process_mesh_index):
            total_prev_pipeline_model_parallel_size += para_ctx._process_meshes[i]._rank_generator.pp
        return get_pipeline_model_parallel_rank() == total_prev_pipeline_model_parallel_size
    else:
        # To judge if current rank is in the last stage of current mesh.
        # In this condition, its pp rank should equal to
        # the (sum of pp size of all previous and current meshes) - 1
        total_current_pipeline_model_parallel_size = 0
        for i in range(0, min(para_ctx._current_process_mesh_index + 1, len(para_ctx._process_meshes))):
            total_current_pipeline_model_parallel_size += para_ctx._process_meshes[i]._rank_generator.pp
        return get_pipeline_model_parallel_rank() == total_current_pipeline_model_parallel_size - 1


def recv_forward_hetero(tensor_shapes, is_first_stage: bool, config: ModelParallelConfig, _communicate: callable = None,
) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """

    unwrap_tensor_shapes = False
    if is_single_shape(tensor_shapes):
        unwrap_tensor_shapes = True
        tensor_shapes = [tensor_shapes]
    input_tensors = []

    for tensor_shape in tensor_shapes:
        if is_first_stage:
            input_tensor = None
        else:
            if config.timers is not None:
                config.timers('forward-recv', log_level=2).start()
            rank = torch.distributed.get_rank()
            para_ctx = get_parallel_context()
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True):
                group = para_ctx.get_pipeline_model_parallel_group(local_pp_group=True)
                input_tensor, _, _ = _communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                    group=group,
                )
            else:
                tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                    rank=rank, local_tensor_shape=tensor_shape, next=False
                )
                input_tensor = torch.empty(tensor_shape,
                                        device=torch.cuda.current_device() if "cpu:gloo" != pp_groups[0].name() else torch.device("cpu"),
                                        dtype=config.pipeline_dtype,
                                        requires_grad=True)
                if tensor_slices is not None:
                    for tensor_slice in tensor_slices:
                        dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                        tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                        group = None
                        pp_groups = para_ctx.get_pipeline_model_parallel_group()
                        for pp_group in pp_groups:
                            pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                            if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                                group = pp_group
                                break
                        input_tensor_sliced, _, _ = _communicate(
                            tensor_send_next=None,
                            tensor_send_prev=None,
                            recv_prev=True,
                            recv_next=False,
                            tensor_shape=tensor_shape_sliced,
                            group=group,
                        )
                        input_tensor.data[sp_start:sp_end, dp_start:dp_end, :] = input_tensor_sliced
            if config.timers is not None:
                config.timers('forward-recv').stop()
        if input_tensor is not None and input_tensor.device == torch.device("cpu"):
            input_tensor = input_tensor.to(torch.cuda.current_device())
        input_tensors.append(input_tensor)
    if unwrap_tensor_shapes:
        return input_tensors[0]
    return input_tensors


def recv_backward_hetero(
    tensor_shapes: Shape, is_last_stage: bool, config: ModelParallelConfig, _communicate: callable = None,
) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """

    unwrap_tensor_shapes = False
    if is_single_shape(tensor_shapes):
        unwrap_tensor_shapes = True
        tensor_shapes = [tensor_shapes]
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if is_last_stage:
            output_tensor_grad = None
        else:
            if config.timers is not None:
                config.timers('backward-recv', log_level=2).start()
            rank = torch.distributed.get_rank()
            para_ctx = get_parallel_context()
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False):
                group = para_ctx.get_pipeline_model_parallel_group(local_pp_group=True)
                _, output_tensor_grad, _ = _communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                    group=group,
                )
            else:
                tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                    rank=rank, local_tensor_shape=tensor_shape, next=True
                )
                output_tensor_grad = torch.empty(tensor_shape,
                                                device=torch.cuda.current_device() if "cpu:gloo" != pp_groups[0].name() else torch.device("cpu"),
                                                dtype=config.pipeline_dtype,
                                                requires_grad=True)
                if tensor_slices is not None:
                    for tensor_slice in tensor_slices:
                        dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                        tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                        group = None
                        for pp_group in pp_groups:
                            pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                            if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                                group = pp_group
                                break
                        _, output_tensor_grad_sliced, _ = _communicate(
                            tensor_send_next=None,
                            tensor_send_prev=None,
                            recv_prev=False,
                            recv_next=True,
                            tensor_shape=tensor_shape_sliced,
                            group=group,
                        )
                        # tensor_shape is current tensor shape
                        dp_coef = para_ctx.get_dp_coef_when_recv_backward()
                        if dp_coef != 1.0:
                            output_tensor_grad.data[sp_start:sp_end, dp_start:dp_end, :] = output_tensor_grad_sliced * dp_coef
                        else:
                            output_tensor_grad.data[sp_start:sp_end, dp_start:dp_end, :] = output_tensor_grad_sliced
            if config.timers is not None:
                config.timers('backward-recv').stop()

        if output_tensor_grad is not None and output_tensor_grad.device == torch.device("cpu"):
            output_tensor_grad = output_tensor_grad.to(torch.cuda.current_device())
        output_tensor_grads.append(output_tensor_grad)

    if unwrap_tensor_shapes:
            return output_tensor_grads[0]
    return output_tensor_grads


def send_forward_hetero(
    output_tensors: torch.Tensor, is_last_stage: bool, config: ModelParallelConfig, _communicate: callable = None,
) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]

    for output_tensor in output_tensors:
        if not is_last_stage:
            if config.timers is not None:
                config.timers('forward-send', log_level=2).start()
            rank = torch.distributed.get_rank()
            para_ctx = get_parallel_context()
            if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False):
                group = para_ctx.get_pipeline_model_parallel_group(local_pp_group=True)
                _communicate(
                    tensor_send_next=output_tensor if "cpu:gloo" != group.name() else output_tensor.cpu(),
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                    group=group,
                )
            else:
                tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                    rank=rank, local_tensor_shape=output_tensor.shape, next=True
                )
                if tensor_slices is not None:
                    for tensor_slice in tensor_slices:
                        dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                        output_tensor_sliced = output_tensor[sp_start:sp_end, dp_start:dp_end, :]
                        group = None
                        pp_groups = para_ctx.get_pipeline_model_parallel_group()
                        for pp_group in pp_groups:
                            pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                            if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                                group = pp_group
                                break
                        _communicate(
                            tensor_send_next=output_tensor_sliced.contiguous() if "cpu:gloo" != group.name() else output_tensor_sliced.cpu(),
                            tensor_send_prev=None,
                            recv_prev=False,
                            recv_next=False,
                            tensor_shape=None,
                            group=group,
                        )
            if config.timers is not None:
                config.timers('forward-send').stop()


def send_backward_hetero(
    input_tensor_grads: torch.Tensor, is_first_stage: bool, config: ModelParallelConfig, _communicate: callable = None,
) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for input_tensor_grad in input_tensor_grads:
        if not is_first_stage:
            if config.timers is not None:
                config.timers('backward-send', log_level=2).start()
            rank = torch.distributed.get_rank()
            para_ctx = get_parallel_context()
            if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True):
                group = para_ctx.get_pipeline_model_parallel_group(local_pp_group=True)
                _communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad if "cpu:gloo" != group.name() else input_tensor_grad.cpu(),
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                    group=group,
                )
            else:
                tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                    rank=rank, local_tensor_shape=input_tensor_grad.shape, next=False
                )
                if tensor_slices is not None:
                    for tensor_slice in tensor_slices:
                        dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                        input_tensor_grad_sliced = input_tensor_grad[sp_start:sp_end, dp_start:dp_end, :]
                        group = None
                        pp_groups = para_ctx.get_pipeline_model_parallel_group()
                        for pp_group in pp_groups:
                            pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                            if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                                group = pp_group
                                break
                        _communicate(
                            tensor_send_next=None,
                            tensor_send_prev=input_tensor_grad_sliced.contiguous() if "cpu:gloo" != group.name() else input_tensor_grad_sliced.cpu(),
                            recv_prev=False,
                            recv_next=False,
                            tensor_shape=None,
                            group=group,
                        )
            if config.timers is not None:
                config.timers('backward-send').stop()


def send_forward_recv_backward_hetero(
    output_tensors: torch.Tensor,
    tensor_shapes: Shape,
    is_last_stage: bool,
    config: ModelParallelConfig,
    _communicate: callable = None,
) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    unwrap_output_tensors = False
    if not isinstance(output_tensors, list):
        unwrap_output_tensors = True
        output_tensors = [output_tensors]
    if not isinstance(tensor_shapes, list):
        tensor_shapes = [tensor_shapes]
    output_tensor_grads = []

    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if is_last_stage:
            output_tensor_grad = None
        else:
            if config.timers is not None:
                config.timers('forward-send-backward-recv', log_level=2).start()
            rank = torch.distributed.get_rank()
            para_ctx = get_parallel_context()
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False):
                group = para_ctx.get_pipeline_model_parallel_group(local_pp_group=True)
                _, output_tensor_grad, _ = _communicate(
                    tensor_send_next=output_tensor if "cpu:gloo" != group.name() else output_tensor.cpu(),
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                    group=group,
                )
            else:
                tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                    rank=rank, local_tensor_shape=output_tensor.shape, next=True
                )
                output_tensor_grad = torch.empty(tensor_shape,
                                                device=torch.cuda.current_device() if "cpu:gloo" != pp_groups[0].name() else torch.device("cpu"),
                                                dtype=config.pipeline_dtype,
                                                requires_grad=True)
                if tensor_slices is not None:
                    for tensor_slice in tensor_slices:
                        dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                        output_tensor_sliced = output_tensor[sp_start:sp_end, dp_start:dp_end, :]
                        tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                        group = None
                        for pp_group in pp_groups:
                            pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                            if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                                group = pp_group
                                break
                        _, output_tensor_grad_sliced, _ = _communicate(
                            tensor_send_next=output_tensor_sliced.contiguous() if "cpu:gloo" != group.name() else output_tensor_sliced.cpu(),
                            tensor_send_prev=None,
                            recv_prev=False,
                            recv_next=True,
                            tensor_shape=tensor_shape_sliced,
                            group=group,
                        )
                        dp_coef = para_ctx.get_dp_coef_when_recv_backward()
                        if dp_coef != 1.0:
                            output_tensor_grad.data[sp_start:sp_end, dp_start:dp_end, :] = output_tensor_grad_sliced * dp_coef
                        else:
                            output_tensor_grad.data[sp_start:sp_end, dp_start:dp_end, :] = output_tensor_grad_sliced
            if config.timers is not None:
                config.timers('forward-send-backward-recv').stop()
        if output_tensor_grad is not None and output_tensor_grad.device == torch.device("cpu"):
            output_tensor_grad = output_tensor_grad.to(torch.cuda.current_device())
        output_tensor_grads.append(output_tensor_grad)
    if unwrap_output_tensors:
        return output_tensor_grads[0]
    return output_tensor_grads


def send_backward_recv_forward_hetero(
    input_tensor_grads: torch.Tensor,
    tensor_shapes: Shape,
    is_first_stage: bool,
    config: ModelParallelConfig,
    _communicate: callable = None,
) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    unwrap_input_tensor_grads = False
    if not isinstance(input_tensor_grads, list):
        unwrap_input_tensor_grads = True
        input_tensor_grads = [input_tensor_grads]
    if not isinstance(tensor_shapes, list):
        tensor_shapes = [tensor_shapes]
    input_tensors = []
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if is_first_stage:
            input_tensor = None
        else:
            if config.timers is not None:
                config.timers('backward-send-forward-recv', log_level=2).start()
            rank = torch.distributed.get_rank()
            para_ctx = get_parallel_context()
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True):
                group = para_ctx.get_pipeline_model_parallel_group(local_pp_group=True)
                input_tensor, _, _ = _communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad if "cpu:gloo" != group.name() else input_tensor_grad.cpu(),
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                    group=group,
                )
            else:
                tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                    rank=rank, local_tensor_shape=input_tensor_grad.shape, next=False
                )
                input_tensor = torch.empty(tensor_shape,
                                        device=torch.cuda.current_device() if "cpu:gloo" != pp_groups[0].name() else torch.device("cpu"),
                                        dtype=config.pipeline_dtype,
                                        requires_grad=True)
                if tensor_slices is not None:
                    for tensor_slice in tensor_slices:
                        dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                        input_tensor_grad_sliced = input_tensor_grad[sp_start:sp_end, dp_start:dp_end, :]
                        tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                        group = None
                        for pp_group in pp_groups:
                            pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                            if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                                group = pp_group
                                break
                        input_tensor_sliced, _, _ = _communicate(
                            tensor_send_next=None,
                            tensor_send_prev=input_tensor_grad_sliced.contiguous() if "cpu:gloo" != group.name() else input_tensor_grad_sliced.cpu(),
                            recv_prev=True,
                            recv_next=False,
                            tensor_shape=tensor_shape_sliced,
                            group=group,
                        )
                        input_tensor.data[sp_start:sp_end, dp_start:dp_end, :] = input_tensor_sliced
            if config.timers is not None:
                config.timers('backward-send-forward-recv').stop()
        if input_tensor is not None and input_tensor.device == torch.device("cpu"):
            input_tensor = input_tensor.to(torch.cuda.current_device())
        input_tensors.append(input_tensor)
    if unwrap_input_tensor_grads:
        return input_tensors[0]
    return input_tensors

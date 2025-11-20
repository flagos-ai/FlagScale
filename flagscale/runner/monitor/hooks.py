"""
Performance Monitor Training Hooks
This module provides hooks for integrating performance monitoring into the training loop.
"""

from typing import Any, Optional

import torch

from flagscale.runner.monitor.perf_metrics import FLOPSMeasurementCallback

# Global variable to store the performance monitor callback
_perf_monitor_callback: Optional[FLOPSMeasurementCallback] = None


def initialize_perf_monitor(args) -> Optional[FLOPSMeasurementCallback]:
    """
    Initialize the performance monitor if enabled.
    Args:
        args: Training arguments
    Returns:
        Performance monitor callback or None if disabled
    """
    global _perf_monitor_callback

    if not getattr(args, 'enable_perf_monitor', False):
        return None

    # Create performance monitor callback
    log_interval = getattr(args, 'perf_log_interval', 10)
    _perf_monitor_callback = FLOPSMeasurementCallback(args, log_interval=log_interval)

    # Log initialization
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[Performance Monitor] Initialized with log interval: {log_interval}")

    return _perf_monitor_callback


def get_perf_monitor() -> Optional[FLOPSMeasurementCallback]:
    """Get the global performance monitor callback."""
    return _perf_monitor_callback


def perf_monitor_start_iteration(iteration: int):
    """
    Hook to call at the start of each training iteration.
    Args:
        iteration: Current iteration number
    """
    if _perf_monitor_callback is not None:
        _perf_monitor_callback.on_train_batch_start(iteration)


def perf_monitor_end_iteration(iteration: int, writer=None, wandb_writer=None):
    """
    Hook to call at the end of each training iteration.
    Args:
        iteration: Current iteration number
        writer: Optional TensorBoard writer
        wandb_writer: Optional Weights & Biases writer
    """
    if _perf_monitor_callback is not None:
        _perf_monitor_callback.on_train_batch_end(iteration, writer, wandb_writer)


def perf_monitor_end_training(writer=None, wandb_writer=None):
    """
    Hook to call at the end of training.
    Args:
        writer: Optional TensorBoard writer
        wandb_writer: Optional Weights & Biases writer
    """
    if _perf_monitor_callback is not None:
        _perf_monitor_callback.on_train_end(writer, wandb_writer)


def add_perf_monitor_to_model_provider(original_model_provider):
    """
    Decorator to add performance monitor initialization to model provider.
    Args:
        original_model_provider: Original model provider function
    Returns:
        Wrapped model provider function
    """

    def wrapped_model_provider(pre_process=True, post_process=True):
        # Initialize performance monitor if enabled
        from megatron.training import get_args

        args = get_args()
        initialize_perf_monitor(args)

        # Call original model provider
        return original_model_provider(pre_process, post_process)

    return wrapped_model_provider


def wrap_forward_step_with_perf_monitor(forward_step_func):
    """
    Wrap the forward step function with performance monitoring.
    Args:
        forward_step_func: Original forward step function
    Returns:
        Wrapped forward step function
    """

    def wrapped_forward_step(data_iterator, model, *args, **kwargs):
        # Note: Actual iteration tracking would need to be handled at the training loop level
        # This is a placeholder for where hooks could be added
        result = forward_step_func(data_iterator, model, *args, **kwargs)
        return result

    return wrapped_forward_step

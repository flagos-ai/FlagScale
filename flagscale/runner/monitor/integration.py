"""
Integration module for performance monitoring with FlagScale training loop.
This module provides easy integration points for the performance monitoring
system with the existing FlagScale training infrastructure.
(NOTE: Temporarily still in the testing procedure, just a experiment feature)
"""

from typing import Any, Optional

import torch
import torch.distributed as dist

from flagscale.runner.monitor.perf_logger import PerfMonitorLogger
from flagscale.runner.monitor.perf_metrics import (
    FLOPSMeasurementCallback,
    ModelFLOPSCalculator,
    PerformanceMonitor,
)


def setup_performance_monitor(args) -> FLOPSMeasurementCallback:
    """
    Setup performance monitoring for training.
    Args:
        args: Training arguments
    Returns:
        FLOPSMeasurementCallback instance
    """
    # Determine log interval from args
    log_interval = getattr(args, 'perf_log_interval', getattr(args, 'log_interval', 100))

    # Create callback
    callback = FLOPSMeasurementCallback(args, log_interval=log_interval)

    # Print initialization message (only from rank 0)
    if not dist.is_initialized() or dist.get_rank() == 0:
        # Use the monitor's logger if available
        if hasattr(callback.monitor, 'file_logger'):
            callback.monitor.file_logger.logger.info(
                f"Performance monitor initialized. Logging every {log_interval} iterations."
            )

    return callback


def integrate_with_training_loop(
    args,
    iteration: int,
    writer: Optional[Any] = None,
    wandb_writer: Optional[Any] = None,
    stage: str = 'train_batch_end',
    callback: Optional[FLOPSMeasurementCallback] = None,
) -> Optional[FLOPSMeasurementCallback]:
    """
    Integration point for the training loop.
    This function should be called at appropriate points in the training loop:
    - 'train_batch_start': Beginning of each iteration
    - 'train_batch_end': End of each iteration
    - 'train_end': End of training
    Args:
        args: Training arguments
        iteration: Current iteration number
        writer: TensorBoard writer (optional)
        wandb_writer: Weights & Biases writer (optional)
        stage: Current stage of training
        callback: Existing callback instance (optional)
    Returns:
        FLOPSMeasurementCallback instance
    Example usage in train.py:
        ```python
        # At the beginning of training
        perf_callback = None
        # In the training loop
        for iteration in range(start_iteration, args.train_iters):
            # Start of iteration
            perf_callback = integrate_with_training_loop(
                args, iteration, stage='train_batch_start',
                callback=perf_callback
            )
            # ... training step ...
            # End of iteration
            perf_callback = integrate_with_training_loop(
                args, iteration, writer, wandb_writer,
                stage='train_batch_end', callback=perf_callback
            )
        # End of training
        integrate_with_training_loop(
            args, iteration, writer, wandb_writer,
            stage='train_end', callback=perf_callback
        )
        ```
    """
    # Initialize callback if not provided
    if callback is None:
        callback = setup_performance_monitor(args)

    # Call appropriate method based on stage
    if stage == 'train_batch_start':
        callback.on_train_batch_start(iteration)
    elif stage == 'train_batch_end':
        callback.on_train_batch_end(iteration, writer, wandb_writer)
    elif stage == 'train_end':
        callback.on_train_end(writer, wandb_writer)
    else:
        if torch.distributed.get_rank() == 0:
            print(f"Warning: Unknown stage '{stage}' in performance monitor integration")

    return callback


def add_performance_args(parser):
    """
    Add performance monitoring arguments to the argument parser.
    Args:
        parser: ArgumentParser instance
    Example usage:
        ```python
        parser = get_args_parser()
        add_performance_args(parser)
        ```
    """
    group = parser.add_argument_group(title='performance monitoring')

    group.add_argument(
        '--enable-perf-monitor',
        action='store_true',
        help='Enable performance monitoring and FLOPS tracking',
    )

    group.add_argument(
        '--perf-log-interval',
        type=int,
        default=100,
        help='Interval for logging performance metrics',
    )

    group.add_argument(
        '--perf-memory-tracking', action='store_true', help='Enable GPU memory usage tracking'
    )

    group.add_argument(
        '--perf-breakdown', action='store_true', help='Log detailed FLOPS breakdown by component'
    )

    return parser


def calculate_standalone_flops(args, batch_size: Optional[int] = None) -> dict:
    """
    Calculate FLOPS for a model configuration without training.
    This is useful for estimating FLOPS before training starts or
    for comparing different model configurations.
    Args:
        args: Model configuration arguments
        batch_size: Batch size (if None, uses args.micro_batch_size * args.num_micro_batches)
    Returns:
        Dictionary with FLOPS metrics
    Example usage:
        ```python
        from flagscale.runner.monitor.integration import calculate_standalone_flops
        # Get FLOPS estimate
        flops_info = calculate_standalone_flops(args)
        print(f"Model FLOPS: {flops_info['total_flops']/1e12:.2f} TFLOPS")
        print(f"FLOPS/GPU: {flops_info['flops_per_gpu']/1e12:.2f} TFLOPS")
        ```
    """
    if batch_size is None:
        batch_size = args.micro_batch_size * getattr(args, 'num_micro_batches', 1)

    calculator = ModelFLOPSCalculator(args)

    # Calculate total FLOPS
    total_flops = calculator.calculate_total_flops(batch_size)

    # Get breakdown if available
    breakdown = calculator.get_flops_breakdown()

    # Calculate per-GPU metrics
    world_size = getattr(args, 'world_size', 1)
    flops_per_gpu = total_flops / world_size

    return {
        'total_flops': total_flops,
        'flops_per_gpu': flops_per_gpu,
        'model_type': calculator.model_type,
        'batch_size': batch_size,
        'breakdown': breakdown,
    }


def log_model_flops_summary(args, writer=None, wandb_writer=None):
    """
    Log a summary of model FLOPS at the start of training.
    Args:
        args: Training arguments
        writer: TensorBoard writer (optional)
        wandb_writer: Weights & Biases writer (optional)
    """
    if torch.distributed.get_rank() != 0:
        return

    flops_info = calculate_standalone_flops(args)

    print("\n" + "=" * 80)
    print("MODEL FLOPS SUMMARY")
    print("=" * 80)
    print(f"Model Type:           {flops_info['model_type'].upper()}")
    print(f"Batch Size:           {flops_info['batch_size']}")
    print(f"Total FLOPS:          {flops_info['total_flops']/1e12:.2f} TFLOPS")
    print(f"FLOPS per GPU:        {flops_info['flops_per_gpu']/1e12:.2f} TFLOPS")

    if flops_info['breakdown']:
        print("\nFLOPS Breakdown:")
        for component, flops in flops_info['breakdown'].items():
            if component != 'total':
                print(f"  {component:12s}:      {flops/1e12:.2f} TFLOPS")

    print("=" * 80 + "\n")

    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('model/total_tflops', flops_info['total_flops'] / 1e12, 0)
        writer.add_scalar('model/tflops_per_gpu', flops_info['flops_per_gpu'] / 1e12, 0)

    # Log to Weights & Biases
    if wandb_writer is not None:
        wandb_writer.log(
            {
                'model/total_tflops': flops_info['total_flops'] / 1e12,
                'model/tflops_per_gpu': flops_info['flops_per_gpu'] / 1e12,
            },
            0,
        )

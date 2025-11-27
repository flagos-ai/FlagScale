#!/usr/bin/env python

"""
Example script demonstrating performance monitoring integration for GPT model training.
This script shows how to integrate the performance monitoring module with
FlagScale's training loop to track FLOPS, throughput, and other metrics.
Usage:
    python examples/monitor_example.py \
        --model-name gpt \
        --hidden-size 4096 \
        --num-layers 32 \
        --num-attention-heads 32 \
        --seq-length 2048 \
        --micro-batch-size 4 \
        --enable-perf-monitor
"""

import argparse
import time

from typing import Optional

import torch
import torch.distributed as dist

# Import performance monitoring components
from flagscale.runner.monitor import (
    FLOPSMeasurementCallback,
    ModelFLOPSCalculator,
    PerformanceMonitor,
)
from flagscale.runner.monitor.integration import (
    add_performance_args,
    calculate_standalone_flops,
    integrate_with_training_loop,
    log_model_flops_summary,
    setup_performance_monitor,
)


def create_mock_args():
    """Create mock arguments for demonstration."""
    parser = argparse.ArgumentParser(description='Performance Monitoring Example')

    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='gpt',
        choices=['gpt', 'llama', 'qwen', 'mixtral', 'aquila'],
        help='Model architecture type',
    )
    parser.add_argument('--hidden-size', type=int, default=4096, help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of transformer layers')
    parser.add_argument(
        '--num-attention-heads', type=int, default=32, help='Number of attention heads'
    )
    parser.add_argument('--seq-length', type=int, default=2048, help='Sequence length')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--ffn-hidden-size', type=int, default=16384, help='FFN hidden dimension')

    # Training configuration
    parser.add_argument('--micro-batch-size', type=int, default=4, help='Micro batch size per GPU')
    parser.add_argument('--num-micro-batches', type=int, default=2, help='Number of micro batches')
    parser.add_argument(
        '--train-iters', type=int, default=100, help='Number of training iterations'
    )

    # Distributed configuration
    parser.add_argument('--world-size', type=int, default=8, help='Number of GPUs')

    # Optional model features
    parser.add_argument('--swiglu', action='store_true', help='Use SwiGLU activation')
    parser.add_argument(
        '--num-query-groups', type=int, default=None, help='Number of query groups for GQA (LLaMA)'
    )

    # MoE configuration
    parser.add_argument(
        '--num-experts', type=int, default=8, help='Number of experts for MoE models'
    )
    parser.add_argument('--moe-router-topk', type=int, default=2, help='Top-k experts to select')

    # Add performance monitoring arguments
    add_performance_args(parser)

    args = parser.parse_args()

    # Set padded vocab size
    args.padded_vocab_size = args.vocab_size

    return args


def simulate_training_step(iteration: int, step_time: float = 0.1):
    """
    Simulate a training step with artificial delay.
    Args:
        iteration: Current iteration number
        step_time: Simulated step time in seconds
    """
    # Simulate computation time
    time.sleep(step_time + (0.01 * (iteration % 5 - 2)))  # Add some variation

    # Simulate loss calculation
    loss = 4.5 - (iteration * 0.01)  # Decreasing loss

    return loss


def example_1_basic_monitoring():
    """Example 1: Basic performance monitoring setup."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Performance Monitoring")
    print("=" * 80)

    # Create mock arguments
    args = create_mock_args()
    args.model_name = 'gpt'

    # Initialize performance monitor
    monitor = PerformanceMonitor(args, enable_memory_tracking=True)

    # Simulate training iterations
    for iteration in range(10):
        # Start iteration timing
        monitor.start_iteration()

        # Simulate training step
        loss = simulate_training_step(iteration, step_time=0.05)

        # End iteration timing
        monitor.end_iteration()

        # Update memory stats (if GPU available)
        monitor.update_memory_stats()

        # Calculate and display metrics every 5 iterations
        if (iteration + 1) % 5 == 0:
            metrics = monitor.calculate_metrics(iteration + 1)
            print(f"\nIteration {iteration + 1}:")
            print(f"  TFLOPS/GPU:      {metrics.tflops_per_gpu:.2f}")
            print(f"  Avg Step Time:   {metrics.avg_step_time*1000:.2f} ms")
            print(f"  Samples/Second:  {metrics.samples_per_second:.2f}")


def example_2_callback_integration():
    """Example 2: Using FLOPSMeasurementCallback."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Callback-based Integration")
    print("=" * 80)

    # Create mock arguments
    args = create_mock_args()
    args.model_name = 'llama'
    args.num_query_groups = 8  # Enable GQA for LLaMA

    # Setup performance monitoring callback
    callback = FLOPSMeasurementCallback(args, log_interval=5)

    # Simulate training loop
    for iteration in range(20):
        # Start batch
        callback.on_train_batch_start(iteration)

        # Simulate training
        loss = simulate_training_step(iteration, step_time=0.03)

        # End batch (logs every 5 iterations)
        callback.on_train_batch_end(iteration)

    # Final summary
    callback.on_train_end()


def example_3_standalone_flops_calculation():
    """Example 3: Calculate FLOPS for different model configurations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Standalone FLOPS Calculation")
    print("=" * 80)

    configurations = [
        {
            'name': 'GPT-3 (175B)',
            'hidden_size': 12288,
            'num_layers': 96,
            'num_attention_heads': 96,
            'model_name': 'gpt',
        },
        {
            'name': 'LLaMA-7B',
            'hidden_size': 4096,
            'num_layers': 32,
            'num_attention_heads': 32,
            'num_query_groups': 32,
            'model_name': 'llama',
        },
        {
            'name': 'Mixtral-8x7B',
            'hidden_size': 4096,
            'num_layers': 32,
            'num_attention_heads': 32,
            'num_experts': 8,
            'moe_router_topk': 2,
            'model_name': 'mixtral',
        },
    ]

    for config in configurations:
        # Create args for this configuration
        args = create_mock_args()
        for key, value in config.items():
            if key != 'name':
                setattr(args, key, value)

        # Calculate FLOPS
        flops_info = calculate_standalone_flops(args, batch_size=32)

        print(f"\n{config['name']}:")
        print(f"  Model Type:      {flops_info['model_type'].upper()}")
        print(f"  Total FLOPS:     {flops_info['total_flops']/1e15:.2f} PFLOPS")
        print(f"  FLOPS per GPU:   {flops_info['flops_per_gpu']/1e12:.2f} TFLOPS")

        # Show breakdown if available
        if flops_info['breakdown']:
            print("  Breakdown:")
            for component, flops in flops_info['breakdown'].items():
                if component not in ['forward', 'backward', 'total']:
                    print(f"    {component:10s}:    {flops/1e12:.2f} TFLOPS")


def example_4_integration_with_training_loop():
    """Example 4: Integration with training loop using helper functions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Training Loop Integration")
    print("=" * 80)

    # Create mock arguments
    args = create_mock_args()
    args.model_name = 'qwen'
    args.train_iters = 30

    # Log initial model FLOPS summary
    log_model_flops_summary(args)

    # Initialize callback
    perf_callback = None

    # Simulate training loop
    for iteration in range(args.train_iters):
        # Start of iteration
        perf_callback = integrate_with_training_loop(
            args, iteration, stage='train_batch_start', callback=perf_callback
        )

        # Simulate training step
        loss = simulate_training_step(iteration, step_time=0.02)

        # End of iteration
        perf_callback = integrate_with_training_loop(
            args, iteration, stage='train_batch_end', callback=perf_callback
        )

        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Completed iteration {iteration + 1}/{args.train_iters}, Loss: {loss:.4f}")

    # End of training
    integrate_with_training_loop(args, iteration, stage='train_end', callback=perf_callback)


def example_5_detailed_breakdown():
    """Example 5: Detailed FLOPS breakdown by component."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Detailed FLOPS Breakdown")
    print("=" * 80)

    # Create mock arguments for a specific model
    args = create_mock_args()
    args.model_name = 'gpt'
    args.hidden_size = 2048
    args.num_layers = 24
    args.num_attention_heads = 16
    args.seq_length = 1024
    args.ffn_hidden_size = 8192
    args.swiglu = True

    # Create calculator
    calculator = ModelFLOPSCalculator(args)

    # Calculate total FLOPS
    batch_size = args.micro_batch_size * args.num_micro_batches
    total_flops = calculator.calculate_total_flops(batch_size)

    # Get detailed breakdown
    breakdown = calculator.get_flops_breakdown()

    print(f"\nModel Configuration:")
    print(f"  Architecture:    {calculator.model_type.upper()}")
    print(f"  Hidden Size:     {args.hidden_size}")
    print(f"  Layers:          {args.num_layers}")
    print(f"  Attention Heads: {args.num_attention_heads}")
    print(f"  Sequence Length: {args.seq_length}")
    print(f"  FFN Size:        {args.ffn_hidden_size}")
    print(f"  SwiGLU:          {args.swiglu}")

    print(f"\nFLOPS Analysis:")
    print(f"  Total FLOPS:     {total_flops/1e12:.2f} TFLOPS")

    if breakdown:
        print(f"\nComponent Breakdown:")
        print(
            f"  Attention:       {breakdown['attention']/1e12:.2f} TFLOPS "
            f"({100*breakdown['attention']/breakdown['total']:.1f}%)"
        )
        print(
            f"  FFN:             {breakdown['ffn']/1e12:.2f} TFLOPS "
            f"({100*breakdown['ffn']/breakdown['total']:.1f}%)"
        )
        print(
            f"  Forward Pass:    {breakdown['forward']/1e12:.2f} TFLOPS "
            f"({100*breakdown['forward']/breakdown['total']:.1f}%)"
        )
        print(
            f"  Backward Pass:   {breakdown['backward']/1e12:.2f} TFLOPS "
            f"({100*breakdown['backward']/breakdown['total']:.1f}%)"
        )


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FLAGSCALE PERFORMANCE MONITORING EXAMPLES")
    print("=" * 80)

    # Run examples
    example_1_basic_monitoring()
    example_2_callback_integration()
    example_3_standalone_flops_calculation()
    example_4_integration_with_training_loop()
    example_5_detailed_breakdown()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    # Initialize distributed if needed (mock for example)
    if not dist.is_initialized():
        import os

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        if torch.cuda.is_available():
            dist.init_process_group(backend='nccl', rank=0, world_size=1)
        else:
            dist.init_process_group(backend='gloo', rank=0, world_size=1)

    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

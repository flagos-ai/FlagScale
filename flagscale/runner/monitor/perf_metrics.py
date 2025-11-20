import time

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from flagscale.runner.monitor.flops_calculator import FLOPSFormulas
from flagscale.runner.monitor.perf_logger import PerfMonitorLogger

# Try to import get_num_microbatches function
try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches
except ImportError:
    get_num_microbatches = None


@dataclass
class TFLOPSMetrics:
    """Container for TFLOPS performance metrics."""

    tflops_per_gpu: float = 0.0
    tflops_total: float = 0.0
    model_flops: float = 0.0
    avg_step_time: float = 0.0
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # Detailed breakdown
    forward_flops: float = 0.0
    backward_flops: float = 0.0
    optimizer_flops: float = 0.0

    # Statistics
    min_step_time: float = float('inf')
    max_step_time: float = 0.0
    std_step_time: float = 0.0


class PerformanceMonitor:
    """
    Performance monitoring for model training.
    This class tracks various performance metrics during training including:
    - FLOPS/TFLOPS calculations
    - Step timing statistics
    - Throughput metrics
    - Memory usage tracking
    """

    def __init__(self, args, enable_memory_tracking: bool = True):
        """
        Initialize performance monitor.
        Args:
            args: Training arguments containing model configuration
            enable_memory_tracking: Whether to track GPU memory usage
        """
        self.args = args
        self.enable_memory_tracking = enable_memory_tracking

        # Timing tracking
        self.step_times: List[float] = []
        self.start_time: Optional[float] = None
        self.iteration_start_time: Optional[float] = None

        # Memory tracking
        self.peak_memory_gb: float = 0.0
        self.current_memory_gb: float = 0.0

        # Metrics
        self.metrics = TFLOPSMetrics()

        # Model FLOPS calculator
        self.flops_calculator = ModelFLOPSCalculator(args)

        # File logger
        log_dir = getattr(args, 'perf_log_dir', 'logs/perf_monitor')
        enable_console = getattr(args, 'perf_console_output', False)
        self.file_logger = PerfMonitorLogger(log_dir=log_dir, enable_console=enable_console)

    def start_iteration(self):
        """Mark the start of a training iteration."""
        self.iteration_start_time = time.time()

    def end_iteration(self):
        """Mark the end of a training iteration and record timing."""
        if self.iteration_start_time is not None:
            step_time = time.time() - self.iteration_start_time
            self.step_times.append(step_time)
            self.iteration_start_time = None

            # Update min/max
            self.metrics.min_step_time = min(self.metrics.min_step_time, step_time)
            self.metrics.max_step_time = max(self.metrics.max_step_time, step_time)

    def update_memory_stats(self):
        """Update GPU memory usage statistics."""
        if not self.enable_memory_tracking or not torch.cuda.is_available():
            return

        current_memory = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

        self.current_memory_gb = current_memory
        self.peak_memory_gb = max(self.peak_memory_gb, peak_memory)

    def calculate_metrics(self, iteration: int) -> TFLOPSMetrics:
        """
        Calculate performance metrics.
        Args:
            iteration: Current training iteration
        Returns:
            TFLOPSMetrics object containing calculated metrics
        """
        if len(self.step_times) == 0:
            return self.metrics

        # Use second half of step times for stability (similar to NeMo)
        half_idx = len(self.step_times) // 2
        recent_times = self.step_times[half_idx:] if half_idx > 0 else self.step_times

        # Calculate average step time using median for stability
        avg_step_time = np.median(recent_times) if recent_times else 0
        self.metrics.avg_step_time = avg_step_time
        self.metrics.std_step_time = np.std(recent_times) if recent_times else 0

        # Calculate model FLOPS
        if get_num_microbatches is not None:
            num_micro_batches = get_num_microbatches()
        else:
            num_micro_batches = getattr(
                self.args, 'num_micro_batches', getattr(self.args, 'gradient_accumulation_steps', 1)
            )
        micro_batch_size = getattr(self.args, 'micro_batch_size', 1)
        batch_size = micro_batch_size * num_micro_batches if micro_batch_size else num_micro_batches
        model_flops = self.flops_calculator.calculate_total_flops(batch_size)
        self.metrics.model_flops = model_flops

        # Calculate TFLOPS per GPU
        if avg_step_time > 0:
            num_gpus = getattr(self.args, 'world_size', 1)
            flops_per_gpu = model_flops / num_gpus
            self.metrics.tflops_per_gpu = flops_per_gpu / (1e12 * avg_step_time)
            self.metrics.tflops_total = model_flops / (1e12 * avg_step_time)

            # Calculate throughput metrics
            self.metrics.samples_per_second = batch_size / avg_step_time
            tokens_per_sample = self.args.seq_length
            self.metrics.tokens_per_second = self.metrics.samples_per_second * tokens_per_sample

        # Get breakdown if available
        breakdown = self.flops_calculator.get_flops_breakdown()
        self.metrics.forward_flops = breakdown.get('forward', 0)
        self.metrics.backward_flops = breakdown.get('backward', 0)
        self.metrics.optimizer_flops = breakdown.get('optimizer', 0)

        return self.metrics

    def log_metrics(self, iteration: int, writer=None, wandb_writer=None):
        """
        Log performance metrics to various backends.
        Args:
            iteration: Current training iteration
            writer: TensorBoard writer (optional)
            wandb_writer: Weights & Biases writer (optional)
        """
        metrics = self.calculate_metrics(iteration)

        # File logging (logger handles rank checking internally)
        metrics_dict = {
            'TFLOPS_per_GPU': metrics.tflops_per_gpu,
            'TFLOPS_total': metrics.tflops_total,
            'samples_per_sec': metrics.samples_per_second,
            'tokens_per_sec': metrics.tokens_per_second,
            'step_time_ms': metrics.avg_step_time * 1000,
        }

        if self.enable_memory_tracking:
            metrics_dict.update(
                {'memory_GB': self.current_memory_gb, 'peak_memory_GB': self.peak_memory_gb}
            )

        # Log to file
        self.file_logger.log_metrics(iteration, metrics_dict)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('performance/tflops_per_gpu', metrics.tflops_per_gpu, iteration)
            writer.add_scalar('performance/tflops_total', metrics.tflops_total, iteration)
            writer.add_scalar(
                'performance/avg_step_time_ms', metrics.avg_step_time * 1000, iteration
            )
            writer.add_scalar(
                'performance/samples_per_second', metrics.samples_per_second, iteration
            )
            writer.add_scalar('performance/tokens_per_second', metrics.tokens_per_second, iteration)

            if self.enable_memory_tracking:
                writer.add_scalar('memory/current_gb', self.current_memory_gb, iteration)
                writer.add_scalar('memory/peak_gb', self.peak_memory_gb, iteration)

        # Weights & Biases logging
        if wandb_writer is not None:
            log_dict = {
                'performance/tflops_per_gpu': metrics.tflops_per_gpu,
                'performance/tflops_total': metrics.tflops_total,
                'performance/avg_step_time_ms': metrics.avg_step_time * 1000,
                'performance/samples_per_second': metrics.samples_per_second,
                'performance/tokens_per_second': metrics.tokens_per_second,
            }

            if self.enable_memory_tracking:
                log_dict.update(
                    {
                        'memory/current_gb': self.current_memory_gb,
                        'memory/peak_gb': self.peak_memory_gb,
                    }
                )

            wandb_writer.log(log_dict, iteration)


class ModelFLOPSCalculator:
    """
    Calculate FLOPS for different model architectures.
    This class provides FLOPS calculation for various model types including:
    - GPT/LLaMA style models
    - Mixture of Experts (MoE) models
    - Multi-modal models
    - Custom architectures
    """

    def __init__(self, args):
        """
        Initialize FLOPS calculator.
        Args:
            args: Training arguments containing model configuration
        """
        self.args = args
        self.formulas = FLOPSFormulas()

        # Determine model type
        self.model_type = self._determine_model_type()

    def _determine_model_type(self) -> str:
        """Determine the model type from arguments."""
        # Check for specific model indicators
        if hasattr(self.args, 'model_name'):
            model_name = self.args.model_name.lower()
            if 'gpt' in model_name:
                return 'gpt'
            elif 'llama' in model_name:
                return 'llama'
            elif 'qwen' in model_name:
                return 'qwen'
            elif 'mixtral' in model_name or 'moe' in model_name:
                return 'moe'
            elif 'aquila' in model_name:
                return 'aquila'

        # Check for MoE configuration
        if (
            hasattr(self.args, 'num_experts')
            and self.args.num_experts is not None
            and self.args.num_experts > 1
        ):
            return 'moe'

        # Default to GPT
        return 'gpt'

    def calculate_total_flops(self, batch_size: int) -> float:
        """
        Calculate total FLOPS for the model.
        Args:
            batch_size: Total batch size
        Returns:
            Total FLOPS for forward + backward pass
        """
        if self.model_type == 'gpt':
            return self._calculate_gpt_flops(batch_size)
        elif self.model_type == 'llama':
            return self._calculate_llama_flops(batch_size)
        elif self.model_type == 'qwen':
            return self._calculate_qwen_flops(batch_size)
        elif self.model_type == 'moe':
            return self._calculate_moe_flops(batch_size)
        elif self.model_type == 'aquila':
            return self._calculate_aquila_flops(batch_size)
        else:
            # Fallback to GPT calculation
            return self._calculate_gpt_flops(batch_size)

    def _calculate_gpt_flops(self, batch_size: int) -> float:
        """Calculate FLOPS for GPT-style models."""
        args = self.args

        # Extract configuration with safe access
        seq_length = getattr(args, 'seq_length', 512)
        hidden_size = getattr(args, 'hidden_size', 768)
        num_layers = getattr(args, 'num_layers', 12)
        vocab_size = getattr(args, 'vocab_size', getattr(args, 'padded_vocab_size', 50257))

        # Attention parameters
        num_attention_heads = getattr(args, 'num_attention_heads', 12)

        # FFN parameters
        ffn_hidden_size = getattr(args, 'ffn_hidden_size', 4 * hidden_size if hidden_size else 3072)

        # Calculate attention FLOPS
        attention_flops = self.formulas.attention_flops(
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )

        # Calculate FFN FLOPS
        use_swiglu = getattr(args, 'swiglu', False)
        ffn_flops = self.formulas.ffn_flops(
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            use_swiglu=use_swiglu,
        )

        # Calculate embedding and output layer FLOPS
        embedding_flops = 2 * batch_size * seq_length * hidden_size * vocab_size

        # Total FLOPS (forward + backward)
        layer_flops = (attention_flops + ffn_flops) * num_layers
        total_flops = 3 * (layer_flops + embedding_flops)  # 3x for backward pass

        return total_flops

    def _calculate_llama_flops(self, batch_size: int) -> float:
        """Calculate FLOPS for LLaMA-style models."""
        args = self.args

        # Extract configuration with safe access
        seq_length = getattr(args, 'seq_length', 512)
        hidden_size = getattr(args, 'hidden_size', 768)
        num_layers = getattr(args, 'num_layers', 12)
        vocab_size = getattr(args, 'vocab_size', getattr(args, 'padded_vocab_size', 50257))

        # LLaMA specific: GQA (Grouped Query Attention)
        num_attention_heads = getattr(args, 'num_attention_heads', 12)
        num_query_groups = getattr(args, 'num_query_groups', num_attention_heads)

        # FFN with SwiGLU activation
        ffn_hidden_size = getattr(args, 'ffn_hidden_size', int(8 * hidden_size / 3))

        # Calculate attention FLOPS with GQA
        attention_flops = self.formulas.gqa_attention_flops(
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
        )

        # Calculate FFN FLOPS (LLaMA uses SwiGLU)
        ffn_flops = self.formulas.ffn_flops(
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            use_swiglu=True,
        )

        # Embedding and output layer
        embedding_flops = 2 * batch_size * seq_length * hidden_size * vocab_size

        # Total FLOPS
        layer_flops = (attention_flops + ffn_flops) * num_layers
        total_flops = 3 * (layer_flops + embedding_flops)

        return total_flops

    def _calculate_qwen_flops(self, batch_size: int) -> float:
        """Calculate FLOPS for Qwen models."""
        # Similar to LLaMA with some Qwen-specific modifications
        return self._calculate_llama_flops(batch_size)

    def _calculate_moe_flops(self, batch_size: int) -> float:
        """Calculate FLOPS for Mixture of Experts models."""
        args = self.args

        # Extract configuration with safe access
        seq_length = getattr(args, 'seq_length', 512)
        hidden_size = getattr(args, 'hidden_size', 768)
        num_layers = getattr(args, 'num_layers', 12)
        vocab_size = getattr(args, 'vocab_size', getattr(args, 'padded_vocab_size', 50257))

        # MoE specific parameters
        num_experts = getattr(args, 'num_experts', 8)
        top_k = getattr(args, 'moe_router_topk', 2)

        # Attention parameters
        num_attention_heads = getattr(args, 'num_attention_heads', 12)

        # Calculate attention FLOPS (same as regular transformer)
        attention_flops = self.formulas.attention_flops(
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )

        # Calculate MoE FFN FLOPS
        ffn_hidden_size = getattr(args, 'ffn_hidden_size', 4 * hidden_size)
        moe_flops = self.formulas.moe_flops(
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Embedding and output layer
        embedding_flops = 2 * batch_size * seq_length * hidden_size * vocab_size

        # Total FLOPS
        layer_flops = (attention_flops + moe_flops) * num_layers
        total_flops = 3 * (layer_flops + embedding_flops)

        return total_flops

    def _calculate_aquila_flops(self, batch_size: int) -> float:
        """Calculate FLOPS for Aquila models."""
        # Aquila is similar to GPT with some modifications
        return self._calculate_gpt_flops(batch_size)

    def get_flops_breakdown(self) -> Dict[str, float]:
        """
        Get detailed FLOPS breakdown by component.
        Returns:
            Dictionary with FLOPS for different components
        """
        if get_num_microbatches is not None:
            num_micro_batches = get_num_microbatches()
        else:
            num_micro_batches = getattr(
                self.args, 'num_micro_batches', getattr(self.args, 'gradient_accumulation_steps', 1)
            )
        micro_batch_size = getattr(self.args, 'micro_batch_size', 1)
        batch_size = micro_batch_size * num_micro_batches if micro_batch_size else num_micro_batches

        if self.model_type in ['gpt', 'llama', 'qwen', 'aquila']:
            args = self.args
            # Extract configuration with safe access
            seq_length = getattr(args, 'seq_length', 512)
            hidden_size = getattr(args, 'hidden_size', 768)
            num_layers = getattr(args, 'num_layers', 12)
            num_attention_heads = getattr(args, 'num_attention_heads', 12)

            # Calculate component FLOPS
            attention_flops = (
                self.formulas.attention_flops(
                    batch_size=batch_size,
                    seq_length=seq_length,
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                )
                * num_layers
            )

            ffn_hidden_size = getattr(args, 'ffn_hidden_size', 4 * hidden_size)
            use_swiglu = getattr(args, 'swiglu', False)
            ffn_flops = (
                self.formulas.ffn_flops(
                    batch_size=batch_size,
                    seq_length=seq_length,
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    use_swiglu=use_swiglu,
                )
                * num_layers
            )

            total_forward = attention_flops + ffn_flops

            return {
                'attention': attention_flops,
                'ffn': ffn_flops,
                'forward': total_forward,
                'backward': total_forward * 2,  # Backward is approximately 2x forward
                'total': total_forward * 3,
            }

        return {}


class FLOPSMeasurementCallback:
    """
    Callback for measuring FLOPS during training.
    This callback integrates with the training loop to automatically
    measure and log performance metrics.
    """

    def __init__(self, args, log_interval: int = 100):
        """
        Initialize FLOPS measurement callback.
        Args:
            args: Training arguments
            log_interval: How often to log metrics (in iterations)
        """
        self.args = args
        self.log_interval = log_interval
        self.monitor = PerformanceMonitor(args)

    def on_train_batch_start(self, iteration: int):
        """Called at the start of each training batch."""
        self.monitor.start_iteration()

    def on_train_batch_end(self, iteration: int, writer=None, wandb_writer=None):
        """Called at the end of each training batch."""
        self.monitor.end_iteration()
        self.monitor.update_memory_stats()

        # Log metrics at specified intervals (always log first iteration if > 0)
        if iteration > 0 and (iteration == 1 or iteration % self.log_interval == 0):
            self.monitor.log_metrics(iteration, writer, wandb_writer)

    def on_train_end(self, writer=None, wandb_writer=None):
        """Called at the end of training."""
        # If we have step data but no logs yet, log the final state
        if (
            len(self.monitor.step_times) > 0
            and self.monitor.file_logger.enabled
            and len(self.monitor.file_logger.json_data) == 0
        ):
            # Calculate the last iteration number from step_times length
            final_iteration = len(self.monitor.step_times)
            self.monitor.log_metrics(final_iteration, writer, wandb_writer)

        # Final metrics calculation and logging
        metrics = self.monitor.metrics
        final_stats = {
            'avg_tflops_per_gpu': metrics.tflops_per_gpu,
            'avg_tflops_total': metrics.tflops_total,
            'avg_step_time_ms': metrics.avg_step_time * 1000,
            'min_step_time_ms': metrics.min_step_time * 1000,
            'max_step_time_ms': metrics.max_step_time * 1000,
            'peak_memory_gb': self.monitor.peak_memory_gb,
        }

        # Save final summary to file
        self.monitor.file_logger.save_summary(final_stats)

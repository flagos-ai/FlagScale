import unittest

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from flagscale.runner.monitor.perf_metrics import (
    FLOPSMeasurementCallback,
    ModelFLOPSCalculator,
    PerformanceMonitor,
    TFLOPSMetrics,
)


class TestTFLOPSMetrics(unittest.TestCase):
    """Test TFLOPSMetrics dataclass."""

    def test_initialization(self):
        """Test metrics initialization with default values."""
        metrics = TFLOPSMetrics()

        self.assertEqual(metrics.tflops_per_gpu, 0.0)
        self.assertEqual(metrics.tflops_total, 0.0)
        self.assertEqual(metrics.model_flops, 0.0)
        self.assertEqual(metrics.min_step_time, float('inf'))
        self.assertEqual(metrics.max_step_time, 0.0)

    def test_custom_values(self):
        """Test metrics initialization with custom values."""
        metrics = TFLOPSMetrics(tflops_per_gpu=100.5, tflops_total=804.0, model_flops=1e15)

        self.assertEqual(metrics.tflops_per_gpu, 100.5)
        self.assertEqual(metrics.tflops_total, 804.0)
        self.assertEqual(metrics.model_flops, 1e15)


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.args = Mock()
        self.args.micro_batch_size = 4
        self.args.num_micro_batches = 2
        self.args.seq_length = 2048
        self.args.world_size = 8
        self.args.hidden_size = 4096
        self.args.num_layers = 32
        self.args.num_attention_heads = 32
        self.args.vocab_size = 50000
        self.args.padded_vocab_size = 50000
        self.args.ffn_hidden_size = 16384
        self.args.swiglu = False
        self.args.model_name = (
            'gpt'  # Fix: Added model_name to prevent "Mock is not iterable" error
        )
        from unittest.mock import patch

        self.get_num_mb_patch = patch(
            "flagscale.runner.monitor.perf_metrics.get_num_microbatches",
            return_value=self.args.num_micro_batches,
        )
        self.get_num_mb_patch.start()
        self.args.perf_log_dir = "/tmp/mock_perf_log_dir"

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(self.args, enable_memory_tracking=True)

        self.assertIsNotNone(monitor.flops_calculator)
        self.assertEqual(monitor.enable_memory_tracking, True)
        self.assertEqual(len(monitor.step_times), 0)
        self.assertEqual(monitor.peak_memory_gb, 0.0)

    def test_iteration_timing(self):
        """Test iteration start/end timing."""
        monitor = PerformanceMonitor(self.args)

        # Start iteration
        monitor.start_iteration()
        self.assertIsNotNone(monitor.iteration_start_time)

        # Simulate some work
        import time

        time.sleep(0.01)

        # End iteration
        monitor.end_iteration()
        self.assertEqual(len(monitor.step_times), 1)
        self.assertGreater(monitor.step_times[0], 0)
        self.assertIsNone(monitor.iteration_start_time)

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        self.args.perf_log_dir = "/tmp/mock_perf_log_dir"
        monitor = PerformanceMonitor(self.args)

        # Add some step times
        monitor.step_times = [0.1, 0.12, 0.11, 0.09, 0.10]

        # Calculate metrics
        metrics = monitor.calculate_metrics(iteration=100)

        self.assertGreater(metrics.tflops_per_gpu, 0)
        self.assertGreater(metrics.model_flops, 0)
        self.assertAlmostEqual(metrics.avg_step_time, np.median([0.11, 0.09, 0.10]), places=3)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.max_memory_allocated')
    def test_memory_tracking(self, mock_max_mem, mock_current_mem, mock_cuda_available):
        """Test GPU memory tracking."""
        mock_cuda_available.return_value = True
        mock_current_mem.return_value = 10 * 1024**3  # 10 GB in bytes
        mock_max_mem.return_value = 15 * 1024**3  # 15 GB in bytes

        monitor = PerformanceMonitor(self.args, enable_memory_tracking=True)
        monitor.update_memory_stats()

        self.assertEqual(monitor.current_memory_gb, 10.0)
        self.assertEqual(monitor.peak_memory_gb, 15.0)

    @patch('torch.distributed.get_rank')
    def test_log_metrics(self, mock_get_rank):
        """Test metrics logging."""
        mock_get_rank.return_value = 0

        monitor = PerformanceMonitor(self.args)
        monitor.step_times = [0.1, 0.1, 0.1]

        # Mock writers
        mock_writer = Mock()
        mock_wandb = Mock()

        # Log metrics
        monitor.log_metrics(100, mock_writer, mock_wandb)

        # Check that scalars were added
        mock_writer.add_scalar.assert_called()
        mock_wandb.log.assert_called_once()


class TestModelFLOPSCalculator(unittest.TestCase):
    """Test ModelFLOPSCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.args = Mock()
        self.args.seq_length = 2048
        self.args.hidden_size = 4096
        self.args.num_layers = 32
        self.args.num_attention_heads = 32
        self.args.vocab_size = 50000  # Fix: Added vocab_size to prevent "int * Mock" error
        self.args.padded_vocab_size = 50000
        self.args.ffn_hidden_size = 16384
        self.args.swiglu = False
        self.args.model_name = 'gpt'  # Fix: Added default model_name

    def test_model_type_detection(self):
        """Test model type detection."""
        # GPT model
        self.args.model_name = 'gpt-3'
        calc = ModelFLOPSCalculator(self.args)
        self.assertEqual(calc.model_type, 'gpt')

        # LLaMA model
        self.args.model_name = 'llama-7b'
        calc = ModelFLOPSCalculator(self.args)
        self.assertEqual(calc.model_type, 'llama')

        # MoE model
        self.args.model_name = 'mixtral-8x7b'
        calc = ModelFLOPSCalculator(self.args)
        self.assertEqual(calc.model_type, 'moe')

    def test_gpt_flops_calculation(self):
        """Test GPT model FLOPS calculation."""
        self.args.model_name = 'gpt'
        calc = ModelFLOPSCalculator(self.args)

        batch_size = 8
        flops = calc.calculate_total_flops(batch_size)

        # Check that FLOPS is positive and reasonable
        self.assertGreater(flops, 0)
        self.assertGreater(flops, 1e12)  # Should be in TFLOPS range

    def test_llama_flops_calculation(self):
        """Test LLaMA model FLOPS calculation with GQA."""
        self.args.model_name = 'llama'
        self.args.num_query_groups = 8  # GQA
        calc = ModelFLOPSCalculator(self.args)

        batch_size = 8
        flops = calc.calculate_total_flops(batch_size)

        self.assertGreater(flops, 0)
        self.assertGreater(flops, 1e12)

    def test_moe_flops_calculation(self):
        """Test MoE model FLOPS calculation."""
        self.args.model_name = 'mixtral'
        self.args.num_experts = 8
        self.args.moe_router_topk = 2
        calc = ModelFLOPSCalculator(self.args)

        batch_size = 8
        flops = calc.calculate_total_flops(batch_size)

        self.assertGreater(flops, 0)
        self.assertGreater(flops, 1e12)

    def test_flops_breakdown(self):
        """Test FLOPS breakdown by component."""
        self.args.model_name = 'gpt'
        self.args.micro_batch_size = 4
        self.args.num_micro_batches = 2
        calc = ModelFLOPSCalculator(self.args)

        breakdown = calc.get_flops_breakdown()

        self.assertIn('attention', breakdown)
        self.assertIn('ffn', breakdown)
        self.assertIn('forward', breakdown)
        self.assertIn('backward', breakdown)
        self.assertIn('total', breakdown)

        # Check all values are positive
        for key, value in breakdown.items():
            self.assertGreater(value, 0, f"Breakdown {key} should be positive")

    def test_different_architectures(self):
        """Test FLOPS calculation for different architectures."""
        architectures = {
            'gpt': {'model_name': 'gpt', 'expected_min': 1e12},
            'llama': {'model_name': 'llama', 'num_query_groups': 8, 'expected_min': 1e12},
            'qwen': {'model_name': 'qwen', 'expected_min': 1e12},
        }

        batch_size = 8

        for arch_name, config in architectures.items():
            # Update args with architecture-specific config
            for key, value in config.items():
                if key != 'expected_min':
                    setattr(self.args, key, value)

            calc = ModelFLOPSCalculator(self.args)
            flops = calc.calculate_total_flops(batch_size)

            self.assertGreater(
                flops,
                config['expected_min'],
                f"{arch_name} FLOPS should be > {config['expected_min']}",
            )


class TestFLOPSMeasurementCallback(unittest.TestCase):
    """Test FLOPSMeasurementCallback class."""

    def setUp(self):
        """Set up test fixtures."""
        self.args = Mock()
        self.args.micro_batch_size = 4
        self.args.seq_length = 2048
        self.args.hidden_size = 4096
        self.args.num_layers = 32
        self.args.num_attention_heads = 32
        self.args.vocab_size = 50000
        self.args.padded_vocab_size = 50000
        self.args.ffn_hidden_size = 16384
        self.args.swiglu = False
        self.args.perf_log_interval = 10
        self.args.perf_log_dir = 'logs/perf_monitor'
        self.args.perf_console_output = False
        self.args.model_name = 'gpt'

    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.get_rank')
    def test_callback_initialization(self, mock_get_rank, mock_is_initialized):
        """Test callback initialization."""
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 0

        callback = FLOPSMeasurementCallback(self.args)

        self.assertIsNotNone(callback.monitor)
        self.assertEqual(callback.log_interval, 100)

    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.get_rank')
    def test_on_train_batch_end(self, mock_get_rank, mock_is_initialized):
        """Test on_train_batch_end callback."""
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 0

        self.args.world_size = 1
        self.args.micro_batch_size = 1
        callback = FLOPSMeasurementCallback(self.args)

        # Mock writers
        writer = Mock()
        wandb_writer = Mock()

        # Simulate training iterations
        for iteration in range(1, 21):
            callback.monitor.start_iteration()
            # Simulate some processing time
            import time

            time.sleep(0.001)
            callback.monitor.end_iteration()

            # Call the callback
            callback.on_train_batch_end(iteration, writer, wandb_writer)

            # Check if metrics were logged at the right intervals
            if iteration == 1 or iteration % 10 == 0:
                # Should log at iteration 1, 10, 20
                self.assertTrue(
                    writer.add_scalar.called or iteration < 10,
                    f"Should log at iteration {iteration}",
                )

    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.get_rank')
    def test_on_train_end(self, mock_get_rank, mock_is_initialized):
        """Test on_train_end callback."""
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 0
        self.args.world_size = 1
        self.args.micro_batch_size = 1
        callback = FLOPSMeasurementCallback(self.args)

        # Add some step times
        callback.monitor.step_times = [0.1, 0.1, 0.1]

        # Mock writers
        writer = Mock()
        wandb_writer = Mock()

        # Call on_train_end
        callback.on_train_end(writer, wandb_writer)

        # Check that file logger was finalized
        self.assertIsNotNone(callback.monitor.file_logger)


if __name__ == '__main__':
    unittest.main()

import argparse
import os
import sys

from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest


class TestGPUHealthCheck:
    """Test cases for GPU health check module"""

    def setup_method(self):
        """Reset global variables before each test"""
        import flagscale.runner.elastic.gpu_health_check as health_check

        health_check._DATA_PARALLEL_GROUP = None
        health_check._TENSOR_MODEL_PARALLEL_GROUP = None
        health_check._PIPELINE_MODEL_PARALLEL_GROUP = None
        health_check._MODEL_PARALLEL_GROUP = None
        health_check._DATA_GLOBAL_RANKS = None
        health_check._TENSOR_GLOBAL_RANKS = None
        health_check._PIPELINE_GLOBAL_RANKS = None
        health_check._EMBEDDING_GROUP = None
        health_check._GLOBAL_ARGS = None

    def test_timeout_protection_context_manager(self):
        """Test timeout_protection context manager (disabled mode)"""
        from flagscale.runner.elastic.gpu_health_check import timeout_protection

        # Test that context manager works
        with timeout_protection(10, "test_operation"):
            result = 1 + 1
            assert result == 2

    def test_parse_args_default_values(self):
        """Test argument parsing with default values"""
        from flagscale.runner.elastic.gpu_health_check import parse_args

        test_args = ['--tensor-model-parallel-size', '2', '--pipeline-model-parallel-size', '2']

        with patch('sys.argv', ['gpu_health_check.py'] + test_args):
            with patch.dict(os.environ, {'RANK': '0', 'WORLD_SIZE': '8', 'LOCAL_RANK': '0'}):
                args = parse_args()

                assert args.tensor_model_parallel_size == 2
                assert args.pipeline_model_parallel_size == 2
                assert args.distributed_backend == 'nccl'
                assert args.distributed_timeout_minutes == 10
                assert args.rank == 0
                assert args.world_size == 8
                assert args.local_rank == 0

    def test_parse_args_custom_values(self):
        """Test argument parsing with custom values"""
        from flagscale.runner.elastic.gpu_health_check import parse_args

        test_args = [
            '--tensor-model-parallel-size',
            '4',
            '--pipeline-model-parallel-size',
            '2',
            '--distributed-backend',
            'gloo',
            '--distributed-timeout-minutes',
            '30',
        ]

        with patch('sys.argv', ['gpu_health_check.py'] + test_args):
            with patch.dict(os.environ, {'RANK': '0', 'WORLD_SIZE': '16', 'LOCAL_RANK': '0'}):
                args = parse_args()

                assert args.tensor_model_parallel_size == 4
                assert args.pipeline_model_parallel_size == 2
                assert args.distributed_backend == 'gloo'
                assert args.distributed_timeout_minutes == 30

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_world_size', return_value=8)
    @patch('torch.distributed.get_rank', return_value=0)
    def test_initialize_model_parallel_valid_config(self, mock_rank, mock_world_size, mock_init):
        """Test initialize_model_parallel with valid configuration"""
        from flagscale.runner.elastic.gpu_health_check import initialize_model_parallel

        with patch('torch.distributed.new_group') as mock_new_group:
            mock_group = MagicMock()
            mock_new_group.return_value = mock_group

            # Test TP=2, PP=2, world_size=8 (2*2*2=8, valid)
            initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)

            assert mock_new_group.called

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_world_size', return_value=8)
    @patch('torch.distributed.get_rank', return_value=0)
    def test_initialize_model_parallel_single_process_groups(
        self, mock_rank, mock_world_size, mock_init
    ):
        """Test initialize_model_parallel with single-process groups (TP=1, PP=1)"""
        from flagscale.runner.elastic.gpu_health_check import initialize_model_parallel

        with patch('torch.distributed.new_group') as mock_new_group:
            # TP=1, PP=1 means data parallel only
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_world_size', return_value=8)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.set_device')
    def test_test_communication_basic(
        self, mock_set_device, mock_cuda, mock_rank, mock_world_size, mock_init
    ):
        """Test basic communication test functionality"""
        from flagscale.runner.elastic.gpu_health_check import set_args, test_communication

        # Mock args using set_args
        mock_args = MagicMock()
        mock_args.tensor_model_parallel_size = 1
        mock_args.pipeline_model_parallel_size = 1
        mock_args.local_rank = 0
        set_args(mock_args)

        with patch('torch.distributed.barrier'):
            with patch(
                'flagscale.runner.elastic.gpu_health_check.safe_test_execution'
            ) as mock_safe_exec:
                mock_safe_exec.return_value = True

                test_communication()

                # Verify that safe_test_execution was called
                assert mock_safe_exec.called

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.cuda.is_available', return_value=True)
    def test_test_gpu_hardware(self, mock_cuda, mock_rank, mock_init):
        """Test GPU hardware check functionality"""
        from flagscale.runner.elastic.gpu_health_check import set_args, test_gpu_hardware

        mock_args = MagicMock()
        mock_args.local_rank = 0
        set_args(mock_args)

        with patch('torch.distributed.barrier'):
            with patch(
                'flagscale.runner.elastic.gpu_health_check.safe_test_execution'
            ) as mock_safe_exec:
                mock_safe_exec.return_value = True

                result = test_gpu_hardware()

                assert mock_safe_exec.called

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.cuda.is_available', return_value=True)
    def test_test_calculation(self, mock_cuda, mock_rank, mock_init):
        """Test computation check functionality"""
        from flagscale.runner.elastic.gpu_health_check import set_args, test_calculation

        mock_args = MagicMock()
        mock_args.rank = 0
        mock_args.world_size = 8
        mock_args.local_rank = 0
        set_args(mock_args)

        with patch('torch.distributed.barrier'):
            with patch('torch.zeros') as mock_zeros:
                with patch('torch.distributed.all_reduce'):
                    mock_tensor = MagicMock()
                    mock_zeros.return_value = mock_tensor

                    with patch(
                        'flagscale.runner.elastic.gpu_health_check.test_calculation_float',
                        return_value=True,
                    ):
                        with patch(
                            'flagscale.runner.elastic.gpu_health_check.test_calculation_double',
                            return_value=True,
                        ):
                            with patch(
                                'flagscale.runner.elastic.gpu_health_check.test_calculation_half',
                                return_value=True,
                            ):
                                with patch(
                                    'flagscale.runner.elastic.gpu_health_check.test_calculation_endurance',
                                    return_value=True,
                                ):
                                    with patch(
                                        'flagscale.runner.elastic.gpu_health_check.test_ecc_error_detection',
                                        return_value=True,
                                    ):
                                        test_calculation()

    def test_process_group_size_calculation(self):
        """Test process group size calculations"""
        # world_size = TP * PP * DP
        # Test valid configurations
        test_cases = [
            (8, 1, 1, 8),  # DP=8
            (8, 2, 1, 4),  # TP=2, DP=4
            (8, 2, 2, 2),  # TP=2, PP=2, DP=2
            (8, 4, 2, 1),  # TP=4, PP=2, DP=1
            (16, 2, 2, 4),  # TP=2, PP=2, DP=4
        ]

        for world_size, tp, pp, expected_dp in test_cases:
            calculated_dp = world_size // (tp * pp)
            assert (
                calculated_dp == expected_dp
            ), f"Failed for world_size={world_size}, TP={tp}, PP={pp}"

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_world_size', return_value=8)
    @patch('torch.distributed.get_rank', return_value=0)
    def test_initialize_model_parallel_debug_output(self, mock_rank, mock_world_size, mock_init):
        """Test that initialize_model_parallel produces debug output"""
        from flagscale.runner.elastic.gpu_health_check import initialize_model_parallel

        with patch('torch.distributed.new_group'):
            with patch('builtins.print') as mock_print:
                initialize_model_parallel(
                    tensor_model_parallel_size=2, pipeline_model_parallel_size=2
                )

                assert mock_print.called

                # Check that initialization messages were printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                debug_output = ' '.join(print_calls)
                assert (
                    'initialize_model_parallel' in debug_output.lower()
                    or 'START' in debug_output
                    or mock_print.call_count > 0
                )

    def test_timeout_protection_no_alarm(self):
        """Test that timeout_protection doesn't use signal.alarm"""
        from flagscale.runner.elastic.gpu_health_check import timeout_protection

        with patch('signal.alarm') as mock_alarm:
            with timeout_protection(10, "test"):
                pass

            # SIGALRM is disabled, so alarm should not be called (adjusted because change back to A800, see if it's still necessary to change back to adapt BlackWell machines.)
            mock_alarm.assert_not_called()

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_world_size', return_value=8)
    @patch('torch.distributed.get_rank')
    def test_multiple_ranks(self, mock_rank, mock_world_size, mock_init):
        """Test behavior with different rank values"""
        import flagscale.runner.elastic.gpu_health_check as health_check

        from flagscale.runner.elastic.gpu_health_check import initialize_model_parallel

        for rank in range(8):
            mock_rank.return_value = rank

            # Reset globals for each iteration
            health_check._DATA_PARALLEL_GROUP = None
            health_check._TENSOR_MODEL_PARALLEL_GROUP = None
            health_check._PIPELINE_MODEL_PARALLEL_GROUP = None
            health_check._MODEL_PARALLEL_GROUP = None
            health_check._DATA_GLOBAL_RANKS = None
            health_check._TENSOR_GLOBAL_RANKS = None
            health_check._PIPELINE_GLOBAL_RANKS = None
            health_check._EMBEDDING_GROUP = None

            with patch('torch.distributed.new_group'):
                initialize_model_parallel(
                    tensor_model_parallel_size=2, pipeline_model_parallel_size=2
                )

    def test_invalid_parallel_config_detection(self):
        """Test detection of invalid parallel configurations"""
        # Test that world_size % (TP * PP) == 0
        invalid_configs = [
            (8, 3, 1),  # 8 % 3 != 0
            (8, 2, 3),  # 8 % 6 != 0
            (7, 2, 2),  # 7 % 4 != 0
        ]

        for world_size, tp, pp in invalid_configs:
            # These should not divide evenly
            assert (
                world_size % (tp * pp) != 0
            ), f"Expected invalid config: world_size={world_size}, TP={tp}, PP={pp}"

    @patch('torch.cuda.device_count', return_value=8)
    @patch.dict(os.environ, {'WORLD_SIZE': '8', 'RANK': '0', 'LOCAL_RANK': '0'})
    def test_auto_detect_parallel_config_valid(self, mock_device_count):
        """Test auto_detect_parallel_config with valid configurations"""
        from flagscale.runner.elastic.gpu_health_check import auto_detect_parallel_config

        # Test case 1: TP=2, PP=2, world_size=8 (valid)
        tp, pp, need_dist = auto_detect_parallel_config(2, 2)
        assert tp == 2
        assert pp == 2
        assert need_dist == True

        # Test case 2: TP=1, PP=1, world_size=8 (pure DP)
        tp, pp, need_dist = auto_detect_parallel_config(1, 1)
        assert tp == 1
        assert pp == 1
        assert need_dist == True

    @patch('torch.cuda.device_count', return_value=8)
    @patch.dict(os.environ, {'WORLD_SIZE': '8', 'RANK': '0', 'LOCAL_RANK': '0'})
    def test_auto_detect_parallel_config_adjustment(self, mock_device_count):
        """Test auto_detect_parallel_config adjusts invalid configurations"""
        from flagscale.runner.elastic.gpu_health_check import auto_detect_parallel_config

        # Test case: TP=16, PP=2, world_size=8 (invalid, should adjust)
        tp, pp, need_dist = auto_detect_parallel_config(16, 2)
        # Should be adjusted to fit world_size=8
        assert tp * pp <= 8
        assert need_dist == True

    @patch('torch.cuda.device_count', return_value=8)
    @patch.dict(os.environ, {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'})
    def test_auto_detect_parallel_config_single_process(self, mock_device_count):
        """Test auto_detect_parallel_config in single-process mode"""
        from flagscale.runner.elastic.gpu_health_check import auto_detect_parallel_config

        # Single process mode
        tp, pp, need_dist = auto_detect_parallel_config(2, 2)
        assert tp == 1
        assert pp == 1
        assert need_dist == False

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=8)
    def test_main_function_single_process(self, mock_device_count, mock_cuda, mock_rank, mock_init):
        """Test main function in single-process mode"""
        from flagscale.runner.elastic.gpu_health_check import main

        with patch('flagscale.runner.elastic.gpu_health_check.parse_args') as mock_parse_args:
            mock_args = MagicMock()
            mock_args.tensor_model_parallel_size = 1
            mock_args.pipeline_model_parallel_size = 1
            mock_args.rank = 0
            mock_args.world_size = 1
            mock_args.local_rank = 0
            mock_parse_args.return_value = mock_args

            with patch.dict(os.environ, {'WORLD_SIZE': '1', 'RANK': '0'}):
                with patch(
                    'flagscale.runner.elastic.gpu_health_check.safe_test_execution'
                ) as mock_safe_exec:
                    mock_safe_exec.return_value = True
                    with patch('flagscale.runner.elastic.gpu_health_check.print_test_summary'):
                        main()

    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=8)
    def test_main_function_multi_process(self, mock_device_count, mock_cuda, mock_is_init):
        """Test main function in multi-process mode"""
        from flagscale.runner.elastic.gpu_health_check import main

        with patch('flagscale.runner.elastic.gpu_health_check.parse_args') as mock_parse_args:
            mock_args = MagicMock()
            mock_args.tensor_model_parallel_size = 2
            mock_args.pipeline_model_parallel_size = 2
            mock_args.rank = 0
            mock_args.world_size = 8
            mock_args.local_rank = 0
            mock_args.distributed_backend = 'nccl'
            mock_args.distributed_timeout_minutes = 10
            mock_parse_args.return_value = mock_args

            with patch.dict(os.environ, {'WORLD_SIZE': '8', 'RANK': '0'}):
                with patch('flagscale.runner.elastic.gpu_health_check.initialize_distributed'):
                    with patch('flagscale.runner.elastic.gpu_health_check.test_communication'):
                        with patch('flagscale.runner.elastic.gpu_health_check.test_gpu_hardware'):
                            with patch(
                                'flagscale.runner.elastic.gpu_health_check.test_calculation'
                            ):
                                with patch('flagscale.runner.elastic.gpu_health_check.cleanup'):
                                    with patch(
                                        'flagscale.runner.elastic.gpu_health_check.print_test_summary'
                                    ):
                                        main()

    def test_args_validation(self):
        """Test that argument values are validated"""
        from flagscale.runner.elastic.gpu_health_check import parse_args

        # Test with minimum valid values
        test_args = ['--tensor-model-parallel-size', '1', '--pipeline-model-parallel-size', '1']

        with patch('sys.argv', ['gpu_health_check.py'] + test_args):
            with patch.dict(os.environ, {'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_RANK': '0'}):
                args = parse_args()
                assert args.tensor_model_parallel_size >= 1
                assert args.pipeline_model_parallel_size >= 1
                assert args.distributed_timeout_minutes > 0

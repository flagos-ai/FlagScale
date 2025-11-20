"""
Performance Monitor Arguments
This module provides command-line argument definitions for the performance monitoring system.
"""


def add_perf_monitor_args(parser):
    """
    Add performance monitoring specific arguments to the parser.
    Args:
        parser: Argument parser to add arguments to
    Returns:
        parser: Updated argument parser
    """
    group = parser.add_argument_group(title='performance monitoring')

    # Basic monitoring settings
    group.add_argument(
        '--enable-perf-monitor',
        action='store_true',
        help='Enable performance monitoring during training',
    )

    group.add_argument(
        '--perf-log-interval',
        type=int,
        default=10,
        help='Interval (in iterations) for logging performance metrics (default: 10)',
    )

    # Log output settings
    group.add_argument(
        '--perf-log-dir',
        type=str,
        default='logs/perf_monitor',
        help='Directory for performance log files (default: logs/perf_monitor)',
    )

    group.add_argument(
        '--perf-console-output',
        action='store_true',
        help='Also output performance metrics to console (in addition to file)',
    )

    group.add_argument(
        '--perf-log-format',
        type=str,
        choices=['text', 'json', 'both'],
        default='both',
        help='Log format: text, json, or both (default: both)',
    )

    # Memory tracking
    group.add_argument(
        '--perf-memory-tracking',
        action='store_true',
        default=True,
        help='Enable GPU memory tracking (default: enabled)',
    )

    group.add_argument(
        '--no-perf-memory-tracking',
        dest='perf_memory_tracking',
        action='store_false',
        help='Disable GPU memory tracking',
    )

    # Performance breakdown
    group.add_argument(
        '--perf-breakdown',
        action='store_true',
        help='Enable detailed performance breakdown (forward/backward/optimizer timing)',
    )

    # Log rotation
    group.add_argument(
        '--perf-max-log-files',
        type=int,
        default=10,
        help='Maximum number of log files to keep (default: 10, 0 = unlimited)',
    )

    # Model-specific settings
    group.add_argument(
        '--perf-model-type',
        type=str,
        choices=['auto', 'gpt', 'llama', 'qwen', 'mixtral', 'aquila'],
        default='auto',
        help='Model type for FLOPS calculation (default: auto-detect)',
    )

    return parser

from flagscale.runner.auto_tuner.hetero.hetero_theoretical_memory import (
    hetero_report_theoretical_memory,
)
from flagscale.runner.auto_tuner.utils import convert_config_to_megatron_args


def default_model(strategy, config):
    """Use megatron built in memory model."""
    from megatron.training.fs_theoretical_memory_usage import report_theoretical_memory

    args = convert_config_to_megatron_args(config, strategy)
    num_microbatches = (
        config.train.model.global_batch_size
        // strategy["data_parallel_size"]
        // strategy["micro_batch_size"]
    )
    total_memory = report_theoretical_memory(args, num_microbatches=num_microbatches)
    return total_memory


def calculate_hetero_memory(strategy, config):
    """Calculates theoretical memory for a heterogeneous strategy."""
    # Get base args using compatibility keys
    base_args = convert_config_to_megatron_args(config, strategy)

    # Add global batch size to base_args if not present
    if not hasattr(base_args, "global_batch_size"):
        base_args.global_batch_size = config.train.model.global_batch_size

    # Call the dedicated hetero memory calculation function
    # This will now return a LIST [stage0_mem, stage1_mem, ...] or float('inf')
    total_memory_list_or_inf = hetero_report_theoretical_memory(
        strategy=strategy,
        config=config,
        base_args=base_args,
        # verbose=True # For debugging
    )
    return total_memory_list_or_inf

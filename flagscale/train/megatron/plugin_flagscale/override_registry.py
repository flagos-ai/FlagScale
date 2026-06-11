"""
Centralized override registry for FlagScale training
"""

from megatron.plugin.decorators import register


# =============================================================================
# DistSignalHandler - get_device
# =============================================================================
register(
    target="megatron.training.dist_signal_handler.get_device",
    impl="megatron.plugin_flagscale.dist_signal_handler.get_device",
)

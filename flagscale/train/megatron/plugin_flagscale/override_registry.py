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

register(
    target="megatron.training.dist_signal_handler.get_device",
    impl="megatron.plugin_flagscale.npu_plugin.get_device",
    vendor="npu",
)

register(
    target="megatron.training.utils.get_device_arch_version",
    impl="megatron.plugin_flagscale.npu_plugin.get_device_arch_version",
    vendor="npu",
)

register(
    target="megatron.training.initialize.initialize._compile_dependencies",
    impl="megatron.plugin_flagscale.npu_plugin._compile_dependencies",
    vendor="npu",
)

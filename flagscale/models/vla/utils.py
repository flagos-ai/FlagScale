def get_vlm_config(vlm_config) -> dict:
    """
    Extract common fields from any VLM config, handling structural differences.

    Args:
        vlm_config: HF config object (may have hidden_size directly or via text_config).
    Returns:
        dict with 'hidden_size' and 'num_hidden_layers'.
    """
    return {
        "hidden_size": _get_hidden_size(vlm_config),
        "num_hidden_layers": _get_num_layers(vlm_config),
    }


def _get_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config"):
        return config.text_config.hidden_size
    raise ValueError(f"Cannot determine hidden_size from config: {type(config)}")


def _get_num_layers(config) -> int:
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "text_config"):
        return config.text_config.num_hidden_layers
    raise ValueError(f"Cannot determine num_hidden_layers from config: {type(config)}")

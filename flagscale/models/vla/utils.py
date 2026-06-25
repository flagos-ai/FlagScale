from flagscale.models.configs.types import FeatureType, PolicyFeature


def reorder_visual_input_features(
    input_features: dict[str, PolicyFeature],
    preferred_image_order: list[str] | None = None,
) -> dict[str, PolicyFeature]:
    """Reorder visual features while leaving other inputs untouched."""

    if not preferred_image_order:
        return dict(input_features)

    visual_keys = [key for key, ft in input_features.items() if ft.type == FeatureType.VISUAL]
    ordered_visual_keys = [key for key in preferred_image_order if key in visual_keys]
    # Keep dataset-defined order for visual keys that are not listed in the recipe.
    ordered_visual_keys.extend(key for key in visual_keys if key not in ordered_visual_keys)

    reordered = {key: ft for key, ft in input_features.items() if ft.type != FeatureType.VISUAL}
    for key in ordered_visual_keys:
        reordered[key] = input_features[key]
    return reordered


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

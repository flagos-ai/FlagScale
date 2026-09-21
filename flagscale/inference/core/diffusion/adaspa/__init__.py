from .adaspa_handler import AdaSpaHandler
from .processor import WanAdaSpaAttnProcessor

# Optional explicit registration when the symbol exists in current diffusers.
try:
    from diffusers.models.attention_processor import WanAttnProcessor2_0

    AdaSpaHandler.register_processor(WanAttnProcessor2_0, WanAdaSpaAttnProcessor)
except Exception:
    pass

__all__ = ["AdaSpaHandler", "WanAdaSpaAttnProcessor"]

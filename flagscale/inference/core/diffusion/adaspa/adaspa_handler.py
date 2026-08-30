from typing import Dict, Type, Optional, Any, Tuple
import torch
from torch import nn
from diffusers.models.attention import Attention

ADASPA_PROCESSOR = None

def get_model_name():
    global ADASPA_PROCESSOR
    name_dict = {
        "HunyuanVideoAdaSpaAttnProcessor" : "HunyuanVideo",
        "CogVideoXAdaSpaAttnProcessor" : "CogVideoX",
        "WanAdaSpaAttnProcessor": "Wan2.1",
        "Wan2.1": "Wan2.1",
    }
    if ADASPA_PROCESSOR in name_dict:
        return name_dict[ADASPA_PROCESSOR]
    # Fallback for unknown processor names: keep behavior safe for new model adapters.
    return "Wan2.1"

class AdaSpaHandler(nn.Module):
    """
    A handler class that provides a generic way to replace attention processors with their sparse versions.
    It maintains a registry of original attention processors and their corresponding sparse versions.
    """
    _processor_registry: Dict[Type, Type] = {}
    adaspa_processor = None

    @classmethod
    def register_processor(cls, original_processor: Type, sparse_processor: Type) -> None:
        """
        Register a mapping between an original attention processor and its sparse version.
        
        Args:
            original_processor: The original attention processor class
            sparse_processor: The corresponding sparse attention processor class
        """
        cls._processor_registry[original_processor] = sparse_processor

    def __init__(
        self,
        model: nn.Module,
        **kwargs: Any
    ):
        """
        Initialize the AdaSpaHandler with a model and optional shape information.
        
        Args:
            model: The model whose attention processors will be replaced
            **kwargs: Additional arguments to pass to the sparse processor constructor
        """
        super().__init__()
        self.model = model
        self.kwargs = kwargs
        self._original_processors = {}
        self._sparse_processors = {}
        self._replace_processors()

    def _replace_processors(self) -> None:
        """Replace attention processors in Attention modules."""
        
        def _replace_module(module: nn.Module) -> None:
            if isinstance(module, Attention):
                if hasattr(module, 'processor'):
                    original_processor = module.processor
                    original_type = type(original_processor)

                    if original_type in self._processor_registry:
                        sparse_type = self._processor_registry[original_type]

                        global ADASPA_PROCESSOR
                        ADASPA_PROCESSOR = sparse_type.__name__

                        # Store original processor
                        key = f"{module.__class__.__name__}.processor"
                        self._original_processors[key] = original_processor

                        # Create sparse processor wrapper around original processor
                        sparse_processor = sparse_type()
                        self._sparse_processors[key] = sparse_processor
                        module.processor = sparse_processor

            for child in module.children():
                _replace_module(child)

        _replace_module(self.model)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

# Register Wan processor mapping for the in-tree AdaSpa package.
from .processor import WanAdaSpaAttnProcessor

# Optional Wan registration (depends on diffusers version naming).
try:
    from diffusers.models.attention_processor import WanAttnProcessor2_0

    AdaSpaHandler.register_processor(WanAttnProcessor2_0, WanAdaSpaAttnProcessor)
except Exception:
    pass

import torch
import contextlib
from typing import Dict, List

from utils import HELPER_SUPPORT_MODEL_LIST, HELPER_SUPPORT_MODEL_TYPES

from sola_neuron_idx.hook import (
    add_training_hook,
    remove_training_hook
)


class Helper(contextlib.ContextDecorator):
    def __init__(self, model: HELPER_SUPPORT_MODEL_TYPES, compute_type, **kwargs):
        self.model = model
        self.device = model.device
        self.compute_type = compute_type
        self.intermediate_size = kwargs["intermediate_size"]
        self.training_data: Dict[str, Dict[str, List[torch.Tensor]]] = {}

        if not isinstance(model, HELPER_SUPPORT_MODEL_LIST):
            raise NotImplementedError("Unsupported model")

    def __enter__(self):
        self.model_last_layer = add_training_hook(self.model, self.training_data, self.intermediate_size)

    def __exit__(self, exc_type, exc_val, exc_tb):
        remove_training_hook(self.model)
        
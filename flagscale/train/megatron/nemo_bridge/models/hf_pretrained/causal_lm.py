#!/usr/bin/env python3
# Copyright (c) 2025, BAAI. All rights reserved.

import sys

from pathlib import Path
from typing import Dict, Generic, List, Optional, TypeVar, Union

import torch

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM as OriginalPreTrainedCausalLM

class PreTrainedCausalLM(OriginalPreTrainedCausalLM):
    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        self.device = "cpu"
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=self.device,  
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        #if hasattr(self, '_model') and self._model is not None:
        #    self._model.to("cpu")

    def save_artifacts(self, save_directory: Union[str, Path]):
        """
        Saves all loaded, generic artifacts that have a `save_pretrained` method
        to the specified directory. Note: This does not save the `model` attribute.

        If the model was loaded with trust_remote_code=True, this method will also
        attempt to preserve any custom modeling files to ensure the saved model
        can be loaded properly.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        _ = getattr(self, "config")  # trigger lazy loading of config
        if hasattr(self, "_config") and self._config is not None:
            self._config.save_pretrained(save_path)

        for name in self.OPTIONAL_ARTIFACTS:
            artifact = getattr(self, name, None)
            if artifact is not None and hasattr(artifact, "save_pretrained"):
                artifact.save_pretrained(save_path)

        # Preserve custom modeling files if trust_remote_code was used
        if hasattr(self, 'trust_remote_code') and self.trust_remote_code:
            # Try original source path first, then fallback to model_name_or_path
            source_paths = []
            if hasattr(self, '_original_source_path') and self._original_source_path:
                source_paths.append(self._original_source_path)
            if hasattr(self, 'model_name_or_path') and self.model_name_or_path:
                source_paths.append(self.model_name_or_path)

            for source_path in source_paths:
                copied_files = self._copy_custom_modeling_files(source_path, save_path)
                if copied_files:
                    # Successfully copied files, no need to try other paths
                    break

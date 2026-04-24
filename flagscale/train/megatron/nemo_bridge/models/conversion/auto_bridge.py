# Copyright (c) 2025, BAAI. All rights reserved.

from megatron.bridge import AutoBridge as OriginalAutoBridge
import transformers
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig

from megatron.core.transformer.module import MegatronModule
from megatron.nemo_bridge.models.conversion import model_bridge
from megatron.nemo_bridge.models.conversion.model_bridge import MegatronModelBridge

from megatron.nemo_bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry
from megatron.bridge.models.conversion.utils import get_causal_lm_class_via_auto_map

from typing import TypeVar, Union
from pathlib import Path
MegatronModelT = TypeVar("MegatronModelT", bound=MegatronModule)


class AutoBridge(OriginalAutoBridge):

    def __init__(self, hf_pretrained: PreTrainedCausalLM | PretrainedConfig):
        if not isinstance(hf_pretrained, (PreTrainedCausalLM, PretrainedConfig)):
            raise ValueError("hf_pretrained must be a PreTrainedCausalLM or PretrainedConfig instance")
        self.hf_pretrained: PreTrainedCausalLM | PretrainedConfig = hf_pretrained
        super().__init__(hf_pretrained)

    @classmethod
    def from_hf_pretrained(cls, path: Union[str, Path], **kwargs) -> "AutoBridge":
        """
        Load an AutoBridge from a pretrained model, automatically detecting the model type.
        """
        # First load just the config to check architecture support
        # Use thread-safe config loading to prevent race conditions
        config = safe_load_config_with_retry(path, trust_remote_code=kwargs.get("trust_remote_code", False))

        cls._validate_config(config, str(path))

        try:
            return cls(PreTrainedCausalLM.from_pretrained(path, **kwargs))
        except Exception as e:
            raise ValueError(f"Failed to load model with AutoBridge: {e}") from e

    @classmethod
    def _validate_config(cls, config: PretrainedConfig, path: str | None = None) -> None:
        # Check if this is a causal LM model
        if not cls.supports(config):
            architectures = getattr(config, "architectures", [])
            raise ValueError(
                f"\n�~\~W Model architecture not supported by AutoBridge\n\n"
                f"Model: {path}\n"
                f"Architectures: {architectures}\n\n"
                f"AutoBridge only supports models with architectures ending in 'ForCausalLM' or"
                f"'ForConditionalGeneration' or 'NemotronH_Nano_VL_V2'.\n"
                f"Found architectures that don't match this pattern.\n\n"
                f"If this is a different model type (e.g., Vision, Sequence-to-Sequence),\n"
                f"you may need to use a different bridge class."
            )

        # Check if we have an implementation for this specific architecture
        architecture = None
        for arch_name in config.architectures:
            if arch_name.endswith(
                ("ForCausalLM", "ForConditionalGeneration", "NemotronH_Nano_VL_V2")
            ):
                architecture = arch_name
                break

        if architecture:
            # Try auto_map first
            arch_class = (
                get_causal_lm_class_via_auto_map(model_name_or_path=path, config=config)
                if path
                else None
            )
            if arch_class is not None:
                # For auto_map models, use class-name string
                arch_key = getattr(arch_class, "__name__", str(arch_class))
            else:
                try:
                    arch_class = getattr(transformers, architecture)
                    arch_key = arch_class
                except AttributeError:
                    # Fall back to name-based registration
                    arch_key = architecture
            
            # Test if we have a registered implementation (type or class-name string)
            has_implementation = False
            if hasattr(model_bridge.get_model_bridge, "_exact_types"):
                registry = model_bridge.get_model_bridge._exact_types
                if isinstance(arch_key, str):
                    has_implementation = arch_key in registry
                else:
                    has_implementation = (arch_key in registry) or (
                        getattr(arch_key, "__name__", None) in registry
                    )

                if not has_implementation:
                    raise ValueError(
                        f"\n�~\~W Model architecture '{architecture}' is not yet supported\n\n"
                        f"Model: {path}\n"
                        f"Architecture: {architecture}\n\n"
                        + f"\n\nTo add support for {architecture}, you need to:\n"
                        f"1. Create a new bridge class that inherits from MegatronModelBridge\n"
                        f"2. Implement the required methods (provider_bridge, mapping_registry)\n"
                        f"3. Register it with @MegatronModelBridge.register_bridge decorator\n\n"
                        f"Example implementation:\n"
                        f"  from megatron.nemo_bridge.models.conversion.model_bridge import MegatronModelBridge\n"
                        f"  from transformers import {architecture}\n"
                        f"  from megatron.core.models.gpt import GPTModel\n\n"
                        f"  @MegatronModelBridge.register_bridge(source={architecture}, target=GPTModel)\n"
                        f"  class Megatron{architecture.replace('ForCausalLM', '')}Bridge(MegatronModelBridge):\n"
                        f"      def provider_bridge(self, hf_pretrained):\n"
                        f"          # Return a ModelProvider instance\n"
                        f"          ...\n\n"
                        f"      def mapping_registry(self):\n"
                        f"          # Return a MegatronMappingRegistry with weight mappings\n"
                        f"          ...\n\n"
                        f"For reference implementations, see:\n"
                        f"  �~@� src/megatron/bridge/models/llama/llama_bridge.py\n"
                        f"  �~@� src/megatron/bridge/models/qwen/qwen_2_causal_bridge.py"
                    ) from None


    @classmethod
    def from_hf_config(cls, config: PretrainedConfig, modeling_path=None, model_name= None) -> "AutoBridge":
        
        cls._validate_config(config)
        model = PreTrainedCausalLM()
        model.config = config
        import torch
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        
        hf_model = None 
        if modeling_path :
            import os,sys
            import importlib.util
            abs_model_path = os.path.abspath(modeling_path)
            model_dir = os.path.dirname(abs_model_path)
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            module_name = "dynamic_model_module"
            try:
                spec = importlib.util.spec_from_file_location(module_name, abs_model_path)
                assert spec is not None
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                model_class = getattr(module, model_name)
                with init_empty_weights():
                    hf_model = model_class._from_config(model.config)
                for name, param in hf_model.named_parameters():
                    set_module_tensor_to_device(
                        hf_model, name, "cpu", torch.empty(*param.size(), dtype=model.config.torch_dtype)
                    )
            except Exception as e:
                print(f"import module error: {e}")

        elif not modeling_path and config :
            with init_empty_weights():
                hf_model = AutoModelForCausalLM.from_config(model.config)

            for name, param in hf_model.named_parameters():
                set_module_tensor_to_device(
                    hf_model, name, "cpu", torch.empty(*param.size(), dtype=model.config.torch_dtype)
                )
        else: 
            raise ValueError("Need one args, model_path or config, to build HF model.")
        model.model = hf_model
        return cls(model)

    def load_hf_weights(
        self, model: list[MegatronModelT], hf_path: str | Path | None = None
    ) -> None:
        if hf_path is None:
            if not isinstance(self.hf_pretrained, PreTrainedCausalLM):
                raise ValueError(
                    "hf_path is required when hf_pretrained is not a PreTrainedCausalLM instance"
                )
            pre_trained = self.hf_pretrained
        else:
            pre_trained = PreTrainedCausalLM.from_pretrained(hf_path)
            # Preserve trust_remote_code setting from the original bridge instance
            trust_remote_code = getattr(self.hf_pretrained, 'trust_remote_code', False)
            pre_trained = PreTrainedCausalLM.from_pretrained(
                hf_path, trust_remote_code=trust_remote_code
            )
        # self._model_bridge.load_weights_hf_to_megatron(model, pre_trained)
        self._model_bridge.load_weights_hf_to_megatron(pre_trained, model)

        return model

    def save_hf_weights(
        self,
        model: list[MegatronModelT],
        path: str | Path,
        show_progress: bool = True,
        strict: bool = True,
    ) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        dispatch_instance = (self._causal_lm_architecture, self._get_model_instance(model))
        generator = model_bridge.stream_weights_megatron_to_hf(
            dispatch_instance, model, self.hf_pretrained, cpu=True, show_progress=show_progress
        )
        source = SafeTensorsStateSource(path)
        # Check if the state source is SafeTensorsStateSource for streaming save.
        if (
            hasattr(self.hf_pretrained, "state")
            and hasattr(self.hf_pretrained.state, "source")
            # and isinstance(self.hf_pretrained.state.source, SafeTensorsStateSource)
        ):
            # self.hf_pretrained.state.source.save_generator(generator, path, strict=strict)
            source.save_generator(generator, path, strict=strict)
        else:
            raise ValueError(
                "The state source is not a SafeTensorsStateSource, cannot save in streaming mode."
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @property
    def _model_bridge(self) -> "MegatronModelBridge":
        return model_bridge.get_model_bridge(self._causal_lm_architecture)

    def convert_mg2hf_config(margs, save_path, model_type):
        assert model_type is not None, "model_type is None"
        if hasattr(model_bridge.get_model_bridge, "_exact_types"):
            registry = model_bridge.get_model_bridge._exact_types
            if isinstance(model_type, str):
                has_implementation = model_type in registry
            else:
                has_implementation = (model_type in registry) or (getattr(model_type, "__name__", None) in registry)

            if not has_implementation:
                raise ValueError(f"\n✗ Model architecture '{model_type}' is not yet supported\n\n")
        model_bridge_class = model_bridge.get_model_bridge(model_type)
        return model_bridge_class.save_args_mg2hf(margs, save_path)

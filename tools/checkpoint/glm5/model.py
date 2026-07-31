import time

import torch

from megatron.core.enums import ModelType

model_type = ModelType.encoder_or_decoder  # Megatron's model_type


def _register_glm_moe_dsa():
    """Register GlmMoeDsaConfig/Model with transformers AutoConfig/AutoModel.

    The glm_moe_dsa model type is not in the standard transformers registry.
    We register a thin subclass of DeepseekV2 so that AutoModelForCausalLM
    can load the checkpoint.
    """
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        DeepseekV2Config,
        DeepseekV2ForCausalLM,
    )

    # Only register once
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    if "glm_moe_dsa" in CONFIG_MAPPING_NAMES:
        return

    # Define config inline to avoid importing vllm_fl (which triggers heavy vllm imports)
    class GlmMoeDsaConfig(DeepseekV2Config):
        model_type = "glm_moe_dsa"

        def __init__(
            self,
            index_topk=2048,
            index_n_heads=32,
            index_head_dim=128,
            indexer_rope_interleave=True,
            num_nextn_predict_layers=1,
            moe_layer_freq=1,
            scoring_func="sigmoid",
            ep_size=1,
            head_dim=None,
            rope_parameters=None,
            dtype="bfloat16",
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.index_topk = index_topk
            self.index_n_heads = index_n_heads
            self.index_head_dim = index_head_dim
            self.indexer_rope_interleave = indexer_rope_interleave
            self.num_nextn_predict_layers = num_nextn_predict_layers
            self.moe_layer_freq = moe_layer_freq
            self.scoring_func = scoring_func
            self.ep_size = ep_size
            if head_dim is not None:
                self.head_dim = head_dim
            if rope_parameters is not None:
                self.rope_theta = rope_parameters.get(
                    "rope_theta", getattr(self, "rope_theta", 10000.0)
                )
            self.dtype = dtype

    AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig)

    class _GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
        config_class = GlmMoeDsaConfig

    AutoModelForCausalLM.register(GlmMoeDsaConfig, _GlmMoeDsaForCausalLM)


def _build_mock_hf_model(model_path, dtype):
    """Build a mock model with proper attribute hierarchy from safetensors.

    The DeepseekV2ForCausalLM class does not have DSA indexer or MTP layers,
    so we load weights directly from safetensors into a simple module structure
    that mirrors the expected HF attribute paths used by ckpt.py.
    """
    import os

    import torch.nn as nn
    from safetensors.torch import load_file

    # Load weights
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    state_dict = {}
    for sf in safetensors_files:
        state_dict.update(load_file(os.path.join(model_path, sf)))

    # Build a module tree that mirrors the state_dict keys
    # e.g. "model.layers.0.self_attn.indexer.wq_b.weight" -> model.layers[0].self_attn.indexer.wq_b.weight
    class _Leaf(nn.Module):
        """Holds a single weight or bias tensor as a buffer."""

        def __init__(self):
            super().__init__()

    class _Container(nn.Module):
        """A container that supports both attribute and index access."""

        def __init__(self):
            super().__init__()
            self._indexed_children = {}

        def __getitem__(self, idx):
            return self._indexed_children[idx]

    # Parse and build the tree
    root = _Container()

    for full_key, tensor in state_dict.items():
        parts = full_key.split(".")
        node = root
        # Navigate/create the tree, except the last part (weight/bias)
        for i, part in enumerate(parts[:-1]):
            # Check if next level is an integer index (list-like)
            if part.isdigit():
                idx = int(part)
                if not hasattr(node, "_indexed_children"):
                    node._indexed_children = {}
                if idx not in node._indexed_children:
                    child = _Container()
                    node._indexed_children[idx] = child
                    # Also register as a named module for traversal
                    node.add_module(f"_{idx}", child)
                node = node._indexed_children[idx]
            else:
                if not hasattr(node, part):
                    child = _Container()
                    setattr(node, part, child)
                    if isinstance(node, nn.Module):
                        node.add_module(part, child)
                node = getattr(node, part)

        # Set the leaf tensor (weight or bias)
        leaf_name = parts[-1]
        # Store as a Parameter-like object with .data attribute
        param = nn.Parameter(tensor.to(dtype), requires_grad=False)
        if isinstance(node, nn.Module):
            node.register_parameter(leaf_name, param)
        else:
            setattr(node, leaf_name, param)

    return root


def get_hf_model(dtype, model_path=None, config=None):
    """Build a HuggingFace GlmMoeDsaForCausalLM model."""
    from transformers import AutoConfig, AutoModelForCausalLM

    s_time = time.time()
    if model_path and not config:
        # Try standard loading first; fall back to mock model if model_type not registered
        try:
            _register_glm_moe_dsa()
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="cpu", trust_remote_code=True, torch_dtype=dtype
            )
            # Check if DSA indexer weights were loaded (they won't be for DeepseekV2)
            import json
            import os

            cfg = json.load(open(os.path.join(model_path, "config.json")))
            if cfg.get("model_type") == "glm_moe_dsa":
                # Verify indexer is present; if not, use mock model
                try:
                    _ = model.model.layers[0].self_attn.indexer.wq_b.weight
                except AttributeError:
                    print(
                        "> DeepseekV2 model lacks DSA indexer layers, using direct weight loading"
                    )
                    model = _build_mock_hf_model(model_path, dtype)
        except (ValueError, ImportError):
            # Model type not recognized at all, use mock
            model = _build_mock_hf_model(model_path, dtype)
    elif not model_path and config:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device

        if isinstance(config, dict):
            config = AutoConfig.for_model(**config)

        _register_glm_moe_dsa()
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, torch_dtype=dtype
            )

        for name, param in model.named_parameters():
            set_module_tensor_to_device(model, name, "cpu", torch.empty(*param.size(), dtype=dtype))
    else:
        raise ValueError("Need one of model_path or config to build HF model.")
    print("> build huggingface model elapsed time:", time.time() - s_time)
    return model


def get_mg_model(dtype, pre_process, post_process):
    """Build a Megatron GPTModel for GLM5."""
    from flagscale.train.megatron.gpt_builders import gpt_builder
    from flagscale.train.megatron.model_provider import model_provider

    s_time = time.time()
    model = model_provider(gpt_builder, pre_process, post_process).to(dtype)
    print("> build megatron model elapsed time:", time.time() - s_time)
    return model

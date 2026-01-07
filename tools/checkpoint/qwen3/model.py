import time

import torch

from megatron.core.enums import ModelType

model_type = ModelType.encoder_or_decoder  # Megatron's model_type


def get_hf_model(dtype, model_path=None, config=None):
    from .modeling_hf.modeling_qwen3 import Qwen3ForCausalLM

    s_time = time.time()
    if config is None:
        model = Qwen3ForCausalLM.from_pretrained(
            model_path, device_map="cpu", trust_remote_code=True, torch_dtype=dtype
        )
    elif model_path is None:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device

        with init_empty_weights():
            model = Qwen3ForCausalLM._from_config(config, torch_dtype=dtype)
        for name, param in model.named_parameters():
            set_module_tensor_to_device(model, name, "cpu", torch.empty(*param.size(), dtype=dtype))
    else:
        raise ValueError("Build HF model must have path or config model_path.")
    print("> build huggingface model elapsed time:", time.time() - s_time)
    return model


def get_mg_model(dtype, pre_process, post_process):
    from flagscale.train.megatron.gpt_builders import gpt_builder
    from flagscale.train.megatron.model_provider import model_provider

    s_time = time.time()
    model = model_provider(gpt_builder, pre_process, post_process).to(dtype)
    print("> build megatron model elapsed time:", time.time() - s_time)
    return model

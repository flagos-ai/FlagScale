import time

from megatron.core.enums import ModelType

model_type = ModelType.encoder_or_decoder  # Megatron's model_type


def get_hf_model(dtype, model_path=None, config=None):
    try:
        from .llama_model.modeling_llama import LlamaForCausalLM
    except ImportError:
        print(
            "Failed to import LlamaForCausalLM from modeling_llama, please add the model of huggingface style."
        )
    s_time = time.time()
    if model_path and not config:
        model = LlamaForCausalLM.from_pretrained(
            model_path, device_map="cpu", trust_remote_code=True, torch_dtype=dtype
        )
    elif not model_path and config:
        import torch

        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device

        with init_empty_weights():
            model = LlamaForCausalLM._from_config(config=config, torch_dtype=dtype)
        for name, param in model.named_parameters():
            set_module_tensor_to_device(model, name, "cpu", torch.empty(*param.size(), dtype=dtype))
    else:
        raise ValueError("Need one args, model_path or config, to build HF model.")
    print("> build huggingface model elapsed time:", time.time() - s_time)
    return model


def get_mg_model(dtype, pre_process, post_process):
    from flagscale.train.megatron.gpt_builders import gpt_builder
    from flagscale.train.megatron.model_provider import model_provider

    s_time = time.time()
    model = model_provider(gpt_builder, pre_process, post_process).to(dtype)
    print("> build megatron model elapsed time:", time.time() - s_time)
    return model

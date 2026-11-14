import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDecoderLayer, LlamaAttention
from transformers.cache_utils import Cache, DynamicCache

import warnings
import functools
import math
import logging
logger = logging.getLogger(__name__)

from typing import List, Dict, Tuple, Optional

from utils import (
    HelperState,
    set_helper_state,
    HELPER_SUPPORT_MODEL_LIST,
    HELPER_SUPPORT_MODEL_TYPES
)


_HELPER_HOOK_KEY = "HelperHook"

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    
def llama_mlp(module: torch.nn.Module, x: torch.Tensor):
    if module.config.pretraining_tp > 1:
        slice = module.intermediate_size // module.config.pretraining_tp
        gate_proj_slices = module.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = module.up_proj.weight.split(slice, dim=0)
        down_proj_slices = module.down_proj.weight.split(slice, dim=1)

        gate_proj = torch.cat(
            [F.linear(x, gate_proj_slices[i]) for i in range(module.config.pretraining_tp)], dim=-1
        )
        up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(module.config.pretraining_tp)], dim=-1)

        intermediate_states = (module.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(module.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:
        h_gate = module.act_fn(module.gate_proj(x))
        h_up = module.up_proj(x)
        intermediate_states = h_gate * h_up
        down_proj = module.down_proj(intermediate_states)

    return down_proj, intermediate_states, h_gate, h_up
    
    
def llama_self_attn(module: torch.nn.Module, 
                    hidden_states: torch.Tensor, 
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_value: Optional[Tuple[torch.Tensor]] = None,
                    output_attentions: Optional[bool] = False,
                    use_cache: Optional[bool] = False,
                    **kwargs
                    ):
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    # print("collect hidden states shape: ", {hidden_states.shape})
    bsz, q_len, _ = hidden_states.size()

    if module.config.pretraining_tp > 1:
        key_value_slicing = (module.num_key_value_heads * module.head_dim) // module.config.pretraining_tp
        query_slices = module.q_proj.weight.split(
            (module.num_heads * module.head_dim) // module.config.pretraining_tp, dim=0
        )
        key_slices = module.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = module.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(module.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(module.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(module.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)

    q_norm = torch.norm(query_states, p=2, dim=0)
    k_norm = torch.norm(key_states, p=2, dim=0)

    query_states = query_states.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if module.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {module.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, module.layer_idx)
    cos, sin = module.rotary_emb(value_states, seq_len=kv_seq_len)
    if cos.device != position_ids.device:
        position_ids = position_ids.to(cos.device)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, module.layer_idx, cache_kwargs)
        key_states, value_states = past_key_value.update_get(key_states, value_states, module.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)

    if attn_weights.size() != (bsz, module.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, module.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        if attn_weights.device != attention_mask.device:
            attn_weights = attn_weights.to(attention_mask.device)
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
    if attn_weights.device != value_states.device:
        attn_weights = attn_weights.to(value_states.device)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, module.num_heads, q_len, module.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, module.num_heads, q_len, module.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    
    head_norms = torch.norm(attn_output, dim = -1)

    attn_output = attn_output.reshape(bsz, q_len, module.hidden_size)

    if module.config.pretraining_tp > 1:
        attn_output = attn_output.split(module.hidden_size // module.config.pretraining_tp, dim=2)
        o_proj_slices = module.o_proj.weight.split(module.hidden_size // module.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(module.config.pretraining_tp)])
    else:
        attn_output = module.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return head_norms, attn_output, attn_weights, past_key_value, q_norm, k_norm
    

def pre_decoder_forward_hook_dejavu_collect(layer_idx: int,
                                        model_type: str,
                                        dest: Dict[str, Dict[str, List[torch.Tensor]]],
                                        module: torch.nn.Module,
                                        inp: Tuple,
                                        *args,
                                        **kwargs):
    x = inp[0]

    residual = x
    hidden_states = module.input_layernorm(x)

    use_legacy_cache = not isinstance(args[0]["past_key_value"], Cache) and args[0]["past_key_value"] is not None
    if use_legacy_cache:
        args[0]["past_key_value"] = DynamicCache.from_legacy_cache(args[0]["past_key_value"])
    
    _, hidden_states, _, _, _, _ = llama_self_attn(module.self_attn, 
                                              hidden_states,
                                              args[0]["attention_mask"], 
                                              args[0]["position_ids"], 
                                              args[0]["past_key_value"], 
                                              args[0]["output_attentions"], 
                                              args[0]["use_cache"])
    
    if hidden_states.device != residual.device:
        hidden_states = hidden_states.to(residual.device)
    hidden_states = residual + hidden_states
    
    # Fully Connected
    residual = hidden_states
    hidden_states = module.post_attention_layernorm(hidden_states)
    
    hidden_states, intermediate_states, _, _  = llama_mlp(module.mlp, hidden_states)
    
    ##### the output norm of different neurons #####
    if hidden_states.shape[1] > 1:
        neurons_dict_cur_layer = getattr(module, f'neurons_dict_{layer_idx}')
        mlp_inter_norm = torch.norm(intermediate_states, p=2, dim=0)
        if mlp_inter_norm.device != neurons_dict_cur_layer['norm'].device:
            mlp_inter_norm = mlp_inter_norm.to(neurons_dict_cur_layer['norm'].device)
        neurons_dict_cur_layer['norm'] += mlp_inter_norm.sum(dim=0)
        neurons_dict_cur_layer['token_num'] += mlp_inter_norm.shape[0]
    

def add_training_hook_to_llama(model: LlamaPreTrainedModel,
                               dest: Dict[str, Dict[str, List[torch.Tensor]]],
                               intermediate_size: int) -> int:
    set_helper_state(model, HelperState.Collecting)
    hooks = []
    last_layer = 0

    for name, module in model.named_modules():
        if not isinstance(module, (LlamaDecoderLayer, LlamaAttention, LlamaMLP)):
            continue

        if isinstance(module, LlamaDecoderLayer):
            layer_idx = int(name.split(".")[-1])
        else:
            layer_idx = int(name.split(".")[-2])

        last_layer = max(layer_idx, last_layer)

        if isinstance(module, LlamaDecoderLayer):
            setattr(module, f"neurons_dict_{layer_idx}", {'norm': torch.zeros(intermediate_size).to(model.device), 'token_num': 0})
            handle_dejavu_collect = module.register_forward_pre_hook(
                functools.partial(
                    pre_decoder_forward_hook_dejavu_collect,
                    layer_idx,
                    "llama",
                    dest,
                ),
                with_kwargs=True
            )
            hooks.append(handle_dejavu_collect)

    setattr(model, _HELPER_HOOK_KEY, hooks)
    return last_layer


def add_training_hook(model: HELPER_SUPPORT_MODEL_TYPES,
                      dest: Dict[str, Dict[str, List[torch.Tensor]]], 
                      intermediate_size: int) -> int:
    if isinstance(model, LlamaPreTrainedModel):
        return add_training_hook_to_llama(model, dest, intermediate_size)
    else:
        raise NotImplementedError(f"Only support {HELPER_SUPPORT_MODEL_LIST}.")


def remove_training_hook(model: HELPER_SUPPORT_MODEL_TYPES):
    hooks = getattr(model, _HELPER_HOOK_KEY)
    for handle in hooks:
        handle.remove()

    setattr(model, _HELPER_HOOK_KEY, None)

import warnings
import functools
import math
import logging
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from transformers import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention

from utils import (
    HelperState,
    HelperCollectState,
    set_helper_state,
    HELPER_SUPPORT_MODEL_LIST,
    HELPER_SUPPORT_MODEL_TYPES
)

logger = logging.getLogger(__name__)

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
    
def add_training_hook_to_llama(model: LlamaPreTrainedModel,
                               dest: Dict[str, Dict[str, List[torch.Tensor]]],
                               intermediate_size: int, 
                               hidden_size: int) -> int:
    set_helper_state(model, HelperState.Collecting)
    hooks = []
    last_layer = 0

    def forward_hook_get_XXT(layer_idx, name, module, inp, out):
        inp = inp[0].detach().float()
        if inp.shape[1] > 1:
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0).cpu()
            
            raw_scaling_diag_matrix = getattr(module, f'raw_scaling_diag_matrix_{layer_idx}')
            raw_scaling_diag_matrix += adds_sum
            
            inp = adds = adds_sum = out = None
            del inp, adds, adds_sum, out
            torch.cuda.empty_cache()
    
    for name, module in model.named_modules():
        suffix = name.split(".")[-1]
        if suffix not in ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "o_proj", "v_proj"]:
            continue
        layer_idx = int(name.split(".")[-3])
        if suffix == "down_proj":
            setattr(module, f"raw_scaling_diag_matrix_{layer_idx}", torch.zeros(intermediate_size, intermediate_size))
        else:
            setattr(module, f"raw_scaling_diag_matrix_{layer_idx}", torch.zeros(hidden_size, hidden_size))
        handle_pre_forward_collect_hook = module.register_forward_hook(
                functools.partial(
                    forward_hook_get_XXT,
                    layer_idx,
                    name
                )
            )
        hooks.append(handle_pre_forward_collect_hook)
    
    setattr(model, _HELPER_HOOK_KEY, hooks)
    return last_layer


def add_training_hook(model: HELPER_SUPPORT_MODEL_TYPES,
                      dest: Dict[str, Dict[str, List[torch.Tensor]]], 
                      intermediate_size: int,
                      hidden_size: int) -> int:
    if isinstance(model, LlamaPreTrainedModel):
        print("+++++++++ add Llama training hook +++++++++")
        return add_training_hook_to_llama(model, dest, intermediate_size, hidden_size)
    else:
        raise NotImplementedError(f"Only support {HELPER_SUPPORT_MODEL_LIST}.")


def remove_training_hook(model: HELPER_SUPPORT_MODEL_TYPES):
    hooks = getattr(model, _HELPER_HOOK_KEY)
    for handle in hooks:
        handle.remove()

    setattr(model, _HELPER_HOOK_KEY, None)


def llama_mlp_forward_sola(module, inp, **kwargs):
    if module.config.pretraining_tp > 1:
        raise NotImplementedError
    
    if module.config.pretraining_tp > 1:
        slice = module.intermediate_size // module.config.pretraining_tp
        gate_proj_slices = module.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = module.up_proj.weight.split(slice, dim=0)
        down_proj_slices = module.down_proj.weight.split(slice, dim=1)

        gate_proj = torch.cat(
            [F.linear(inp, gate_proj_slices[i]) for i in range(module.config.pretraining_tp)], dim=-1
        )
        up_proj = torch.cat([F.linear(inp, up_proj_slices[i]) for i in range(module.config.pretraining_tp)], dim=-1)

        intermediate_states = (module.act_fn(gate_proj) * up_proj)
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(module.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:
        if kwargs['layer_idx'] not in kwargs['pruned_layer_idx_list']:
            down_proj = module.down_proj(module.act_fn(module.gate_proj(inp)) * module.up_proj(inp))
        else:
            # prime
            h_gate_heavy = module.act_fn(torch.nn.functional.linear(inp, module.gate_weight_heavy))
            h_up_heavy = torch.nn.functional.linear(inp, module.up_weight_heavy)
            intermediate_states_heavy = h_gate_heavy * h_up_heavy
            down_proj_heavy = torch.nn.functional.linear(intermediate_states_heavy, module.down_weight_heavy)
            
            # marginal
            if module.gate_proj_use > 0:
                h_gate = module.act_fn(module.gate_proj(inp))
            else:
                if inp.device != module.gate_weight_U_top.device:
                    module.gate_weight_U_top = module.gate_weight_U_top.to(inp.device)
                tmp = torch.nn.functional.linear(inp, module.gate_weight_U_top)
                if tmp.device != module.gate_weight_SVh_top.device:
                    module.gate_weight_SVh_top = module.gate_weight_SVh_top.to(tmp.device)
                h_gate = module.act_fn(torch.nn.functional.linear(tmp, module.gate_weight_SVh_top))
            
            if module.up_proj_use > 0:
                h_up = module.up_proj(inp)
            else:
                if inp.device != module.up_weight_U_top.device:
                    module.up_weight_U_top = module.up_weight_U_top.to(inp.device)
                tmp = torch.nn.functional.linear(inp, module.up_weight_U_top)
                if tmp.device != module.up_weight_SVh_top.device:
                    module.up_weight_SVh_top = module.up_weight_SVh_top.to(tmp.device)
                h_up = torch.nn.functional.linear(tmp, module.up_weight_SVh_top)
            
            if h_gate.device != h_up.device:
                h_gate = h_gate.to(h_up.device)
            intermediate_states = h_gate * h_up
            
            if module.down_proj_use > 0:
                down_proj = module.down_proj(intermediate_states)
            else:
                if intermediate_states.device != module.down_weight_U_top.device:
                    module.down_weight_U_top = module.down_weight_U_top.to(intermediate_states.device)
                tmp = torch.nn.functional.linear(intermediate_states, module.down_weight_U_top)
                if tmp.device != module.down_weight_SVh_top.device:
                    module.down_weight_SVh_top = module.down_weight_SVh_top.to(tmp.device)
                down_proj_light = torch.nn.functional.linear(tmp, module.down_weight_SVh_top)
            
            down_proj = down_proj_heavy + down_proj_light
            
    return down_proj

def llama_attn_forward_sola(module: torch.nn.Module, 
                           hidden_states: torch.Tensor, 
                           attention_mask: Optional[torch.Tensor] = None, 
                           position_ids: Optional[torch.LongTensor] = None, 
                           past_key_value: Optional[Tuple[torch.Tensor]] = None, 
                           output_attentions: Optional[bool] = False, 
                           use_cache: Optional[bool] = False, 
                           **kwargs):
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    
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
        ##### Attn q/k decomposition #####
        value_states = module.v_proj(hidden_states)
        if kwargs['layer_idx'] not in kwargs['pruned_layer_idx_list']:
            query_states = module.q_proj(hidden_states)
            key_states = module.k_proj(hidden_states)
        else:
            if module.q_proj_use > 0:
                query_states = module.q_proj(hidden_states)
            else:
                if hidden_states.device != module.q_weight_U_top.device:
                    module.q_weight_U_top = module.q_weight_U_top.to(hidden_states.device)
                tmp = torch.nn.functional.linear(hidden_states, module.q_weight_U_top)
                if tmp.device != module.q_weight_SVh_top.device:
                    module.q_weight_SVh_top = module.q_weight_SVh_top.to(tmp.device)
                query_states = torch.nn.functional.linear(tmp, module.q_weight_SVh_top)
            
            if module.k_proj_use > 0:
                key_states = module.k_proj(hidden_states)
            else:
                if hidden_states.device != module.k_weight_U_top.device:
                    module.k_weight_U_top = module.k_weight_U_top.to(hidden_states.device)
                tmp = torch.nn.functional.linear(hidden_states, module.k_weight_U_top)
                if tmp.device != module.k_weight_SVh_top.device:
                    module.k_weight_SVh_top = module.k_weight_SVh_top.to(tmp.device)
                key_states = torch.nn.functional.linear(tmp, module.k_weight_SVh_top)
        
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
        key_states, value_states = past_key_value.update(key_states, value_states, module.layer_idx, cache_kwargs)

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
    
    attn_output = attn_output.reshape(bsz, q_len, module.hidden_size)

    if module.config.pretraining_tp > 1:
        attn_output = attn_output.split(module.hidden_size // module.config.pretraining_tp, dim=2)
        o_proj_slices = module.o_proj.weight.split(module.hidden_size // module.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(module.config.pretraining_tp)])
    else:
        ##### Attn o decomposition #####
        if kwargs['layer_idx'] not in kwargs['pruned_layer_idx_list']:
            attn_output = module.o_proj(attn_output)
        else:
            if module.o_proj_use > 0:
                attn_output = module.o_proj(attn_output)
            if attn_output.device != module.o_weight_U_top.device:
                module.o_weight_U_top = module.o_weight_U_top.to(attn_output.device)
            tmp = torch.nn.functional.linear(attn_output, module.o_weight_U_top)
            if tmp.device != module.o_weight_SVh_top.device:
                module.o_weight_SVh_top = module.o_weight_SVh_top.to(tmp.device)
            attn_output = torch.nn.functional.linear(tmp, module.o_weight_SVh_top)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def add_inference_hook_to_llama_sola(pruned_layer_idx_list, 
                                        model: HELPER_SUPPORT_MODEL_TYPES):
    set_helper_state(model, HelperState.Inference)
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, (LlamaMLP, LlamaAttention)):
            continue
        layer_idx = int(name.split(".")[-2])
        
        if isinstance(module, LlamaMLP):
            module.forward = functools.partial(
                llama_mlp_forward_sola,
                module,
                layer_idx=layer_idx,
                pruned_layer_idx_list=pruned_layer_idx_list,
                module_name=name
            )
        elif isinstance(module, LlamaAttention):
            hitter_dict_cur_layer_q = None
            hitter_dict_cur_layer_k = None
            
            module.forward = functools.partial(
                llama_attn_forward_sola,
                module,
                layer_idx=layer_idx,
                pruned_layer_idx_list=pruned_layer_idx_list,
                hitter_dict_q=hitter_dict_cur_layer_q,
                hitter_dict_k=hitter_dict_cur_layer_k
            )
        
def add_inference_hook(pruned_layer_idx_list, 
                       model: HELPER_SUPPORT_MODEL_TYPES):
    if isinstance(model, LlamaPreTrainedModel):
        add_inference_hook_to_llama_sola(pruned_layer_idx_list, model)
    else:
        raise NotImplementedError(f"Only support {HELPER_SUPPORT_MODEL_LIST}.")


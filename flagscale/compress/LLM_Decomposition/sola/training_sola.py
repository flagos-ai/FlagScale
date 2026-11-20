import os
import torch
import contextlib
from typing import Dict, List

from utils import HELPER_SUPPORT_MODEL_LIST, HELPER_SUPPORT_MODEL_TYPES

from sola.hook_sola import (
    add_training_hook,
    remove_training_hook,
    add_inference_hook,
)
import utils


class Helper(contextlib.ContextDecorator):
    def __init__(self, model: HELPER_SUPPORT_MODEL_TYPES, compute_type, **kwargs):
        self.model = model
        self.device = model.device
        self.compute_type = compute_type
        self.hidden_size = kwargs["hidden_size"]
        self.intermediate_size = kwargs["intermediate_size"]
        self.training_data: Dict[str, Dict[str, List[torch.Tensor]]] = {}

        if not isinstance(model, HELPER_SUPPORT_MODEL_LIST):
            raise NotImplementedError("Unsupported model")

    def __enter__(self):
        self.model_last_layer = add_training_hook(self.model, self.training_data, self.intermediate_size, self.hidden_size)

    def __exit__(self, exc_type, exc_val, exc_tb):
        remove_training_hook(self.model)

    def apply_sola_to_model(self, pruned_layer_idx_list, desired_rank_pref, hot_ratio, usv_dump_dest, model: HELPER_SUPPORT_MODEL_TYPES):
        import json
        hitter_dict_path = f'/data/hitter_dict_{hot_ratio}_256_4096_wiki_13b.json'
        with open(hitter_dict_path, 'r') as json_file:
            hitter_dict = json.load(json_file)
        
        def infer_device() -> torch.device:
            if not torch.cuda.is_available():
                return torch.device("cpu")
            max_free_memory = -1
            best_device_index = -1
            for i in range(torch.cuda.device_count()):
                current_device = torch.device(f"cuda:{i}")
                torch.cuda.set_device(current_device)
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated()
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device_index = i
            if best_device_index == -1:
                return torch.device("cpu")
            else:
                return torch.device(f"cuda:{best_device_index}")
        
        def regis_mlp_attn_func(pruned_layer_idx_list, model, hitter_dict):
            from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
            for name, module in model.named_modules():
                if not isinstance(module, (LlamaMLP, LlamaAttention)):
                    continue
                layer_idx = int(name.split(".")[-2])
                if layer_idx not in pruned_layer_idx_list:
                    continue
                
                dump_dest = usv_dump_dest
                
                if isinstance(module, (LlamaMLP)):
                    # prime
                    hitter_dict_cur_layer = hitter_dict[f'{layer_idx}']
                    heavy_25p_hitter = torch.tensor(hitter_dict_cur_layer['heavy_15p_neuron_idx'])
                    gate_weight_heavy = module.gate_proj.weight[heavy_25p_hitter, :]
                    up_weight_heavy =  module.up_proj.weight[heavy_25p_hitter, :]
                    down_weight_heavy = module.down_proj.weight[:, heavy_25p_hitter]
                    module.register_buffer('gate_weight_heavy', gate_weight_heavy.to(torch.bfloat16))
                    module.register_buffer('up_weight_heavy', up_weight_heavy.to(torch.bfloat16))
                    module.register_buffer('down_weight_heavy', down_weight_heavy.to(torch.bfloat16))
                    module.gate_proj = module.up_proj = module.down_proj = None
                    del module.gate_proj
                    del module.up_proj
                    del module.down_proj
                    utils.clear_torch_cache()
                    
                    # marginal
                    suffix_list = ["gate_proj", "up_proj", "down_proj"]
                    for suffix in suffix_list:
                        if suffix not in desired_rank_pref[f'{layer_idx}'].keys():
                            module.register_buffer(f'{suffix}_use', torch.Tensor([True]))
                            print(f"{suffix} not in desired rank {desired_rank_pref[f'{layer_idx}'].keys()}.")
                            light_idx = torch.tensor(hitter_dict_cur_layer['light_85p_neuron_idx'])
                            if suffix == "gate_proj":
                                gate_weight_light = module.gate_proj.weight[light_idx, :]
                                module.register_buffer('gate_weight_light', gate_weight_light.to(torch.bfloat16))
                            elif suffix == "up_proj":
                                up_weight_light = module.up_proj.weight[light_idx, :]
                                module.register_buffer('up_weight_light', up_weight_light.to(torch.bfloat16))
                            else:
                                down_weight_light = module.down_proj.weight[:, light_idx]
                                module.register_buffer('down_weight_light', down_weight_light.to(torch.bfloat16))
                        else:
                            module.register_buffer(f'{suffix}_use', torch.Tensor([False]))
                            u = torch.load(os.path.join(dump_dest, f"{name}.{suffix}.wu"), map_location=torch.device(infer_device()))
                            v = torch.load(os.path.join(dump_dest, f"{name}.{suffix}.wv"), map_location=torch.device(infer_device()))
                            print('get u v: ', name, suffix, u.shape, v.shape, u.device, v.device)
                            if suffix == "gate_proj":
                                module.register_buffer('gate_weight_U_top', v.t().to(torch.bfloat16))
                                module.register_buffer('gate_weight_SVh_top', u.t().to(torch.bfloat16))
                            elif suffix == "up_proj":
                                module.register_buffer('up_weight_U_top', v.t().to(torch.bfloat16))
                                module.register_buffer('up_weight_SVh_top', u.t().to(torch.bfloat16))
                            else:
                                module.register_buffer('down_weight_U_top', v.t().to(torch.bfloat16))
                                module.register_buffer('down_weight_SVh_top', u.t().to(torch.bfloat16))
                            u = s = v = None
                            del u, s, v
                            utils.clear_torch_cache()
                else:
                    suffix_list = ["q_proj", "k_proj", "o_proj"]
                    for suffix in suffix_list:
                        if suffix not in desired_rank_pref[f'{layer_idx}'].keys():
                            print(f"{suffix} not in {desired_rank_pref[f'{layer_idx}'].keys()}.")
                            module.register_buffer(f'{suffix}_use', torch.Tensor([True]))
                        else:
                            module.register_buffer(f'{suffix}_use', torch.Tensor([False]))
                            u = torch.load(os.path.join(dump_dest, f"{name}.{suffix}.wu"), map_location=torch.device(infer_device()))
                            v = torch.load(os.path.join(dump_dest, f"{name}.{suffix}.wv"), map_location=torch.device(infer_device()))
                            print('attn get u v: ', name, suffix, u.shape, v.shape, u.device, v.device)

                            if suffix == "q_proj":
                                module.register_buffer('q_weight_U_top', v.t().to(torch.bfloat16))
                                module.register_buffer('q_weight_SVh_top', u.t().to(torch.bfloat16))
                                module.q_proj = None
                                del module.q_proj
                                utils.clear_torch_cache()
                            elif suffix == "k_proj":
                                module.register_buffer('k_weight_U_top', v.t().to(torch.bfloat16))
                                module.register_buffer('k_weight_SVh_top', u.t().to(torch.bfloat16))
                                module.k_proj = None
                                del module.k_proj
                                utils.clear_torch_cache()
                            elif suffix == "o_proj":
                                module.register_buffer('o_weight_U_top', v.t().to(torch.bfloat16))
                                module.register_buffer('o_weight_SVh_top', u.t().to(torch.bfloat16))
                                module.o_proj = None
                                del module.o_proj
                                utils.clear_torch_cache()
                            # elif suffix == "v_proj":
                            #     module.register_buffer('v_weight_U_top', v.t().to(torch.bfloat16))
                            #     module.register_buffer('v_weight_SVh_top', u.t().to(torch.bfloat16))
                            #     module.v_proj = None
                            #     del module.v_proj
                            #     utils.clear_torch_cache()
                            u = s = v = None
                            del u, s, v
                            utils.clear_torch_cache()
        
        regis_mlp_attn_func(pruned_layer_idx_list, model, hitter_dict)
        
        add_inference_hook(pruned_layer_idx_list, model)
        
import os
import random
import click
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    pipeline
)
import torch
import numpy as np
import pandas as pd

import utils
from sola.training_sola import Helper

import json
import time
import logging
logger = logging.getLogger(__name__)

from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention

import lm_eval
from lm_eval.models.huggingface import HFLM

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
        

@click.command()
@click.option("-m", "--model", type=click.Path(file_okay=True), help="path to model file", default=None)
def cli(**kwargs):
    args = utils.EasyDict(**kwargs)
    print(args)

    model_path = args.model
    max_memory = "80000MB"
    max_memory = {i: max_memory for i in range(1)}
    
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=None,
            device_map="auto",
            quantization_config = None,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
    )
    print("Model created")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=None,
            padding_side="right",
            use_fast=False, # Fast tokenizer giving issues.
            tokenizer_type='llama' if 'ama' in model_path else None, # Needed for HF name change
        )
    if tokenizer._pad_token is None:
        utils.smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=utils.DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'ama' in model_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id and model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })
    print("Tokenizer created")
    
    generation_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    print("Pipeline created")
    
    helper_params = {'intermediate_size': model.config.intermediate_size, 
                     'hidden_size': model.config.hidden_size}
    helper = Helper(model, torch.bfloat16, **helper_params)
    print("Helper created")
    
    # Calibration data
    prompts_path = '/data/wiki_256_4096.json'
    with open(prompts_path, 'r') as file:
        prompts = json.load(file)
    
    # Compute XX^T
    t_start_time = time.time()
    with helper:
        for text in prompts:
            prompt_token_count = len(generation_pipeline.tokenizer.encode(text, return_tensors="pt")[0])
            generation_pipeline(text, max_length=int(prompt_token_count), pad_token_id=tokenizer.eos_token_id, truncation=True)
    t_end_time = time.time()
    t_duration = t_end_time - t_start_time
    print(f"Collect training data costs avg: {t_duration/len(prompts): .5f} s, all: {t_duration/60: .2f} min, {t_duration: .5f} s. ")
    print('Collect training data Done')
    
    # Record XX^T
    """
    for name, module in model.named_modules():
        suffix = name.split(".")[-1]
        if suffix not in ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]:
            continue
        layer_idx = int(name.split(".")[-3])
        raw_scaling_diag_matrix = getattr(module, f'raw_scaling_diag_matrix_{layer_idx}')
        # torch.save(raw_scaling_diag_matrix, os.path.join('/data/model_params/wiki/13b/raw_scaling_diag_matrix/', f"{name}.raw_scaling_diag_matrix"))
        print(name, 'raw scaling diag matrix saved')
    """
    
    # Low-Rank Decomposition
    dump_dest = f'/data/model_params/wiki/13b/light'
    for name, module in model.named_modules():
        suffix = name.split(".")[-1]
        if suffix not in ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]:
            continue
        layer_idx = int(name.split(".")[-3])
        
        raw_scaling_diag_matrix = getattr(module, f'raw_scaling_diag_matrix_{layer_idx}').double().to(model.device)
        # raw_scaling_diag_matrix = torch.load(f'/data/model_params/wiki/13b/raw_scaling_diag_matrix/{name}.raw_scaling_diag_matrix').double().to(model.device)
        
        with open('/data/hitter_dict_15_256_4096_wiki_13b.json', 'r') as json_file:
            hitter_dict = json.load(json_file)
        light_75p_hitter = torch.tensor(hitter_dict[f'{layer_idx}']['light_85p_neuron_idx']).to(model.device)
        if suffix == "down_proj":
            raw_scaling_diag_matrix = raw_scaling_diag_matrix[light_75p_hitter, :][:, light_75p_hitter]
        
        try:
            scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float().to(model.device)
        except Exception as e:
            print(name, "Warning: eigen scaling_diag_matrix is not positive!")
            if torch.isnan(raw_scaling_diag_matrix).any():
                print("Warning: scaling_diag_matrix contains NaN!")
            elif torch.isinf(raw_scaling_diag_matrix).any():
                print("Warning: scaling_diag_matrix contains Inf!")
            if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                print("Warning: scaling_diag_matrix is not a symmetric matrix!")
            eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
            raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-3) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(model.device)
            scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float().to(model.device)
        
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
        except Exception as e:
            print(name, "Warning: scaling_diag_matrix is not full rank!")
            scaling_diag_matrix += 1e-3 * torch.eye(scaling_diag_matrix.shape[0]).to(model.device)
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(model.device)
        
        W = module.weight.float()
        
        # MLP beta weight matrix decomposition
        if suffix in ["gate_proj", "up_proj", "down_proj"]:
            if light_75p_hitter.device != W.device:
                light_75p_hitter = light_75p_hitter.to(W.device)
            if suffix in ["gate_proj", "up_proj"]:
                W = W[light_75p_hitter, :].to(model.device)
            else:
                W = W[:, light_75p_hitter].to(model.device)
        
        if W.device != scaling_diag_matrix.device:
            scaling_diag_matrix = scaling_diag_matrix.to(W.device)
        W_scale = torch.matmul(W, scaling_diag_matrix)
        if layer_idx == 0:
            print('W scale shape: ', suffix, W_scale.shape)
        
        u, s, v = torch.linalg.svd(W_scale, full_matrices=False)    # The singular values are returned in descending order.
        if layer_idx == 0:
            print('decomposition: ', name, u.shape, s.shape, v.shape, W.shape, scaling_matrix_inv.shape)
        
        torch.save(u, os.path.join(dump_dest, f"{name}.u"))
        # torch.save(v, os.path.join(dump_dest, f"{name}.v"))
        torch.save(s, os.path.join(dump_dest, f"{name}.s"))
        print(name, 'u s v saved.', dump_dest)
        
        if v.device != scaling_matrix_inv.device:
            v = v.to(scaling_matrix_inv.device)
        v_inv = v @ scaling_matrix_inv
        torch.save(v_inv, os.path.join(dump_dest, f"{name}.v_inv"))
        print(name, 'v_inv saved.', dump_dest)
    
    # Compute the rank of each component
    target_rate = 0.7
    if target_rate >= 0.8:
        target_modules = ["q_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # config = utils.get_rank_config(model, target_modules, target_rate, args.model, start=99.9, end=90)
    # '''
    if '13b' in model_path:     # Llama-2-13b
        if target_rate == 0.8:  # 20% compression rate
            config = {"q_proj": 1440, "k_proj": 1440, 
                        "gate_proj": 2048, "up_proj": 2816, "down_proj": 2976}
        elif target_rate == 0.7:    # 30% compression rate
            config = {"q_proj": 960, "k_proj": 640, "o_proj": 1920, 
                        "gate_proj": 2080, "up_proj": 2400, "down_proj": 2560}
        elif target_rate == 0.6:    # 40% compression rate
            config = {"q_proj": 480, "k_proj": 480, "o_proj": 1440, 
                        "gate_proj": 1120, "up_proj": 2080, "down_proj": 2240}
        elif target_rate == 0.5:    # 50% compression rate
            config = {"q_proj": 320, "k_proj": 320, "o_proj": 768, 
                        "gate_proj": 1280, "up_proj": 1280, "down_proj": 1440}
    else:                       # Llama-2-7b
        if target_rate == 0.8:  # 20% compression rate
            config = {"q_proj": 1120, "k_proj": 800, 
                        "gate_proj": 1760, "up_proj": 2400, "down_proj": 2240}
        elif target_rate == 0.7:    # 30% compression rate
            config = {"q_proj": 640, "k_proj": 640, "o_proj": 1440, 
                        "gate_proj": 1760, "up_proj": 1920, "down_proj": 1760}
    # '''
    print(config)
    
    desired_ranks = {}
    layer_num = 40 if '13b' in model_path else 32
    for layer_idx in range(layer_num):
        for suffix in ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "o_proj"]:
            if f'{layer_idx}' not in desired_ranks.keys():
                desired_ranks[f'{layer_idx}'] = {suffix: (config[suffix], None)}
            else:
                desired_ranks[f'{layer_idx}'][suffix] = (config[suffix], None)
                
    if '13b' in model_path:
        pruned_layer_idx_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
                                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
                                    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
    elif '7b' in model_path:
        pruned_layer_idx_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
                                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
                                    26, 27, 28, 29]
    print('pruned layer number: ', len(pruned_layer_idx_list))
    
    # Reduce memory through low rank decomposition
    hot_ratio = 15
    if '13b' in model_path:
        active_params = model_params = 13343959040
        # edit file path
        dump_dest = '/data/model_params/wiki/13b/light'
        save_dest = f'/data/model_params/wiki/13b/uv_fils_{hot_ratio}_30'
    elif '7b' in model_path:
        active_params = model_params = 7000842240
        # dump_dest save_dest edit file path
    for filename in os.listdir(save_dest):
        if filename.split('.')[-1] not in ['wu', 'wv']:
            continue
        file_path = os.path.join(save_dest, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f'{file_path} deletion.')
    for name, module in model.named_modules():
        if not isinstance(module, (LlamaMLP, LlamaAttention)):
            continue
        layer_idx = int(name.split(".")[-2])
        if layer_idx not in pruned_layer_idx_list:
            continue
        
        suffix_list = ["q_proj", "k_proj" , "o_proj", "gate_proj", "up_proj", "down_proj"]
        for suffix in suffix_list:
            u = torch.load(os.path.join(dump_dest, f"{name}.{suffix}.u"), map_location=torch.device('cuda'))
            s = torch.load(os.path.join(dump_dest, f"{name}.{suffix}.s"), map_location=torch.device('cuda'))
            v = torch.load(os.path.join(dump_dest, f"{name}.{suffix}.v_inv"), map_location=torch.device('cuda'))
            k = desired_ranks[f'{layer_idx}'][suffix][0]
            u, v = utils.get_uv(u, s, v, k)
            
            if suffix in ["gate_proj", "up_proj", "down_proj"]:
                module_weight_numel = model.config.intermediate_size * model.config.hidden_size
                module_weight_numel = int(0.85 * module_weight_numel)
            elif suffix in ["q_proj", "k_proj"]:
                module_weight_numel = model.config.hidden_size * \
                                            (model.config.hidden_size / 
                                             (model.config.num_attention_heads / model.config.num_key_value_heads))
            elif suffix in ["o_proj"]:
                module_weight_numel = model.config.hidden_size * model.config.hidden_size
            active_params -= module_weight_numel - v.numel() - u.numel()
            
            torch.save(u, os.path.join(save_dest, f"{name}.{suffix}.wu"))
            torch.save(v, os.path.join(save_dest, f"{name}.{suffix}.wv"))
            print(f"{name}.{suffix} {k} wu wv saved.")

            u = s = v = None
            del u, s, v
            utils.clear_torch_cache()
    print(f"Estimated compression rate: {1 - active_params/model_params:.4f}")
    
    helper.apply_sola_to_model(pruned_layer_idx_list, desired_ranks, hot_ratio, save_dest, model)
    utils.clear_torch_cache()
    print(f'Appling Done')
    
    # torch.save(model.state_dict(), f'/data/model_params/model_{target_rate}.pth')
    
    model.eval()
    setup_seed(42)
    
    # Evaluate perplexity
    ppl = utils.eval_ppl(model, tokenizer)
    print('ppl: ', ppl)
    
    # Evaluate lm eval accuracy
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)
    task_names = ['piqa', 'hellaswag', 'boolq', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=0, batch_size=8)[
        'results'
    ]
    print(results)
    metric_vals = {task: round(result.get(utils.TASK_METRIC_MAP[task]), 4) for task, result in results.items()}
    acc_avg = utils.calculate_avg_accuracy(task_names, results)
    metric_vals['average'] = round(acc_avg, 4)
    print(metric_vals)
    
    # Evaluate mmlu accuracy
    utils.eval_mmlu(model, tokenizer, 5, "data/mmlu-data")
    print('Eval MMLU Done \n')

    
if __name__ == "__main__":
    cli()

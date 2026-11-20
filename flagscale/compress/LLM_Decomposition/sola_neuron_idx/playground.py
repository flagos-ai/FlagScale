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
from training import Helper

import json
import time
import logging
logger = logging.getLogger(__name__)


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
    
    setup_seed(42)
    
    if '13b' in model_path:
        intermediate_size = 13824
    elif '7b' in model_path:
        intermediate_size = 11008
    helper_params = {'intermediate_size': intermediate_size}
    helper = Helper(model, torch.bfloat16, **helper_params)
    print("Helper created")
    
    # Construct calibration data
    # wiki_dataset = utils.get_wikitext2(256, 3, 4096, tokenizer, 'wiki')
    # with open('/data/wiki_256_4096.json', 'w') as json_file:
    #     json.dump(wiki_dataset, json_file, ensure_ascii=False, indent=4)
    with open('/data/wiki_256_4096.json', 'r') as file:
        prompts_all = json.load(file)
    
    # calibration prompt number
    prompt_num_list = [256]
    for prompt_num in prompt_num_list:
        prompts = prompts_all[:prompt_num]
        print('prompt number: ', len(prompts))

        # Compute neurons norm
        t_start_time = time.time()
        with helper:
            for text in prompts:
                prompt_token_count = len(generation_pipeline.tokenizer.encode(text, return_tensors="pt")[0])
                generation_pipeline(text, max_length=prompt_token_count, pad_token_id=tokenizer.eos_token_id, truncation=True)
        t_end_time = time.time()
        t_duration = t_end_time - t_start_time
        print(f"Collect training data costs avg: {t_duration/len(prompts): .5f} s, all: {t_duration/60: .2f} min, {t_duration: .5f} s. ")
        print('Collect training data Done')
        
        # Record neurons norm
        import csv
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        
        for name, module in model.named_modules():
            if not isinstance(module, LlamaDecoderLayer):
                continue
            layer_idx = int(name.split(".")[-1])
            
            neurons_dict_cur_layer = getattr(module, f'neurons_dict_{layer_idx}')
            average_values = neurons_dict_cur_layer['norm'] / neurons_dict_cur_layer['token_num']
            csv_data = [(k, average_values[k].item()) for k in range(average_values.shape[0])]
            csv_file = f'/data/mlp_neurons_norm/sample_{prompt_num}/13b_{layer_idx}.csv'

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)
            print(layer_idx, csv_file, 'write done')
        print(prompt_num, 'Neurons Norm Write Done')
        
        ###### Record prime/marginal neurons index ######
        dir_name = f'/data/mlp_neurons_norm/sample_{prompt_num}'
        checkpoint_path_list = os.listdir(dir_name)
        hitter_dict = {}
        for checkpoint_path in checkpoint_path_list:
            layer_idx = int(checkpoint_path.split('_')[-1].split('.')[0])
            data_df = pd.read_csv(os.path.join(dir_name, checkpoint_path), index_col=False, header=None)
            data_df = data_df.sort_values(by=1, ascending=False)
            num_top = int(len(data_df) * 0.15)  # prime neurons 15%
            heavy_neron_idx = data_df[0][:num_top].tolist()
            light_neron_idx = data_df[0][num_top:].tolist()
            hitter_dict[layer_idx] = {'heavy_15p_neuron_idx': heavy_neron_idx, 
                                    'light_85p_neuron_idx': light_neron_idx}
        
        with open(f'{dir_name}/hitter_dict_15_{prompt_num}_4096_wiki_13b.json', 'w') as json_file:
            json.dump(hitter_dict, json_file, ensure_ascii=False, indent=4)
        print(prompt_num, 'Prime Dict Done')
        
    
if __name__ == "__main__":
    cli()

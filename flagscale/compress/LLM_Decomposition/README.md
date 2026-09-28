<div align="center">
<h1>SoLA</h1>
<h3>
[AAAI 2025] SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compression
<h3>
</div>

<p align="center">
<img width="100%" alt="image" src="data/overview.jpg">    
</p>

## Usage

### Construct Calibration Data
    sola_neuron_idx/utils.py get_wikitext2() --> data/wiki_256_4096.json

### Compute Neurons Norm
    bash sola_neuron_idx/playground.sh --> data/hitter_dict_15_256_4096_wiki_13b.json

### Low-Rank Decomposition and Evaluation
    bash sola/playground_sola.sh

    Llama-2-7B/13B low-rank decomposition and evaluation

    Remember modify file path to your own 

### Others
    sola_neuron_idx/hook.py llama_self_attn() past_key_value.update_get() in transformers-4.37.1/src/transformers/cache_utils.py

    You can install modified transformers package: cd transformers-4.37.1; pip install -e .

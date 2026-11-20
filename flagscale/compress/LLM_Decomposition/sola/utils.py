import os
import re
import gc
import time
import json
import pickle
import torch
import transformers
import lm_eval
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
from datasets import load_dataset
from typing import Any, Union, Dict, TypeVar, Generic, Iterable, List, Iterator
from transformers import OPTPreTrainedModel, LlamaPreTrainedModel
from optim import Optimizer
import copy
import itertools
import bisect
import warnings


T_co = TypeVar('T_co', covariant=True)
class Dataset(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])


class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class IterableDataset(Dataset[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])

class ChainDataset(IterableDataset):
    r"""Dataset for chainning multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chainning operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ChainDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            # Cannot verify that all self.datasets are Sized
            total += len(d)  # type: ignore
        return total

def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):
    model.eval()
    token_num = 0
    factor = 1 if batch_size > 32 else 3
    num_batches_to_fetch = 13 * factor
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)
    start_time = time.time()
    progress_bar = tqdm(enumerate(itertools.islice(test_loader, num_batches_to_fetch)))
    for batch_idx, batch_data in progress_bar:
        batch = batch_data.to(device)
        if batch_idx == 3 * factor:
            start_time = time.time()
        if batch_idx >= 3 * factor:
            token_num += batch.shape[0] * generated_len
            progress_bar.set_postfix_str(f"{token_num=}")
        generation_output = model.generate(
            input_ids=batch,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            use_cache=True,
            max_length=original_len+generated_len,
        )
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    throughput = token_num / total_time
    return throughput


def eff_eval_v2(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):
    model.eval()
    throughput = 0
    token_num = 0
    num_batches_to_fetch = 130
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        if batch_idx >= 30:
            break
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    
    start_time = time.time()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        if batch_idx < 30:
            continue
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
    torch.cuda.synchronize()
    end_time = time.time()
    throughput = end_time - start_time
    print("time: {}".format(end_time - start_time))
    print("Throughput: {} tokens/sec".format(token_num / throughput))
    return token_num / throughput


def clear_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()

def get_uv(u, s, v, k):
    svd_u = u[:, :k]
    svd_s = s[:k]
    svd_v = v[:k, :]
    sqrt_s = torch.diag(torch.sqrt(svd_s))
    if svd_u.device != sqrt_s.device:
        print('svd u s device: ', svd_u.device, sqrt_s.device)
        svd_u = svd_u.to(sqrt_s.device)
    if sqrt_s.device != svd_v.device:
        print('svd s v device: ', sqrt_s.device, svd_v.device)
        svd_v = svd_v.to(sqrt_s.device)
    clear_torch_cache()
    u=(svd_u @ sqrt_s).T
    v=(sqrt_s @ svd_v).T
    return u, v

def get_rank_config(model, target_modules, target_rate, model_path, start=99.9, end=90):
    optimizer = constitute_mapping(model, target_modules, target_rate, model_path, start=99.9, end=90)
    optimized_state = optimizer.constringe()
    rank = {}
    for name, module in model.named_modules():
        suffix = name.split(".")[-1]
        if target_rate >= 0.8:
            target_modules = ["q_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if suffix not in target_modules or name not in optimized_state.keys():
            print(f"{name} skipped.")
            continue
        cur = optimized_state[name]["cur"]
        desired_rank = (cur + 1) * 32                           # rank
        accum = optimized_state[name]["perf_var"][cur]          # perf score
        layer_idx = int(name.split(".")[-3])
        if layer_idx not in rank.keys():
            rank[layer_idx] = {suffix: (desired_rank, accum)}
        else:
            rank[layer_idx][suffix] = (desired_rank, accum)
        print(f"{name=}, {desired_rank=}, {accum:.2f}%.")
    
    q_proj_values_non_zero = []
    k_proj_values_non_zero = []
    o_proj_values_non_zero = []
    gate_proj_values_non_zero = []
    up_proj_values_non_zero = []
    down_proj_values_non_zero = []
    for key in rank.keys():
        if 'q_proj' in rank[key].keys():
            q_proj_values_non_zero.append(rank[key]['q_proj'][0])
        if 'k_proj' in rank[key].keys():
            k_proj_values_non_zero.append(rank[key]['k_proj'][0])
        if 'o_proj' in rank[key].keys():
            o_proj_values_non_zero.append(rank[key]['o_proj'][0])
        if 'gate_proj' in rank[key].keys():
            gate_proj_values_non_zero.append(rank[key]['gate_proj'][0])
        if 'up_proj' in rank[key].keys():
            up_proj_values_non_zero.append(rank[key]['up_proj'][0])
        if 'down_proj' in rank[key].keys():
            down_proj_values_non_zero.append(rank[key]['down_proj'][0])
    
    q_r = sorted(q_proj_values_non_zero)[-int(len(sorted(q_proj_values_non_zero)) * 0.3)]
    k_r = sorted(k_proj_values_non_zero)[-int(len(sorted(k_proj_values_non_zero)) * 0.3)]
    q_r = ((q_r + 159) // 160) * 160
    k_r = ((k_r + 159) // 160) * 160
    print('q:', q_r, int(len(sorted(q_proj_values_non_zero)) * 0.3))
    print('k:', k_r, int(len(sorted(k_proj_values_non_zero)) * 0.3))
    if target_rate < 0.8:
        o_r = sorted(o_proj_values_non_zero)[-int(len(sorted(o_proj_values_non_zero)) * 0.55)]
        o_r = ((o_r + 159) // 160) * 160
        print('o:', o_r, int(len(sorted(o_proj_values_non_zero)) * 0.55))
    if target_rate == 0.7:
        gate_r = sorted(gate_proj_values_non_zero)[-int(len(sorted(gate_proj_values_non_zero)) * 0.3)]
    else:
        gate_r = sorted(gate_proj_values_non_zero)[-int(len(sorted(gate_proj_values_non_zero)) * 0.7)]
    gate_r = ((gate_r + 159) // 160) * 160
    up_r = sorted(up_proj_values_non_zero)[-int(len(sorted(up_proj_values_non_zero)) * 0.6)]
    up_r = ((up_r + 159) // 160) * 160
    print('gate:', gate_r, int(len(sorted(gate_proj_values_non_zero)) * 0.5))
    print('up:', up_r, int(len(sorted(up_proj_values_non_zero)) * 0.5))
    if '13b' in model_path:
        down_r = int((5120 * int(13824 * 0.85)) / (5120 + int(13824 * 0.85)) * target_rate)
    if '7b' in model_path:
        down_r = int((4096 * int(11008 * 0.85)) / (4096 + int(11008 * 0.85)) * target_rate)
    down_r = ((down_r + 159) // 160) * 160
    print('down:', down_r)
    
    if target_rate < 0.8:
        config = {"q_proj": q_r, "k_proj": k_r, "o_proj": o_r, 
                "gate_proj": gate_r, "up_proj": up_r, "down_proj": down_r}
    else:
        config = {"q_proj": q_r, "k_proj": k_r, 
                "gate_proj": gate_r, "up_proj": up_r, "down_proj": down_r}
    
    return config

def constitute_mapping(model, target_modules, target_rate, model_name,
                       granularity=32, start=99, end=90, 
                       dump_dest="/home/xinhao/sep-peft/model_params/wiki/svd_llm/light", 
                       dump_dest_attn="/home/xinhao/sep-peft/model_params/wiki/svd_llm/attn"):
    if "Llama-2-13b-hf" in model_name:
        config = {
            "meta-llama/Llama-2-13b-hf": {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 5120,
                "intermediate_size": 13824,
                "num_attention_heads": 40,
                "num_hidden_layers": 40,
                "vocab_size": 32000,
                "params": 13343959040,
                "lora": 125173760,
                "optim": 250347520,
                "grad": 125173760,
            }
        }
        skip_layers = [0, 1, 38, 39]
    elif "Llama-2-7b-hf" in model_name:
        config = {
            "meta-llama/Llama-2-7b-hf": {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "params": 7000842240,
                "lora": 79953920,
                "optim": 159907840,
                "grad": 79953920,
            }
        }
        skip_layers = [0, 1, 29, 30]
    model_params = config[model_name]["params"]
    target_rate *= 100
    optimizer = Optimizer(model_params, target_rate, granularity, start, end)

    # Called during adding hook for updating
    for name, module in model.named_modules():
        suffix = name.split(".")[-1]
        
        if suffix not in target_modules:
            print(f"[constitute-mapping] skipping {name} due to configuration")
            continue
        
        layer_idx = int(name.split(".")[-3])
        if layer_idx in skip_layers:
            print(f"[constitute-mapping] skipping {layer_idx} due to configuration")
            continue
        
        if target_rate >= 80:
            suffix_attn_list = ["q_proj", "k_proj"]
        else:
            suffix_attn_list = ["q_proj", "k_proj", "o_proj"]
        if suffix in suffix_attn_list:
            if not os.path.exists(os.path.join(dump_dest_attn, f"{name}.s")):
                print(f"[constitute-mapping] skipping {name} since it cannot find sigma file")
                continue
            s = torch.load(os.path.join(dump_dest_attn, f"{name}.s"), map_location="cuda:0")
        else:
            if not os.path.exists(os.path.join(dump_dest, f"{name}.s")):
                print(f"[constitute-mapping] skipping {name} since it cannot find sigma file")
                continue
            s = torch.load(os.path.join(dump_dest, f"{name}.s"), map_location="cuda:0")

        m, n = module.weight.shape
        max_trunc = int((m * n) / (m + n))
        sigma_square = s ** 2
        total = sigma_square.sum()

        perf_var = []
        grad = []
        start_idx = 10000000  # random large number
        for trunc in range(granularity, max_trunc, granularity):
            perf = (sigma_square[:trunc].sum() / total * 100).item()  # in percentage
            if perf >= start:
                start_idx = min(len(perf_var), start_idx)
            idx = len(perf_var)
            perf_var.append(perf)
            grad.append(0 if idx == 0 else perf - perf_var[idx-1])

        start_idx = min(start_idx, len(perf_var) - 1)

        # skip if it is not profitable
        deft_uniform = None
        if deft_uniform is not None and (start_idx + 1) * granularity >= max_trunc - 5 * granularity:
            print(f"[constitute-mapping] skipping {name} for trivial profit")
            continue
        
        # collect
        optimizer.add_item(name, start_idx, tuple([m, n]), perf_var, grad)

    return optimizer


class HelperState(Enum):
    KEY = 10000

    Collecting = 0
    Inference = 1

    Invalid = 9999


HelperState.KEY.label = "HelperState"
HelperState.Collecting.label = "Helper-Data-Collection"  # hook forward() to collect data
HelperState.Inference.label = "Helper-Ready-Inference"    # with updated forward()

class HelperState(Enum):
    KEY = 10000

    Collecting = 0
    Inference = 1

    Invalid = 9999


HelperState.KEY.label = "HelperState"
HelperState.Collecting.label = "Helper-Data-Collection"  # hook forward() to collect data
HelperState.Inference.label = "Helper-Ready-Inference"    # with updated forward()


class HelperCollectState(Enum):
    KEY = 10001

    Pre = 0
    Post = 1
    End = 2

    Invalid = 9999


HelperCollectState.KEY.label = "HelperCollectState"
HelperCollectState.Pre.label = "HelperCollectState-Pre"
HelperCollectState.Post.label = "HelperCollectState-Post"
HelperCollectState.End.label = "HelperCollectState-End"

def set_helper_state(model, state: HelperState) -> None:
    setattr(model, HelperState.KEY.label, state)


HELPER_SUPPORT_MODEL_LIST = (LlamaPreTrainedModel)
HELPER_SUPPORT_MODEL_TYPES = Union[LlamaPreTrainedModel]


# https://pypi.org/project/lm-eval/0.0.1/
TASK_METRIC_MAP = {
    "piqa": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    'wsc': 'acc,none',
    "openbookqa": "acc_norm,none"
}

def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if 'mmlu' not in task)

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get(TASK_METRIC_MAP[task]) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)


def easy_dump(obj, dest, label):
    with open(os.path.join(dest, f"{label}.pkl"), "wb") as f:
        pickle.dump(obj, f)

    # also dump as json if it is a dict
    if isinstance(obj, dict):
        with open(os.path.join(dest, f"{label}.json"), "w") as f:
            f.write(json.dumps(obj, indent=4))
            
def make_run_dir(outdir: Union[str, os.PathLike], desc: str) -> str:
    """Reject modernity, return to automatically create the run dir."""
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):  # sanity check, but click.Path() should clear this one
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1  # start with 00000
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    os.makedirs(run_dir, exist_ok=False)  # make sure it doesn't already exist
    return run_dir

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

DEFAULT_PAD_TOKEN = "[PAD]"
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def evaluate_subject(subject, model, tokenizer, ntrain, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in tqdm(range(test_df.shape[0]), desc=subject):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def eval_mmlu(model, tokenizer, ntrain, data_dir):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    
    start_time = time.time()
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = evaluate_subject(subject, model, tokenizer, ntrain, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    end_time = time.time()
    results["cost_time"] = end_time - start_time
    
    return results

def eval_ppl(model, tokenizer):
    model.eval()
    max_length = model.config.max_position_embeddings
    stride = max_length
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    nlls = []
    encodings = tokenizer("\n\n".join(test), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

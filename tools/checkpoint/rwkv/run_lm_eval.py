########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
# pip install rwkv lm_eval --upgrade

import os, sys, types, json, math, time
from tqdm import tqdm
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch
from torch.nn import functional as F

os.environ["RWKV_V7_ON"] = '1'
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.tasks import get_task_dict
from lm_eval.evaluator import simple_evaluate

########################################################################################################

if len(sys.argv) < 2:
    print("Usage: python your_script_name.py /path/to/your/model.pth")
    sys.exit(1)

MODEL_NAME = sys.argv[1]

print(f'Loading model - {MODEL_NAME}')
model = RWKV(model=MODEL_NAME, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

eval_tasks = [
    'lambada_openai', 'piqa', 'storycloze_2016', 'hellaswag', 'winogrande',
    'arc_challenge', 'arc_easy', 'headqa_en', 'openbookqa', 'sciq',
    'mmlu','glue']
num_fewshot = 0

RWKV_PAD = pipeline.tokenizer.encode('\n')
STOP_TOKEN = pipeline.tokenizer.encode('\n\n')
print('RWKV_PAD', RWKV_PAD)
print('STOP_TOKEN', STOP_TOKEN)

########################################################################################################

class EvalHarnessAdapter(LM):
    def __init__(self):
        super().__init__()
        self.tokenizer = pipeline.tokenizer
        self.model = model

    @property
    def eot_token_id(self):
        # End of text token
        return self.tokenizer.encode('\n\n')[0]

    @property
    def max_length(self):
        return 4096

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def loglikelihood(self, requests):
        res = []
        for request in tqdm(requests, "Running loglikelihood requests"):
            context, continuation = request.args
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation)

            full_enc = context_enc + continuation_enc
            outputs, _ = self.model.forward(full_enc, None, full_output=True)
            log_probs = F.log_softmax(outputs, dim=-1).cpu()

            continuation_log_likelihood = 0.0
            for i in range(len(continuation_enc)):
                token_id = continuation_enc[i]
                token_log_prob = log_probs[len(context_enc) - 1 + i, token_id]
                continuation_log_likelihood += token_log_prob

            last_token_logits = outputs[len(full_enc) - 2].float()
            pred_token = torch.argmax(last_token_logits).item()
            is_greedy = (pred_token == continuation_enc[-1])
            
            res.append((continuation_log_likelihood.item(), is_greedy))
        return res

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        loglikelihoods = []
        for request in tqdm(requests, "Running loglikelihood_rolling requests"):
            string, = request.args
            tokens = self.tok_encode(string)
            
            if not tokens:
                loglikelihoods.append(0.0)
                continue

            outputs, _ = self.model.forward(tokens, None, full_output=True)
            log_probs = F.log_softmax(outputs, dim=-1).cpu()

            total_log_likelihood = 0.0
            for i in range(1, len(tokens)):
                token_id = tokens[i]
                total_log_likelihood += log_probs[i - 1, token_id]
            
            loglikelihoods.append(total_log_likelihood.item())
        
        return loglikelihoods

    @torch.no_grad()
    def generate_until(self, requests):
        res = []
        for request in tqdm(requests, "Running generation requests"):
            context = request.args[0]
            gen_kwargs = request.args[1]
            until = gen_kwargs.get("until", None)
            
            context_tokens = self.tok_encode(context)
            
            all_tokens = []
            state = None
            
            out, state = model.forward(context_tokens, state)

            for i in range(self.max_gen_toks):
                token = torch.argmax(out).item()
                
                if until and any(self.tok_decode([token]).startswith(stop_str) for stop_str in until):
                    break
                if token in STOP_TOKEN:
                    break

                all_tokens.append(token)
                out, state = model.forward([token], state)

            res.append(self.tok_decode(all_tokens))
        return res

adapter = EvalHarnessAdapter()

print(f'Running evaluation on {eval_tasks} with {num_fewshot}-shot examples')
results = simple_evaluate(
    model=adapter,
    tasks=eval_tasks,
    num_fewshot=num_fewshot,
    limit=None,
    bootstrap_iters=10000,
)

print(json.dumps(results['results'], indent=2))

task_str = '-'.join(eval_tasks)
output_dir = os.path.dirname(MODEL_NAME)
if not output_dir:
    output_dir = "."
base_name = os.path.basename(MODEL_NAME)
metric_output_path = os.path.join(output_dir, f"{base_name.replace('.pth', '')}_{task_str}.json")

output_dict = dict(model=MODEL_NAME, tasks=eval_tasks, num_fewshot=num_fewshot, results=results['results'])
with open(metric_output_path, 'w', encoding='utf-8') as f:
    json.dump(output_dict, f, indent=2, ensure_ascii=False)

print(f"Results saved to {metric_output_path}")

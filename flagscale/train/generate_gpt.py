"""
MemRift-enabled Megatron GPT text generation.

Layer-by-layer weight streaming: decompress one layer at a time (CPU → GPU),
run forward, release. GPU peak memory ≈ 1 layer (~400 MB) vs 13.5 GB full load.

Usage example (Mistral-7B):
  torchrun --nnodes 1 --nproc_per_node 1 flagscale/train/generate_gpt.py \\
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \\
    --use-flash-attn --bf16 --disable-bias-linear \\
    --transformer-impl transformer_engine \\
    --num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336 \\
    --num-attention-heads 32 --num-query-groups 8 \\
    --seq-length 2048 --max-position-embeddings 131072 \\
    --norm-init-weight 1.0 --use-rotary-position-embeddings \\
    --rotary-base 1000000 --no-position-embedding \\
    --normalization RMSNorm --norm-epsilon 1e-05 --swiglu \\
    --untie-embeddings-and-output-weights --make-vocab-size-divisible-by 64 \\
    --tokenizer-type HuggingFaceTokenizer \\
    --tokenizer-path /path/to/model --tokenizer-model /path/to/model \\
    --memrift-enable --memrift-weight-enable \\
    --memrift-compressed-weight-dir /path/to/compressed_weights \\
    --memrift-zstd-level 3 --memrift-prefetch-layers 1 \\
    --memrift-weight-async --memrift-decode-pool-workers 48 \\
    --prompt "Once upon a time" --max-new-tokens 50 --greedy
"""

import os
import sys
import time
from functools import partial

import torch

# ── path: make flagscale/train importable (same as train_gpt.py) ──
_train_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _train_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(_train_dir)))

from gpt_builders import gpt_builder
from model_provider import model_provider

from megatron.core import InferenceParams  # = StaticInferenceContext
from megatron.core.enums import ModelType
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.initialize import initialize_megatron

# ════════════════════════════════════════════════════════════════════════════
# Extra CLI args
# ════════════════════════════════════════════════════════════════════════════


def _extra_args(parser):
    group = parser.add_argument_group("MemRift Generate")
    group.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt for generation.",
    )
    group.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate.",
    )
    group.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="If >0, build a synthetic prompt of exactly N tokens (ignores --prompt). "
        "Used for max-context-length benchmarking.",
    )
    group.add_argument(
        "--kv-quant-8bit",
        action="store_true",
        help="Store the KV cache in int8 (per-token asymmetric, KIVI-style) to halve "
        "KV memory and extend max context. Near-lossless.",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (1.0 = no scaling).",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-K sampling (0 = disabled).",
    )
    group.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (argmax).",
    )
    return parser


# ════════════════════════════════════════════════════════════════════════════
# Attention mask helpers
# ════════════════════════════════════════════════════════════════════════════


def _causal_mask(seq_len: int, device) -> torch.Tensor:
    """Lower-triangular causal mask [1, 1, seq_len, seq_len]; True = masked."""
    m = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return (m < 0.5).unsqueeze(0).unsqueeze(0)  # [1, 1, s, s] bool


def _decode_mask(seq_len: int, device) -> torch.Tensor:
    """Decode-step mask [1, 1, 1, seq_len]; all False = new token attends to all."""
    return torch.zeros(1, 1, 1, seq_len, dtype=torch.bool, device=device)


# ════════════════════════════════════════════════════════════════════════════
# Sampling
# ════════════════════════════════════════════════════════════════════════════


def _sample(logits: torch.Tensor, args) -> torch.Tensor:
    """Return next-token id [batch, 1]."""
    if args.greedy or args.temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / max(args.temperature, 1e-5)
    if getattr(args, "top_k", 0) > 0:
        v, _ = torch.topk(logits, args.top_k)
        logits[logits < v[:, -1:]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ════════════════════════════════════════════════════════════════════════════
# Generation loop
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def generate(model, tokenizer, args):
    """
    Auto-regressive generation with KV cache (StaticInferenceContext).

    Phase 1 – Prefill : forward entire prompt at once, collect logits for next token.
    Phase 2 – Decode  : forward one token per step, reusing KV cache.

    With MemRift hooks, each forward pass decompresses exactly the layers being
    computed and releases them immediately after, keeping GPU peak ~1 layer.
    """
    device = torch.cuda.current_device()

    # ── Build prompt (synthetic for benchmarking, or tokenise text) ──
    if getattr(args, "prompt_tokens", 0) and args.prompt_tokens > 0:
        # Synthetic prompt of exactly N tokens for max-context benchmarking.
        # Use a fixed safe token id (avoids EOS / special tokens).
        n = int(args.prompt_tokens)
        safe_id = 100 % getattr(args, "padded_vocab_size", 32000)
        prompt_tokens = [safe_id] * n
        print_rank_0(f"[generate] synthetic prompt of {n} tokens (id={safe_id})")
    else:
        prompt_tokens = tokenizer.tokenize(args.prompt)
    if not prompt_tokens:
        print_rank_0("[generate] empty tokenisation, aborting")
        return
    prompt_len = len(prompt_tokens)
    max_seq = prompt_len + args.max_new_tokens

    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # ── StaticInferenceContext (KV cache) ──
    ctx = InferenceParams(max_batch_size=1, max_sequence_length=max_seq)
    ctx.enable_prefill_mode()

    print_rank_0(f"\nPrompt ({prompt_len} tokens): {args.prompt!r}")
    print_rank_0("Generating …")
    t_start = time.perf_counter()

    # ── Phase 1: Prefill ──
    # attention_mask=None: let the flash/TE backend apply causal masking via the
    # model's attn_mask_type + inference_context. Materializing an explicit
    # [N, N] causal mask is O(N^2) memory (~32 GB at N=90k) and is the dominant
    # long-context memory cost, so we avoid it entirely.
    pos_ids = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(tokens, pos_ids, None, inference_context=ctx, runtime_gather_output=True)
    # logits: [batch, seq_len, vocab]  (or last-token only in decode mode)
    next_id = _sample(logits[:, -1, :], args)  # [1, 1]
    tokens = torch.cat([tokens, next_id], dim=-1)

    ctx.sequence_len_offset = prompt_len
    ctx.enable_decode_mode()
    ttft = time.perf_counter() - t_start  # time to first token

    # ── Phase 2: Decode ──
    for step in range(args.max_new_tokens - 1):
        cur_pos = ctx.sequence_len_offset  # 0-indexed position of new token
        cur_tok = tokens[:, -1:]  # [1, 1]
        pos_single = torch.tensor([[cur_pos]], dtype=torch.long, device=device)

        logits = model(cur_tok, pos_single, None, inference_context=ctx, runtime_gather_output=True)
        next_id = _sample(logits[:, -1, :], args)
        tokens = torch.cat([tokens, next_id], dim=-1)
        ctx.sequence_len_offset += 1

        # Stop at EOS
        if next_id.item() == tokenizer.eod:
            break

    # ── Report ──
    elapsed = time.perf_counter() - t_start
    new_tokens = tokens.shape[1] - prompt_len
    try:
        generated_text = tokenizer.detokenize(tokens[0, prompt_len:].tolist())
    except Exception:
        generated_text = str(tokens[0, prompt_len:].tolist())

    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    print_rank_0(f"\nGenerated ({new_tokens} new tokens):\n  {generated_text}")
    print_rank_0(f"\n{'─' * 50}")
    print_rank_0(f"  Time to first token : {ttft * 1000:.1f} ms")
    print_rank_0(f"  Total time          : {elapsed * 1000:.1f} ms")
    print_rank_0(f"  Throughput          : {new_tokens / elapsed:.2f} tokens/s")
    print_rank_0(f"  GPU peak memory     : {peak_gb:.3f} GB")
    print_rank_0(f"{'─' * 50}")


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=_extra_args,
        args_defaults={
            # Tokeniser default
            "tokenizer_type": "HuggingFaceTokenizer",
            # Training-specific required-but-unused defaults
            "train_iters": 1,
            "eval_iters": 0,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "lr": 1e-4,
            "min_lr": 1e-5,
            "lr_decay_style": "cosine",
            "lr_warmup_iters": 0,
            "mock_data": True,
            "split": "1,0,0",
            "no_load_optim": True,
            "no_load_rng": True,
        },
    )

    args = get_args()
    tokenizer = get_tokenizer()

    # ── Build model ──
    model_list = get_model(
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )
    model = model_list[0]
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ── Generate ──
    generate(model, tokenizer, args)

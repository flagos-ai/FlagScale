from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
from transformers import AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

# Import Qwen3 components (dense version)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

from tokenizers import Regex, normalizers

from .configuration_qwen3engram import Qwen3EngramConfig

logger = logging.get_logger(__name__)


class ShortConv(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)])

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))

        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        return y


class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )

        SENTINEL = "\ue000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

        self.lookup_table, self.num_new_token = self._build_lookup_table()
        # Create a tensor version for GPU operations
        self.lookup_table_tensor = torch.from_numpy(self.lookup_table).long()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def _compress(self, input_ids):
        # Keep computation on the same device as input_ids (GPU or CPU)
        if isinstance(input_ids, torch.Tensor):
            # Move lookup table to the same device as input_ids if needed
            if self.lookup_table_tensor.device != input_ids.device:
                self.lookup_table_tensor = self.lookup_table_tensor.to(input_ids.device)
            # Use PyTorch operations to stay on device
            vocab_size = len(self.lookup_table_tensor)
            pos_mask = (input_ids >= 0) & (input_ids < vocab_size)
            out = input_ids.clone()
            valid_ids = input_ids[pos_mask]
            out[pos_mask] = self.lookup_table_tensor[valid_ids]
            return out
        else:
            # Fallback to numpy for non-tensor inputs
            arr = np.asarray(input_ids, dtype=np.int64)
            vocab_size = len(self.lookup_table)
            pos_mask = (arr >= 0) & (arr < vocab_size)
            out = arr.copy()
            valid_ids = arr[pos_mask]
            out[pos_mask] = self.lookup_table[valid_ids]
            return out

    def __call__(self, input_ids):
        return self._compress(input_ids)


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    def __init__(
        self,
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        # Prevent overflow in hash function calculation
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []

                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                for _ in range(num_head):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            shifted = np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id)[
                :, :T
            ]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []

        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids, layer_id):
        # Keep track of original device to return result on same device
        original_device = None
        if isinstance(input_ids, torch.Tensor):
            original_device = input_ids.device

        # Compress the input_ids (stays on original device if tensor)
        input_ids = self.compressed_tokenizer(input_ids)

        # Hash computation still uses numpy, so convert if needed
        if isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.cpu().numpy()
        else:
            input_ids_np = input_ids

        hash_ids = self._get_ngram_hashes(input_ids_np, layer_id=layer_id)

        # Convert back to tensor on original device if input was a tensor
        if original_device is not None:
            hash_ids = torch.from_numpy(hash_ids).to(original_device)

        return hash_ids


class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: list[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        total_N = sum(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        return output


class Qwen3Engram(nn.Module):
    def __init__(self, engram_cfg, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hc_mult = engram_cfg.hc_mult
        self.hidden_size = engram_cfg.hidden_size

        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.emb_size_enngram,
            n_head_per_ngram=engram_cfg.num_engram_heads,
            layer_ids=engram_cfg.engram_layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=[
                x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y
            ],
            D=engram_cfg.emb_size_enngram // engram_cfg.num_engram_heads,
        )
        self.short_conv = ShortConv(
            hidden_size=engram_cfg.hidden_size,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size,
            hc_mult=engram_cfg.hc_mult,
        )
        engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.emb_size_enngram
        self.value_proj = nn.Linear(engram_hidden_size, engram_cfg.hidden_size)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(engram_hidden_size, engram_cfg.hidden_size)
                for _ in range(engram_cfg.hc_mult)
            ]
        )
        self.norm1 = nn.ModuleList(
            [nn.RMSNorm(engram_cfg.hidden_size) for _ in range(engram_cfg.hc_mult)]
        )
        self.norm2 = nn.ModuleList(
            [nn.RMSNorm(engram_cfg.hidden_size) for _ in range(engram_cfg.hc_mult)]
        )

    def forward(self, hidden_states, input_ids):
        """
        hidden_states: [B, L, HC_MULT, D] or [B, L, D]
        input_ids: [B, L]
        """
        output_dim = 4
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.unsqueeze(2)
            output_dim = 3

        hash_input_ids = self.hash_mapping.hash(input_ids, self.layer_id)
        # Convert to tensor if hash returned numpy array
        if not isinstance(hash_input_ids, torch.Tensor):
            hash_input_ids = torch.from_numpy(hash_input_ids).to(input_ids.device)
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

        gates = []
        for hc_idx in range(self.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates, dim=2)
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)

        if output_dim == 3:
            output = output.squeeze(2)
        # Ensure output is contiguous for compatibility with flash attention
        return output.contiguous()


class Qwen3EngramDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3EngramConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.use_engram = False

        self.self_attn = Qwen3Attention(config, layer_idx)

        if layer_idx in config.engram_layer_ids:
            self.engram = Qwen3Engram(config, layer_idx)
            self.use_engram = True

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            input_ids (`torch.LongTensor`, *optional*):
                Input token IDs of shape `(batch, seq_len)`. Required when using Engram layers.
        """
        residual = hidden_states

        # Apply Engram module before attention if enabled
        if self.use_engram:
            engram_output = self.engram(hidden_states, input_ids)
            hidden_states = engram_output + hidden_states

        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected (MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@auto_docstring
class Qwen3EngramPreTrainedModel(PreTrainedModel):
    config: Qwen3EngramConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3EngramDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3EngramDecoderLayer,
        "attentions": Qwen3Attention,
    }


@auto_docstring
class Qwen3EngramModel(Qwen3EngramPreTrainedModel):
    def __init__(self, config: Qwen3EngramConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3EngramDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = (
            create_causal_mask
            if self.config.sliding_window is None
            else create_sliding_window_causal_mask
        )
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                input_ids=input_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class Qwen3EngramForCausalLM(Qwen3EngramPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3EngramModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    config = Qwen3EngramConfig(
        num_hidden_layers=36,
        engram_layer_ids=[2],
        hidden_size=2560,
        initializer_range=0.02,
        num_key_value_heads=8,
        num_attention_heads=32,
        intermediate_size=9728,
        max_position_embeddings=32768,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, trust_remote_code=True)
    model = Qwen3EngramForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total parameters: {total_params:,}")

    # Calculate engram parameters
    engram_params = 0
    for layer_idx in config.engram_layer_ids:
        if hasattr(model.model.layers[layer_idx], "engram"):
            engram_params += sum(
                p.numel() for p in model.model.layers[layer_idx].engram.parameters()
            )
    print(f"Engram parameters: {engram_params:,}")

    model = model.to("cuda")

    prompts = ["Hello, my dog is cute", "Hello, my cat is really cute, hahahahaha"]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    outputs = model(**batch)
    print(outputs)

    import os

    save_dir = "./saved_qwen3engram"
    os.makedirs(save_dir, exist_ok=True)

    # Save config to config.json
    config.save_pretrained(save_dir)

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    # Save model weights
    model.save_pretrained(save_dir)

    # Save modeling file
    import shutil

    model_code_file = os.path.abspath(__file__)
    shutil.copy2(model_code_file, os.path.join(save_dir, os.path.basename(model_code_file)))

    # Save config source code
    import inspect

    import configuration_qwen3engram as qwen_config_mod

    config_src = inspect.getfile(qwen_config_mod)
    shutil.copy2(config_src, os.path.join(save_dir, os.path.basename(config_src)))

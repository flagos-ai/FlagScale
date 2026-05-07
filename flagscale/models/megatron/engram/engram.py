## built-in
import copy
import math

## third-party
import torch
import torch.nn as nn

from megatron.plugin.platform import get_platform

from .engram_config import EngramConfig
from .multi_head_embedding import MultiHeadEmbedding

## engram
from .ngram_hash import get_or_create_hash_mapping
from .short_conv import ShortConv


## Megatron
from megatron.core.transformer.utils import sharded_state_dict_default

cur_platform = get_platform()

class Engram(nn.Module):
    def __init__(self, engram_cfg: EngramConfig, layer_id):
        super().__init__()
        assert engram_cfg.engram_hc_mult == 1, (
            "Engram do not support hyper-connection now, engram_hc_mult must be 1"
        )
        self.engram_cfg = engram_cfg
        self.backbone_config = copy.deepcopy(engram_cfg)

        self.layer_id = layer_id
        global_hash_mapping = get_or_create_hash_mapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.engram_layer_ids,
            tokenizer_name_or_path=engram_cfg.engram_tokenizer_name_or_path,
            pad_id=engram_cfg.engram_pad_id,
            seed=engram_cfg.engram_seed,
        )
        self.memory = MultiHeadEmbedding(
            engram_cfg,
            list_of_N=[
                x for y in global_hash_mapping.vocab_size_across_layers[self.layer_id] for x in y
            ],
            D=engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        self.embedding_cache = None  # Cache for pre-computed embeddings
        self.embedding_stream = None  # Stream for pre-computing embeddings
        if cur_platform.is_available():
            self.embedding_stream = cur_platform.Stream()
        self.short_conv = ShortConv(
            hidden_size=self.backbone_config.hidden_size,
            kernel_size=engram_cfg.engram_kernel_size,
            dilation=engram_cfg.max_ngram_size,
            hc_mult=self.backbone_config.engram_hc_mult,
        )
        engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, self.backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(engram_hidden_size, self.backbone_config.hidden_size)
                for _ in range(self.backbone_config.engram_hc_mult)
            ]
        )
        self.norm1 = nn.ModuleList(
            [
                nn.RMSNorm(self.backbone_config.hidden_size)
                for _ in range(self.backbone_config.engram_hc_mult)
            ]
        )
        self.norm2 = nn.ModuleList(
            [
                nn.RMSNorm(self.backbone_config.hidden_size)
                for _ in range(self.backbone_config.engram_hc_mult)
            ]
        )

    def forward(self, hidden_states, hash_input_ids):
        """
        # hidden_states: [L, B, HC_MULT, D]
        hidden_states: [L, B, D] # do not support hyper-connection now, hc_mult must be 1
        input_ids: [B, L]

        # return: [L, B, HC_MULT, D]
        return: [L, B, D] # do not support hyper-connection now, hc_mult must be 1
        """
        assert hash_input_ids is not None, "Hash input ids can not be None for EngramModel"
        # [B, L, N_GRAM * N_HEADS_PER_GRAM]
        # fake hyper-connection
        hidden_states = hidden_states.unsqueeze(2)
        if self.embedding_cache is not None:
            embeddings, embedding_event = self.embedding_cache
            if embedding_event is not None:
                cur_platform.current_stream().wait_event(embedding_event)  # Ensure pre-computed embeddings are ready
            self.embedding_cache = None  # Clear cache after use
            del embedding_event  # Free the event
        else:
            embeddings = self.memory(hash_input_ids).flatten(start_dim=-2)
        # [L/tp_size, B, N_GRAM * N_HEADS_PER_GRAM, N_EMBED_PER_GRAM // N_HEADS_PER_GRAM]
        # [L/tp_size, B, N_GRAM * N_EMBED_PER_NGRAM]

        # Pre-compute scaling factor for efficiency
        scale = 1.0 / math.sqrt(self.backbone_config.hidden_size)
        gates = []
        for hc_idx in range(self.backbone_config.engram_hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            # [L/tp_size, B, HIDDEN_SIZE]
            normed_key = self.norm1[hc_idx](key)

            query = hidden_states[:, :, hc_idx, :]
            # [L, B, HIDDEN_SIZE]
            normed_query = self.norm2[hc_idx](query)

            # Compute scaled dot product similarity
            gate = torch.sum(normed_key * normed_query, dim=-1, keepdim=True) * scale
            # Apply smooth absolute value transformation: sign(x) * sqrt(|x|)
            # This is equivalent to: abs().clamp_min(1e-6).sqrt() * sign()
            gate = torch.sign(gate) * torch.sqrt(torch.abs(gate).clamp_min(1e-6))
            gate = torch.sigmoid(gate)
            # [L, B, 1]

            gates.append(gate)
        gates = torch.stack(gates, dim=2)
        # [L, B, HC_MULT, 1]

        value = gates * self.value_proj(embeddings).unsqueeze(2)
        # [L, B, HC_MULT, HIDDEN_SIZE]
        output = value + self.short_conv(value)
        # [L, B, HC_MULT, HIDDEN_SIZE]

        # re-fake hyper-connection
        assert output.shape[2] == 1, "Engram do not support hyper-connection now, hc_mult must be 1"
        output = output.squeeze(2)

        return output

    def pre_compute_embedding(self, input_ids: torch.Tensor):
        """
        Pre-compute the multi-head embedding for the given input IDs.
        This can be called before the forward pass to warm up the embedding cache.
        """
        assert input_ids is not None, "Input ids can not be None for EngramModel"
        self.embedding_stream.synchronize()  # Ensure previous computations on the stream are finished
        with cur_platform.stream(self.embedding_stream):
            embedding_result = self.memory(input_ids).flatten(start_dim=-2)
        embedding_event = cur_platform.Event()
        embedding_event.record(self.embedding_stream)
        self.embedding_cache = (embedding_result, embedding_event)
        
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ):
        sharded_dict = {}
        memory_prefix = f"{prefix}memory."
        sharded_dict.update(self.memory.sharded_state_dict(memory_prefix, sharded_offsets, metadata))
        conv_prefix = f"{prefix}short_conv."
        sharded_dict.update(sharded_state_dict_default(self.short_conv, conv_prefix, sharded_offsets, metadata))
        value_proj_prefix = f"{prefix}value_proj."
        sharded_dict.update(sharded_state_dict_default(self.value_proj, value_proj_prefix, sharded_offsets, metadata))
        key_projs_prefix = f"{prefix}key_projs."
        sharded_dict.update(sharded_state_dict_default(self.key_projs, key_projs_prefix, sharded_offsets, metadata))
        norm1_prefix = f"{prefix}norm1."
        sharded_dict.update(sharded_state_dict_default(self.norm1, norm1_prefix, sharded_offsets, metadata))
        norm2_prefix = f"{prefix}norm2."
        sharded_dict.update(sharded_state_dict_default(self.norm2, norm2_prefix, sharded_offsets, metadata))
        return sharded_dict

## built-in
import copy
import math

## third-party
import torch
import torch.nn as nn

from .engram_config import EngramConfig
from .multi_head_embedding import MultiHeadEmbedding

## engram
from .ngram_hash import get_or_create_hash_mapping
from .short_conv import ShortConv


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
        self.multi_head_embedding = MultiHeadEmbedding(
            engram_cfg,
            list_of_N=[
                x for y in global_hash_mapping.vocab_size_across_layers[self.layer_id] for x in y
            ],
            D=engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
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

        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
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

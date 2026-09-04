"""Declarative source inventory rules for the BAGEL checkpoint.

This module intentionally contains no Megatron runtime imports.  It is usable
on login/CPU nodes to audit a released checkpoint before distributed tensor
materialization is attempted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Iterable, Match, Optional, Pattern


class MappingKind(str, Enum):
    """How a source tensor participates in target materialization."""

    DIRECT = "direct"
    QKV = "qkv"
    GATED_MLP = "gated_mlp"
    REGENERATE = "regenerate"


@dataclass(frozen=True)
class MappingRule:
    """A regex rule assigning a source key to a conversion operation."""

    name: str
    pattern: Pattern[str]
    kind: MappingKind
    target_template: Optional[str]
    group_template: Optional[str] = None
    note: str = ""

    @classmethod
    def create(
        cls,
        name: str,
        pattern: str,
        kind: MappingKind,
        target_template: Optional[str],
        *,
        group_template: Optional[str] = None,
        note: str = "",
    ) -> "MappingRule":
        return cls(name, re.compile(pattern), kind, target_template, group_template, note)

    def match(self, key: str) -> Optional[Match[str]]:
        return self.pattern.fullmatch(key)

    def target(self, match: Match[str]) -> Optional[str]:
        return _format_match(self.target_template, match)

    def group(self, match: Match[str]) -> Optional[str]:
        return _format_match(self.group_template, match)


def _format_match(template: Optional[str], match: Match[str]) -> Optional[str]:
    if template is None:
        return None
    values = {name: value or "" for name, value in match.groupdict().items()}
    return template.format(**values)


@dataclass(frozen=True)
class ResolvedMapping:
    source: str
    rule: str
    kind: MappingKind
    target: Optional[str]
    group: Optional[str]
    note: str


class MappingRegistry:
    """Ordered mapping rules with ambiguity detection."""

    def __init__(self, rules: Iterable[MappingRule]):
        self.rules = tuple(rules)

    def resolve(self, key: str) -> Optional[ResolvedMapping]:
        matches = [(rule, rule.match(key)) for rule in self.rules]
        matches = [(rule, match) for rule, match in matches if match is not None]
        if not matches:
            return None
        if len(matches) != 1:
            names = ", ".join(rule.name for rule, _ in matches)
            raise ValueError(f"ambiguous mapping for {key!r}: {names}")
        rule, match = matches[0]
        assert match is not None
        return ResolvedMapping(
            source=key,
            rule=rule.name,
            kind=rule.kind,
            target=rule.target(match),
            group=rule.group(match),
            note=rule.note,
        )


@lru_cache(maxsize=1)
def build_bagel_registry() -> MappingRegistry:
    """Build the released BAGEL-7B-MoT source-key inventory registry.

    Target names are the current FlagScale BagelModel state-dict names. Keeping
    source recognition and target generation together prevents phase drift.
    """

    layer = r"(?P<layer>\d+)"
    branch = r"(?:_moe(?P<branch>_gen))?"
    wb = r"(?P<param>weight|bias)"
    rules = [
        MappingRule.create(
            "position_embedding_regenerate",
            r"(?P<name>vit_pos_embed|latent_pos_embed)\.pos_embed",
            MappingKind.REGENERATE,
            "{name}.pos_embed",
            note="Upstream loader regenerates this table from target configuration.",
        ),
        MappingRule.create(
            "language_embedding",
            r"language_model\.model\.embed_tokens\.weight",
            MappingKind.DIRECT,
            "language_model.embedding.word_embeddings.weight",
        ),
        MappingRule.create(
            "language_output",
            r"language_model\.lm_head\.weight",
            MappingKind.DIRECT,
            "output_layer.weight",
        ),
        MappingRule.create(
            "language_final_norm",
            rf"language_model\.model\.norm{branch}\.weight",
            MappingKind.DIRECT,
            "final_layernorm{branch}.weight",
        ),
        MappingRule.create(
            "language_input_norm",
            rf"language_model\.model\.layers\.{layer}\.input_layernorm{branch}\.weight",
            MappingKind.DIRECT,
            "language_model.decoder.layers.{layer}.input_layernorm{branch}.weight",
        ),
        MappingRule.create(
            "language_post_attention_norm",
            rf"language_model\.model\.layers\.{layer}\.post_attention_layernorm{branch}\.weight",
            MappingKind.DIRECT,
            "language_model.decoder.layers.{layer}.pre_mlp_layernorm{branch}.weight",
        ),
        MappingRule.create(
            "language_qkv",
            rf"language_model\.model\.layers\.{layer}\.self_attn\.(?P<projection>[qkv])_proj{branch}\.{wb}",
            MappingKind.QKV,
            "language_model.decoder.layers.{layer}.self_attention.linear_qkv{branch}.{param}",
            group_template="language.{layer}.qkv{branch}.{param}",
        ),
        MappingRule.create(
            "language_attention_output",
            rf"language_model\.model\.layers\.{layer}\.self_attn\.o_proj{branch}\.weight",
            MappingKind.DIRECT,
            "language_model.decoder.layers.{layer}.self_attention.linear_proj{branch}.weight",
        ),
        MappingRule.create(
            "language_qk_norm",
            rf"language_model\.model\.layers\.{layer}\.self_attn\.(?P<qk>[qk])_norm{branch}\.weight",
            MappingKind.DIRECT,
            "language_model.decoder.layers.{layer}.self_attention.{qk}_layernorm{branch}.weight",
        ),
        MappingRule.create(
            "language_gated_mlp",
            rf"language_model\.model\.layers\.{layer}\.mlp{branch}\.(?P<projection>gate|up)_proj\.weight",
            MappingKind.GATED_MLP,
            "language_model.decoder.layers.{layer}.mlp{branch}.linear_fc1.weight",
            group_template="language.{layer}.gated_mlp{branch}.weight",
        ),
        MappingRule.create(
            "language_mlp_output",
            rf"language_model\.model\.layers\.{layer}\.mlp{branch}\.down_proj\.weight",
            MappingKind.DIRECT,
            "language_model.decoder.layers.{layer}.mlp{branch}.linear_fc2.weight",
        ),
        MappingRule.create(
            "connector",
            r"connector\.fc(?P<index>[12])\.(?P<param>weight|bias)",
            MappingKind.DIRECT,
            "connector.encoder.linear_fc{index}.{param}",
        ),
        MappingRule.create(
            "generation_adapter",
            r"(?P<name>vae2llm|llm2vae)\.(?P<param>weight|bias)",
            MappingKind.DIRECT,
            "{name}.{param}",
        ),
        MappingRule.create(
            "timestep_mlp",
            r"time_embedder\.mlp\.(?P<index>[02])\.(?P<param>weight|bias)",
            MappingKind.DIRECT,
            "time_embedder.mlp.{index}.{param}",
        ),
        MappingRule.create(
            "vit_patch_embedding",
            r"vit_model\.vision_model\.embeddings\.patch_embedding\.(?P<param>weight|bias)",
            MappingKind.DIRECT,
            "vision_model.patch_embedding.{param}",
        ),
        MappingRule.create(
            "vit_position_embedding",
            r"vit_model\.vision_model\.embeddings\.position_embedding\.weight",
            MappingKind.DIRECT,
            "vision_model.position_embeddings.weight",
        ),
        MappingRule.create(
            "vit_qkv",
            rf"vit_model\.vision_model\.encoder\.layers\.{layer}\.self_attn\.(?P<projection>[qkv])_proj\.{wb}",
            MappingKind.QKV,
            "vision_model.decoder.layers.{layer}.self_attention.linear_qkv.{param}",
            group_template="vision.{layer}.qkv.{param}",
        ),
        MappingRule.create(
            "vit_attention_output",
            rf"vit_model\.vision_model\.encoder\.layers\.{layer}\.self_attn\.out_proj\.{wb}",
            MappingKind.DIRECT,
            "vision_model.decoder.layers.{layer}.self_attention.linear_proj.{param}",
        ),
        MappingRule.create(
            "vit_input_norm",
            rf"vit_model\.vision_model\.encoder\.layers\.{layer}\.layer_norm1\.{wb}",
            MappingKind.DIRECT,
            "vision_model.decoder.layers.{layer}.self_attention.linear_qkv.layer_norm_{param}",
        ),
        MappingRule.create(
            "vit_pre_mlp_norm",
            rf"vit_model\.vision_model\.encoder\.layers\.{layer}\.layer_norm2\.{wb}",
            MappingKind.DIRECT,
            "vision_model.decoder.layers.{layer}.mlp.linear_fc1.layer_norm_{param}",
        ),
        MappingRule.create(
            "vit_mlp",
            rf"vit_model\.vision_model\.encoder\.layers\.{layer}\.mlp\.fc(?P<index>[12])\.{wb}",
            MappingKind.DIRECT,
            "vision_model.decoder.layers.{layer}.mlp.linear_fc{index}.{param}",
        ),
        MappingRule.create(
            "vit_post_norm",
            r"vit_model\.vision_model\.post_layernorm\.(?P<param>weight|bias)",
            MappingKind.DIRECT,
            "vision_model.ln_post.{param}",
        ),
        MappingRule.create(
            "vae",
            r"(?P<path>(?:encoder|decoder|quant_conv|post_quant_conv)\..+)",
            MappingKind.DIRECT,
            "vae_model.{path}",
        ),
    ]
    return MappingRegistry(rules)


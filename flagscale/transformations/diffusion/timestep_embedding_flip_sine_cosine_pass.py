import torch

from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from flagscale.compilation.inductor_pass import InductorPass
from flagscale.runner.utils import logger

aten = torch.ops.aten
_timestep_embedding_flip_sine_cosine_matcher = PatternMatcherPass(
    pass_name="timestep_embedding_flip_sine_cosine_matcher"
)

# Shared inputs we want to capture once and reuse
base = Arg()  # %mul_17 feeding both sin and cos
split_idx = Arg()  # %floordiv reused by both slices

inner_cat = CallFunction(
    aten.cat.default,
    [CallFunction(aten.sin.default, base), CallFunction(aten.cos.default, base)],
    -1,
    _users=2,  # The cat feeds both slices
)
slice_hi = CallFunction(aten.slice.Tensor, inner_cat, 1, split_idx, 9223372036854775807)
slice_lo = CallFunction(aten.slice.Tensor, inner_cat, 1, 0, split_idx)
pattern = CallFunction(aten.cat.default, [slice_hi, slice_lo], -1)


@register_graph_pattern(pattern, pass_dict=_timestep_embedding_flip_sine_cosine_matcher)
def _rewrite_timestep_embedding_flip_sine_cosine(match: Match, base, split_idx) -> None:
    logger.debug(f"Applying TimestepEmbeddingFlipSineCosinePattern at {match.nodes}")

    sin_node, cos_node, inner_cat, slice_hi, slice_lo, cat_node = match.nodes

    graph = cat_node.graph

    with graph.inserting_before(cat_node):
        new_cat = graph.call_function(torch.ops.aten.cat.default, args=([cos_node, sin_node], -1))
        new_cat.meta.update(cat_node.meta)

    cat_node.replace_all_uses_with(new_cat)

    for node in (slice_hi, slice_lo):
        if len(node.users) == 0 and node in graph.nodes:
            graph.erase_node(node)

    if len(inner_cat.users) == 0 and inner_cat in graph.nodes:
        graph.erase_node(inner_cat)

    if len(cat_node.users) == 0 and cat_node in graph.nodes:
        graph.erase_node(cat_node)

    graph.eliminate_dead_code()


class TimestepEmbeddingFlipSineCosinePass(InductorPass):
    """
    A pass to rewrite the following code snippet from [diffusers](https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/models/embeddings.py#L69):
    ```
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    ```
    to the following code snippet:
    ```
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    This pass serves only as a demonstration, no significant performance improvement is expected.
    Also, we could register the pattern to post grad matcher pass, in that case, this class would be unnecessary.
    """

    def __call__(self, graph: torch.fx.Graph) -> None:
        _timestep_embedding_flip_sine_cosine_matcher.apply(graph)

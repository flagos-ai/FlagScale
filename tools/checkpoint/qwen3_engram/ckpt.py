import sys

sys.path.append("..")
from mistral.ckpt import *
from mixtral.ckpt import _get_parallel_size


def get_engram_ckpt(message, models, layer_id, args):
    print("engram ckpt conversion, get engram ckpt")
    tp_size, _, _, _ = _get_parallel_size(args)

    # parallel tensor
    multi_head_embedding_weight = []

    # non-parallel tensor
    short_conv_conv_weight = None
    short_conv_norms_weight = []  # consider hc_mult
    value_proj_weight = None
    keys_proj_weight = []  # consider hc_mult
    norms1_weight = []
    norms2_weight = []

    assert len(models) == tp_size
    for model in models:
        engram_module = model.decoder.layers[layer_id].engram
        multi_head_embedding_weight.append(engram_module.multi_head_embedding.embedding.weight.data)

        short_conv_conv_weight = engram_module.short_conv.conv.weight.data
        value_proj_weight = engram_module.value_proj.weight.data

        for hc_idx in range(args.engram_hc_mult):
            short_conv_norms_weight.append(engram_module.short_conv.norms[hc_idx].weight.data)
            keys_proj_weight.append(engram_module.key_projs[hc_idx].weight.data)
            norms1_weight.append(engram_module.norm1[hc_idx].weight.data)
            norms2_weight.append(engram_module.norm2[hc_idx].weight.data)

    message["multi_head_embedding_weight"] = torch.cat(multi_head_embedding_weight, dim=0)
    message["short_conv_conv_weight"] = short_conv_conv_weight
    message["value_proj_weight"] = value_proj_weight
    message["short_conv_norms_weight"] = short_conv_norms_weight
    message["keys_proj_weight"] = keys_proj_weight
    message["norms1_weight"] = norms1_weight
    message["norms2_weight"] = norms2_weight


def set_hf_engram_ckpt(message, model, layer_id, md, args):
    print("engram ckpt conversion, set hf engram ckpt")
    hf_engram_module = model.model.layers[layer_id].engram

    # copy multi-head-embedding
    orig_multi_head_embedding_weight = message.pop("multi_head_embedding_weight")

    # re-pad, cut tp padding
    orig_total_N = hf_engram_module.multi_head_embedding.embedding.weight.shape[0]

    hf_engram_module.multi_head_embedding.embedding.weight.data.copy_(
        orig_multi_head_embedding_weight[:orig_total_N, :]
    )

    # copy short conv.conv and value_proj
    hf_engram_module.short_conv.conv.weight.data.copy_(message.pop("short_conv_conv_weight"))
    hf_engram_module.value_proj.weight.data.copy_(message.pop("value_proj_weight"))

    # copy short conv.norms and key_projs and norms1 and norms2
    for hc_idx in range(args.engram_hc_mult):
        hf_engram_module.short_conv.norms[hc_idx].weight.data.copy_(
            message.pop("short_conv_norms_weight")[hc_idx]
        )
        hf_engram_module.key_projs[hc_idx].weight.data.copy_(
            message.pop("keys_proj_weight")[hc_idx]
        )
        hf_engram_module.norm1[hc_idx].weight.data.copy_(message.pop("norms1_weight")[hc_idx])
        hf_engram_module.norm2[hc_idx].weight.data.copy_(message.pop("norms2_weight")[hc_idx])

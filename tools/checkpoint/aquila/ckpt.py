import sys

import torch

sys.path.append("..")
from mixtral.ckpt import (  # noqa: F401
    get_embedding_ckpt,
    get_final_norm_ckpt,
    get_hf_attn_ckpt,
    get_output_layer_ckpt,
    set_embedding_ckpt,
    set_final_norm_ckpt,
    set_hf_attn_ckpt,
    set_hf_embedding_ckpt,
    set_hf_final_norm_ckpt,
    set_hf_output_layer_ckpt,
    set_output_layer_ckpt,
)


def get_hf_mlp_ckpt(message, model, layer_id, args):
    assert args.swiglu is True

    tf_layer = model.model.layers[layer_id]
    message["mlp l0 weight W"] = tf_layer.mlp.gate_proj.weight.data
    message["mlp l0 weight V"] = tf_layer.mlp.up_proj.weight.data
    message["mlp l1 weight"] = tf_layer.mlp.down_proj.weight.data

    if args.add_bias_linear:
        message["mlp l0 bias W"] = tf_layer.mlp.gate_proj.bias.data
        message["mlp l0 bias V"] = tf_layer.mlp.up_proj.bias.data
        message["mlp l1 bias"] = tf_layer.mlp.down_proj.bias.data


def set_hf_mlp_ckpt(message, model, layer_id, md, args):
    assert args.swiglu is True

    tf_layer = model.model.layers[layer_id]
    tf_layer.mlp.gate_proj.weight.data.copy_(message.pop("mlp l0 weight W"))
    tf_layer.mlp.up_proj.weight.data.copy_(message.pop("mlp l0 weight V"))
    tf_layer.mlp.down_proj.weight.data.copy_(message.pop("mlp l1 weight"))

    if md.add_bias_linear:
        tf_layer.mlp.gate_proj.bias.data.copy_(message.pop("mlp l0 bias W"))
        tf_layer.mlp.up_proj.bias.data.copy_(message.pop("mlp l0 bias V"))
        tf_layer.mlp.down_proj.bias.data.copy_(message.pop("mlp l1 bias"))


def _get_parallel_size(args):
    assert args.expert_model_parallel_size == 1
    return (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.expert_model_parallel_size,
        args.virtual_pipeline_model_parallel_size or 1,
    )


def get_attn_ckpt(message, models, layer_id, args):
    tp_size, _, _, _ = _get_parallel_size(args)

    # parallel tensor
    qkv_weight = []
    qkv_bias = []
    proj_weight = []
    # non-parallel tensor
    proj_bias = None
    input_norm_weight = None
    input_norm_bias = None
    post_norm_weight = None
    post_norm_bias = None

    assert len(models) == tp_size
    for model in models:
        tf_layer = model.decoder.layers[layer_id]
        # weight
        qkv_weight.append(tf_layer.self_attention.linear_qkv.weight.data)
        proj_weight.append(tf_layer.self_attention.linear_proj.weight.data)
        input_norm_weight = tf_layer.self_attention.linear_qkv.layer_norm_weight.data
        post_norm_weight = tf_layer.mlp.linear_fc1.layer_norm_weight.data
        # bias
        if args.norm_has_bias:
            input_norm_bias = tf_layer.self_attention.linear_qkv.layer_norm_bias.data
            post_norm_bias = tf_layer.mlp.linear_fc1.layer_norm_bias.data
        if args.add_qkv_bias or args.add_bias_linear:
            qkv_bias.append(tf_layer.self_attention.linear_qkv.bias.data)
        if args.add_bias_linear:
            proj_bias = tf_layer.self_attention.linear_proj.bias.data

    # weight
    message["qkv weight"] = torch.cat(qkv_weight, dim=0)
    message["proj weight"] = torch.cat(proj_weight, dim=1)
    message["input norm weight"] = input_norm_weight
    message["post norm weight"] = post_norm_weight
    # bias
    if args.norm_has_bias:
        message["input norm bias"] = input_norm_bias
        message["post norm bias"] = post_norm_bias
    if args.add_qkv_bias or args.add_bias_linear:
        message["qkv bias"] = torch.cat(qkv_bias, dim=0)
    if args.add_bias_linear:
        message["proj bias"] = proj_bias


def get_mlp_ckpt(message, models, layer_id, args):
    tp_size, _, _, _ = _get_parallel_size(args)

    # parallel tensor
    l0_weight = []
    l0_bias = []
    l1_weight = []
    # non-parallel tensor
    l1_bias = None

    assert len(models) == tp_size
    for model in models:
        tf_layer = model.decoder.layers[layer_id]
        # weight
        l0_weight.append(tf_layer.mlp.linear_fc1.weight.data)
        l1_weight.append(tf_layer.mlp.linear_fc2.weight.data)
        # bias
        if args.add_bias_linear:
            l0_bias.append(tf_layer.mlp.linear_fc1.bias.data)
            l1_bias = tf_layer.mlp.linear_fc2.bias.data

    # weight
    message["mlp l1 weight"] = torch.cat(l1_weight, dim=1)
    if args.swiglu:
        for tp_rank in range(tp_size):
            l0_weight[tp_rank] = torch.chunk(l0_weight[tp_rank], 2, dim=0)
        message["mlp l0 weight W"] = torch.cat([w[0] for w in l0_weight], dim=0)
        message["mlp l0 weight V"] = torch.cat([w[1] for w in l0_weight], dim=0)
    else:
        message["mlp l0 weight"] = torch.cat(l0_weight, dim=0)
    # bias
    if args.add_bias_linear:
        message["mlp l1 bias"] = l1_bias
        if args.swiglu:
            for tp_rank in range(tp_size):
                l0_bias[tp_rank] = torch.chunk(l0_bias[tp_rank], 2, dim=0)
            message["mlp l0 bias W"] = torch.cat([b[0] for b in l0_bias], dim=0)
            message["mlp l0 bias V"] = torch.cat([b[1] for b in l0_bias], dim=0)
        else:
            message["mlp l0 bias"] = torch.cat(l0_bias, dim=0)


def set_attn_ckpt(message, models, layer_id, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)

    # weight
    qkv_weight = torch.chunk(message.pop("qkv weight"), tp_size, dim=0)
    proj_weight = torch.chunk(message.pop("proj weight"), tp_size, dim=1)
    input_norm_weight = message.pop("input norm weight")
    post_norm_weight = message.pop("post norm weight")
    # bias
    if md.norm_has_bias:
        input_norm_bias = message.pop("input norm bias")
        post_norm_bias = message.pop("post norm bias")
    if md.add_qkv_bias or md.add_bias_linear:
        qkv_bias = torch.chunk(message.pop("qkv bias"), tp_size, dim=0)
    if md.add_bias_linear:
        proj_bias = message.pop("proj bias")

    # set data to transformer layer's self-attention
    for tp_rank, model in enumerate(models):
        tf_layer = model.decoder.layers[layer_id]
        tf_layer.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
        tf_layer.self_attention.linear_proj.weight.data.copy_(proj_weight[tp_rank])
        tf_layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(input_norm_weight)
        tf_layer.mlp.linear_fc1.layer_norm_weight.data.copy_(post_norm_weight)
        if md.norm_has_bias:
            tf_layer.self_attention.linear_qkv.layer_norm_bias.data.copy_(input_norm_bias)
            tf_layer.mlp.linear_fc1.layer_norm_bias.data.copy(post_norm_bias)
        if md.add_qkv_bias or md.add_bias_linear:
            tf_layer.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])
        if md.add_bias_linear:
            tf_layer.self_attention.linear_proj.bias.data.copy_(proj_bias)


def set_mlp_ckpt(message, models, layer_id, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)

    # weight
    l1_weight = torch.chunk(message.pop("mlp l1 weight"), tp_size, dim=1)
    if md.swiglu:
        l0_weight_W = torch.chunk(message.pop("mlp l0 weight W"), tp_size, dim=0)
        l0_weight_V = torch.chunk(message.pop("mlp l0 weight V"), tp_size, dim=0)
        l0_weight = [torch.cat(weights, dim=0) for weights in zip(l0_weight_W, l0_weight_V)]
    else:
        l0_weight = torch.chunk(message.pop("mlp l0 weight"), tp_size, dim=0)
    # bias
    if md.add_bias_linear:
        l1_bias = message.pop("mlp l1 bias")
        if md.swiglu:
            l0_bias_W = torch.chunk(message.pop("mlp l0 bias W"), tp_size, dim=0)
            l0_bias_V = torch.chunk(message.pop("mlp l0 bias V"), tp_size, dim=0)
            l0_bias = [torch.cat(bias, dim=0) for bias in zip(l0_bias_W, l0_bias_V)]
        else:
            l0_bias = torch.chunk(message.pop("mlp l0 bias"), tp_size, dim=0)

    # set data to transformer layer for mlp
    for tp_rank, model in enumerate(models):
        tf_layer = model.decoder.layers[layer_id]
        tf_layer.mlp.linear_fc1.weight.data.copy_(l0_weight[tp_rank])
        tf_layer.mlp.linear_fc2.weight.data.copy_(l1_weight[tp_rank])

        if md.add_bias_linear:
            tf_layer.mlp.linear_fc1.bias.data.copy_(l0_bias[tp_rank])
            tf_layer.mlp.linear_fc2.bias.data.copy_(l1_bias)

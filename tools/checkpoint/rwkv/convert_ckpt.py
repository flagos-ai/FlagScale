import argparse

import torch


def deepspeed_to_megatron(deepspeed_ckpt, megatron_ckpt):
    state_dict = torch.load(deepspeed_ckpt, map_location="cpu")
    new_state_dict = {}

    mapping = {
        "emb.weight": "model.embeddings.word_embeddings.weight",
        "head.weight": "model.head.weight",
        "ln_out.weight": "model.ln_out.weight",
        "ln_out.bias": "model.ln_out.bias",
    }

    for k, v in state_dict.items():
        new_k = None

        if k in mapping:
            new_k = mapping[k]

        elif k.startswith("blocks."):
            block_id = k.split(".")[1]
            suffix = ".".join(k.split(".")[2:])
            new_k = f"model.blocks.block_{block_id}.{suffix}"

        if new_k is None:
            print(f"[WARN] Unmapped parameter: {k}")
            continue

        new_state_dict[new_k] = v

    torch.save(new_state_dict, megatron_ckpt)
    print(f"[INFO] DeepSpeed → Megatron conversion completed: {deepspeed_ckpt} → {megatron_ckpt}")


def megatron_to_deepspeed(megatron_ckpt, deepspeed_ckpt):
    state_dict = torch.load(megatron_ckpt, map_location="cpu")
    new_state_dict = {}

    mapping = {
        "model.embeddings.word_embeddings.weight": "emb.weight",
        "model.head.weight": "head.weight",
        "model.ln_out.weight": "ln_out.weight",
        "model.ln_out.bias": "ln_out.bias",
    }

    for k, v in state_dict.items():
        new_k = None

        if k in mapping:
            new_k = mapping[k]

        elif k.startswith("model.blocks.block_"):
            block_id = k.split(".")[2].replace("block_", "")
            suffix = ".".join(k.split(".")[3:])
            new_k = f"blocks.{block_id}.{suffix}"

        if new_k is None:
            print(f"[WARN] Unmapped parameter: {k}")
            continue

        new_state_dict[new_k] = v

    torch.save(new_state_dict, deepspeed_ckpt)
    print(f"[INFO] Megatron → DeepSpeed conversion completed: {megatron_ckpt} → {deepspeed_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between DeepSpeed and Megatron checkpoint formats."
    )
    parser.add_argument("src", type=str, help="source checkpoint file (.pt)")
    parser.add_argument("dst", type=str, help="destination checkpoint file (.pt)")
    parser.add_argument(
        "--direction",
        choices=["ds2mg", "mg2ds"],
        required=True,
        help="conversion direction: ds2mg (DeepSpeed → Megatron) or mg2ds (Megatron → DeepSpeed)",
    )

    args = parser.parse_args()

    if args.direction == "ds2mg":
        deepspeed_to_megatron(args.src, args.dst)
    else:
        megatron_to_deepspeed(args.src, args.dst)

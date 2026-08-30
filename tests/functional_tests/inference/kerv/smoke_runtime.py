# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Weight-free CUDA smoke for the native KERV control path."""

import argparse

import torch

from flagscale.models.kerv import KERVConfig, KERVRuntime


def _draft(paths: torch.LongTensor) -> torch.Tensor:
    logits = torch.full((paths.shape[0], 10), -20.0, device=paths.device)
    logits[:, 1] = 20.0
    logits[:, 2] = 19.0
    return logits


def _verify(tree) -> torch.Tensor:
    paths = tree.candidate_paths
    logits = torch.full((paths.shape[0], paths.shape[1], 10), -100.0, device=paths.device)
    for path_index in range(paths.shape[0]):
        for position in range(paths.shape[1] - 1):
            token = int(paths[path_index, position + 1].item())
            logits[path_index, position, token] = 100.0
        logits[path_index, paths.shape[1] - 1, 9] = 100.0
    return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("KERV CUDA functional smoke requires an available CUDA device")

    device = torch.device(args.device)
    runtime = KERVRuntime(
        KERVConfig(
            action_dim=3,
            candidate_depth=2,
            top_k=2,
            max_paths=3,
            accept_threshold=0,
        )
    )
    result = runtime.step(
        _draft,
        _verify,
        root_token=torch.tensor(0, dtype=torch.long, device=device),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    expected = torch.tensor([1, 1, 9], dtype=torch.long, device=device)
    if not torch.equal(result.output_tokens, expected):
        raise AssertionError(
            f"unexpected KERV output: {result.output_tokens.detach().cpu().tolist()}"
        )
    if result.verification.accept_length != 2:
        raise AssertionError(f"unexpected accept length: {result.verification.accept_length}")
    if result.tree.attention_mask.device.type != device.type:
        raise AssertionError("candidate tree did not stay on the requested device")

    print("**************************************************")
    print("KERV native runtime smoke: output=[1, 1, 9], accepted=2, device=cuda")
    print("##################################################")


if __name__ == "__main__":
    main()

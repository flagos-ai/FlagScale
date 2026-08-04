"""Dump trainable (LoRA) initial weights for precision alignment verification.

Usage: Called from train_dreamzero.py after model init, before FSDP2.
Only rank 0 dumps. Saves lora_A and lora_B tensors to a .pt file.
"""
import torch
from pathlib import Path


def dump_trainable_weights(model, output_path: str, rank: int = 0):
    """Dump all trainable parameters (LoRA weights) to a .pt file.
    
    Only called on rank 0 since pre-FSDP2 all ranks have identical weights.
    """
    if rank != 0:
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.detach().cpu().clone()
    
    torch.save(trainable_state, output_path)
    print(f"[ALIGN] Dumped {len(trainable_state)} trainable params to {output_path}")
    for name, tensor in list(trainable_state.items())[:5]:
        print(f"  {name}: shape={tensor.shape}, mean={tensor.float().mean():.6e}, std={tensor.float().std():.6e}")
    if len(trainable_state) > 5:
        print(f"  ... and {len(trainable_state) - 5} more")


def compare_init_weights(path_a: str, path_b: str):
    """Compare two init weight dumps tensor-by-tensor."""
    state_a = torch.load(path_a, map_location="cpu")
    state_b = torch.load(path_b, map_location="cpu")
    
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    
    if keys_a != keys_b:
        print(f"[ALIGN] KEY MISMATCH!")
        print(f"  Only in A: {keys_a - keys_b}")
        print(f"  Only in B: {keys_b - keys_a}")
        return False
    
    all_match = True
    max_diff = 0.0
    for key in sorted(keys_a):
        a = state_a[key].float()
        b = state_b[key].float()
        if a.shape != b.shape:
            print(f"[ALIGN] SHAPE MISMATCH: {key}: {a.shape} vs {b.shape}")
            all_match = False
            continue
        diff = (a - b).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 0:
            print(f"[ALIGN] DIFF: {key}: max_abs_diff={diff:.6e}")
            all_match = False
    
    if all_match:
        print(f"[ALIGN] ✓ ALL {len(keys_a)} trainable params EXACTLY MATCH (max_diff={max_diff:.6e})")
    else:
        print(f"[ALIGN] ✗ MISMATCH DETECTED (max_diff={max_diff:.6e})")
    
    return all_match


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        compare_init_weights(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python dump_init_weights.py <path_a.pt> <path_b.pt>")

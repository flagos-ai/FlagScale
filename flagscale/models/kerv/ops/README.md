# KERV optimized runtime

This directory is the KERV-specific runtime shipped with the FlagScale model
integration. It is intentionally scoped to KERV's batch-one speculative
decoding path and does not register global replacements for unrelated models.

## Runtime integration

KERV imports its runtime through the canonical top-level package
`KERVRuntimeOptimization`. Before invoking an upstream KERV entrypoint,
FlagScale prepends this directory to `PYTHONPATH`. This makes the source
checkout use the implementation versioned with FlagScale and avoids loading a
second operator package from the checkout.

```text
flagscale/models/kerv/ops/
└── KERVRuntimeOptimization/
    ├── embodied_ops/              # 18 torch.library operator interfaces
    ├── adaptive_linear_fusion.py  # packed QKV/Gate-Up and fusion hooks
    ├── adaptive_rms_norm.py       # inference RMSNorm hook
    ├── adaptive_rotary_fusion.py  # Q/K RoPE and resident-cache hook
    ├── fused_logsoftmax_topk.py   # draft LogSoftmax/Top-K path
    ├── rotary_cache.py            # per-forward rotary cache
    └── tree_attention_mask.py     # verification-tree mask path
```

The operator namespace remains `torch.ops.flagos_embodied.*` to preserve the
public KERV contract.

## Operator inventory

The default KERV profile requests 14 interfaces. Three additional interfaces
remain experimental, and one V-cache interface is retained for compatibility
with earlier KERV configurations.

| Operator | Function | Status |
| --- | --- | --- |
| `kerv_silu_mul` | `SiLU(gate) * up` post-GEMM fusion | KERV profile |
| `kerv_add_rms_norm` | residual add and RMSNorm | KERV profile |
| `kerv_verify_accept_control` | token matching, prefix length and best-path reduction | KERV profile |
| `kerv_static_tree_pack` | static-tree token gather and padding | KERV profile |
| `kerv_static_tree_attention` | prefix-and-ancestor attention for a fixed verification tree | KERV profile, A100 Triton route |
| `kerv_action_projection_select` | action-vocabulary projection and argmax | KERV profile |
| `kerv_kv_commit` | resident K/V cache commit | KERV profile |
| `kerv_tree_embed_pack` | candidate gather, padding and embedding lookup | KERV profile |
| `kerv_rope_kv_store` | Q/K RoPE plus resident K/V write | KERV profile, A100 Triton route |
| `kerv_action_verify_accept` | action projection, verification and next-token selection | KERV profile |
| `kerv_draft_action_topk` | action projection, LogSoftmax and deterministic Top-K | KERV profile |
| `kerv_kv_accept_commit` | accepted-path gather and K/V commit | KERV profile |
| `kerv_vision_add_layer_norm` | vision residual add and LayerNorm | KERV profile |
| `kerv_vision_bias_gelu` | vision bias add and GELU | KERV profile |
| `kerv_o_proj_residual_rms_norm` | attention output projection epilogue | Experimental |
| `kerv_down_proj_residual_rms_norm` | MLP down-projection epilogue | Experimental |
| `kerv_logical_kv_commit` | logical cache-address commit | Experimental |
| `kerv_value_cache_store` | V-only cache write | Compatibility; covered by RoPE/KV store in the default profile |

Every interface has an exact PyTorch implementation. Triton is optional at
import time; unsupported devices and layouts use the platform-native
implementation. Experimental interfaces are never selected implicitly.

## A100 selection policy

`backend=auto` selects a Triton implementation only after the real KERV shape
passes a 1.05x local speed threshold. With 100 warmups, 200 timed iterations
and five median windows on an NVIDIA A100-PCIE-40GB, the current automatic
routes are:

| Operator | PyTorch | Triton | Speedup | Correctness |
| --- | ---: | ---: | ---: | --- |
| `kerv_rope_kv_store` | 0.1556 ms | 0.0889 ms | 1.75x | BF16 pass |
| `kerv_static_tree_attention` | 0.3602 ms | 0.0892 ms | 4.04x | BF16 pass |

The other public interfaces remain available to KERV but use their exact
PyTorch route in `auto` mode. A correct interface is not described as an
accelerated kernel until it passes the backend-specific timing threshold.

## Validation

Run all KERV integration and operator tests:

```bash
pytest tests/unit_tests/models/kerv -q
```

Run the A100 microbenchmark:

```bash
python examples/kerv/benchmark_ops.py \
  --device cuda --backend auto \
  --warmup 100 --iterations 200 --repeats 5
```

The tests cover all 18 schemas, CPU implementations, Triton-free imports,
BF16 CUDA correctness, fixed-prefix tree masking, namespace registration, and
the actual FlagScale-to-KERV import precedence.

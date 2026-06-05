import copy
import json
import math
import os
import re
from typing import Any

import torch
from llmcompressor.core import active_session
from llmcompressor.core.session_functions import LifecycleCallbacks
from llmcompressor.pipelines.registry import CalibrationPipeline
from torch.utils.data.dataloader import DataLoader

try:
    from llmcompressor.pipelines.sequential.helpers import get_sequential_targets, match_modules
except ImportError:
    from llmcompressor.pipelines.layer_sequential.helpers import (
        get_sequential_targets,
        match_modules,
    )

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import QuIPModifier

try:
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.quant_args import QuantizationArgs
except ImportError:
    raise ImportError(
        "Could not import quantization functions. Please ensure llmcompressor/compressed-tensors is installed."
    )

try:
    from llmcompressor.modifiers.transform.utils.hadamard import get_hadamard_matrix
except ImportError:

    def get_hadamard_matrix(n, dtype=torch.float32, device="cpu"):
        from scipy.linalg import hadamard

        H = torch.tensor(hadamard(n), dtype=dtype, device=device)
        return H / math.sqrt(n)


@CalibrationPipeline.register("mix_precision_search")
class MixPrecisionPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(
        model: torch.nn.Module, dataloader: DataLoader | None, dataset_args: Any, **kwargs
    ):
        session = active_session()

        session.initialize(model=model)

        modifiers = session.lifecycle.recipe.modifiers

        if dataset_args is None:
            from types import SimpleNamespace

            dataset_args = SimpleNamespace(sequential_targets=None)

        sequential_targets = get_sequential_targets(modifiers, model, dataset_args)
        found_modules = match_modules(model, sequential_targets)

        module_to_name = {m: n for n, m in model.named_modules()}
        layers_to_process = []

        if isinstance(found_modules, dict):
            layers_to_process = list(found_modules.items())
        elif isinstance(found_modules, list):
            for m in found_modules:
                real_name = module_to_name.get(m, "unknown_layer")
                layers_to_process.append((real_name, m))

        if len(layers_to_process) == 0:
            print(">>> [DEBUG] Standard discovery failed (0 layers). Activating Manual Fallback...")
            candidates = [
                "model.layers",
                "model.decoder.layers",
                "transformer.h",
                "layers",
                "blocks",
            ]
            target_module_list = None
            list_name = ""

            for name, module in model.named_modules():
                if any(name.endswith(c) for c in candidates) and isinstance(
                    module, torch.nn.ModuleList
                ):
                    target_module_list = module
                    list_name = name
                    break
            if target_module_list is not None:
                print(
                    f">>> [DEBUG] Manually found ModuleList: {list_name} with {len(target_module_list)} layers."
                )
                for i, layer in enumerate(target_module_list):
                    layer_name = f"{list_name}.{i}"
                    layers_to_process.append((layer_name, layer))
            else:
                print(
                    ">>> [ERROR] Manual Fallback failed: Could not find any Transformer Block List."
                )

        def natural_keys(item):
            text = item[0]
            return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

        sorted_layers = sorted(layers_to_process, key=natural_keys)

        LifecycleCallbacks.calibration_epoch_start()
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(
            f"DEBUG: Processing {len(sorted_layers)} layers with Auto-Search (8-bit vs QuIP-4bit)."
        )

        search_results = []
        all_layer_scores = []  # 新增：收集每层各策略cos sim
        global_quip_targets = []

        # [MOD-1] 扩展搜索候选：加入 W4A16 / W4A16_ASYM
        # 理由：不改变搜索主循环，仅增加候选配置维度（scheme + symmetry）
        candidate_configs = [
            {
                "name": "Std-8bit",
                "bits": 8,
                "quip": False,
                "symmetric": True,
                "group_id": "group_0",
            },
            {
                "name": "QuIP-4bit",
                "bits": 4,
                "quip": True,
                "symmetric": True,
                "group_id": "group_1",
            },
            {"name": "W4A16", "bits": 4, "quip": False, "symmetric": True, "group_id": "group_2"},
            {
                "name": "W4A16_ASYM",
                "bits": 4,
                "quip": False,
                "symmetric": False,
                "group_id": "group_3",
            },
        ]

        ACCEPTANCE_THRESHOLD = 0.008

        for i, (layer_name, layer) in enumerate(sorted_layers):
            best_score = -1.0
            best_config = candidate_configs[1]
            layer_stats = {}

            match = re.search(r"\.(\d+)(?:\.|$)", layer_name)
            real_layer_idx = int(match.group(1)) if match else i

            print(f"\nSearching Layer {real_layer_idx}: {layer_name}")

            for config in candidate_configs:
                bit = config["bits"]
                use_quip = config["quip"]
                name = config["name"]

                _set_layer_bits_only(layer, bit, symmetric=config.get("symmetric", True))

                current_score, func_name, param_bytes = _calculate_layer_metrics(
                    layer, bit, use_quip=use_quip, symmetric=config.get("symmetric", True)
                )

                layer_stats[name] = {
                    "score": current_score,
                    "size": param_bytes,
                    "config": config,
                    "func_name": func_name,
                }

                print(
                    f"  - Testing {name:<10} | Cos Sim: {current_score:.6f} | Size: {param_bytes / 1024 / 1024:.2f} MB | Func: {func_name}"
                )

            layer_score_row = {
                "layer": layer_name,
                "Std-8bit": layer_stats.get("Std-8bit", {}).get("score", float("nan")),
                "QuIP-4bit": layer_stats.get("QuIP-4bit", {}).get("score", float("nan")),
                "W4A16": layer_stats.get("W4A16", {}).get("score", float("nan")),
                "W4A16_ASYM": layer_stats.get("W4A16_ASYM", {}).get("score", float("nan")),
            }
            all_layer_scores.append(layer_score_row)

            print(
                "  >>> ScoreBoard | "
                f"Std-8bit={layer_score_row['Std-8bit']:.6f} | "
                f"QuIP-4bit={layer_score_row['QuIP-4bit']:.6f} | "
                f"W4A16={layer_score_row['W4A16']:.6f} | "
                f"W4A16_ASYM={layer_score_row['W4A16_ASYM']:.6f}"
            )

            # [MOD-4] 三个4bit先内部选优，再与8bit比较（沿用原阈值思想）
            score_8bit = layer_stats["Std-8bit"]["score"]

            # 默认值：保证所有分支下 search_results 都可安全写入
            score_diff = float("nan")
            size_diff_mb = 0.0

            # four_bit_names = ["QuIP-4bit", "W4A16", "W4A16_ASYM"]
            four_bit_names = [c["name"] for c in candidate_configs if c["bits"] == 4]

            valid_4_names = [n for n in four_bit_names if not math.isnan(layer_stats[n]["score"])]

            if not valid_4_names:
                # 所有 4-bit 策略评分均失败，强制回退 8-bit 并打印警告
                print(f"[WARN] Layer {layer_name}: all 4-bit evaluations failed, forcing Std-8bit.")
                best_config = layer_stats["Std-8bit"]["config"]
                best_score = score_8bit
            else:
                best_4_name = max(valid_4_names, key=lambda n: layer_stats[n]["score"])
                score_4best = layer_stats[best_4_name]["score"]
                score_diff = score_8bit - score_4best
                size_diff_mb = (
                    (layer_stats["Std-8bit"]["size"] - layer_stats[best_4_name]["size"])
                    / 1024
                    / 1024
                )

                if math.isnan(score_8bit):
                    # 8-bit 评分也失败，无法比较，默认选最优 4-bit
                    print(
                        f"[WARN] Layer {layer_name}: Std-8bit evaluation failed, using best 4-bit."
                    )
                    best_config = layer_stats[best_4_name]["config"]
                    best_score = score_4best
                elif score_diff <= ACCEPTANCE_THRESHOLD:
                    best_config = layer_stats[best_4_name]["config"]
                    best_score = score_4best
                else:
                    best_config = layer_stats["Std-8bit"]["config"]
                    best_score = score_8bit

            if best_config["quip"]:
                _apply_official_quip_transform(model, layer_name, layer, block_size=128)
                _set_layer_quantization_bits(
                    session,
                    layer,
                    layer_name,
                    best_config["bits"],
                    group_id=best_config["group_id"],  # [MOD-8]
                    symmetric=best_config.get("symmetric", True),
                )

                for sub_name, sub_mod in layer.named_modules():
                    if "quip" in sub_name:
                        continue
                    if isinstance(sub_mod, torch.nn.Linear) or "proj" in sub_name:
                        if "observer" in sub_name:
                            continue
                        if not hasattr(sub_mod, "weight"):
                            continue
                        full_target_string = f"re:{layer_name}.{sub_name}"
                        global_quip_targets.append(full_target_string)
            else:
                _set_layer_quantization_bits(
                    session,
                    layer,
                    layer_name,
                    best_config["bits"],
                    group_id=best_config["group_id"],  # [MOD-8]
                    symmetric=best_config.get("symmetric", True),
                )

            search_results.append(
                {
                    "layer": layer_name,
                    "best_mode": best_config["name"],
                    "score": best_score,
                    "size_saved_mb": size_diff_mb if best_config["bits"] == 4 else 0,
                    "score_drop": score_diff,
                    # [MOD-13]
                    "group_id": best_config["group_id"],
                    "bits": best_config["bits"],
                    "symmetric": best_config.get("symmetric", True),
                }
            )

            dummy_input = _create_dummy_input(layer, model)
            with torch.no_grad():
                try:
                    if isinstance(dummy_input, dict):
                        _ = layer(**dummy_input)
                    else:
                        _ = layer(dummy_input)
                except Exception:
                    pass
                finally:
                    del dummy_input
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            LifecycleCallbacks.sequential_epoch_end(subgraph=layer)

        print("\n+++++++++++++++++++++++++++++++++++++++++++++")
        print("Auto-Search Summary (Standard 8-bit vs QuIP 4-bit):")
        print(f"{'Layer':<40} | {'Mode':<10} | {'Cos Sim':<10} | {'Save(MB)':<10} | {'Drop'}")
        for res in search_results:
            print(
                f"{res['layer']:<40} | {res['best_mode']:<10} | {res['score']:.6f}   | {res['size_saved_mb']:.2f}       | {res['score_drop']:.6f}"
            )

        print("Per-layer CosSim Score Table (all strategies):")
        print(
            f"{'Layer':<40} | {'Std-8bit':<10} | {'QuIP-4bit':<10} | {'W4A16':<10} | {'W4A16_ASYM':<12}"
        )
        for row in all_layer_scores:
            print(
                f"{row['layer']:<40} | "
                f"{row['Std-8bit']:.6f}   | "
                f"{row['QuIP-4bit']:.6f}   | "
                f"{row['W4A16']:.6f}   | "
                f"{row['W4A16_ASYM']:.6f}"
            )
        print("+++++++++++++++++++++++++++++++++++++++++++++\n")

        _sync_modifier_config_to_model(session, model, global_quip_targets)

        tokenizer = kwargs.get("tokenizer", None)
        _simulate_and_verify(model, tokenizer)

        LifecycleCallbacks.calibration_epoch_end()


def _set_layer_bits_only(layer, target_bits, symmetric=True):
    for name, submodule in layer.named_modules():
        if ("gate" in name and "proj" not in name) or name.endswith(".gate"):
            continue
        if hasattr(submodule, "quantization_scheme") and submodule.quantization_scheme is not None:
            submodule.quantization_scheme = copy.deepcopy(submodule.quantization_scheme)
            if (
                hasattr(submodule.quantization_scheme, "weights")
                and submodule.quantization_scheme.weights is not None
            ):
                submodule.quantization_scheme.weights.num_bits = target_bits
                if hasattr(submodule.quantization_scheme.weights, "symmetric"):
                    submodule.quantization_scheme.weights.symmetric = symmetric


def _apply_official_quip_transform(model, layer_name, layer_module, block_size=128):
    print(f"  >>> [QuIP Fix] Applying official QuIPModifier logic to {layer_name}...")
    current_layer_targets = []
    for name, submodule in layer_module.named_modules():
        if isinstance(submodule, torch.nn.Linear) or "proj" in name:
            if "observer" in name or "quip" in name:
                continue
            if not hasattr(submodule, "weight"):
                continue
            full_target = f"re:{layer_name}.{name}"
            current_layer_targets.append(full_target)

    if not current_layer_targets:
        return

    modifier = QuIPModifier(
        targets=current_layer_targets,
        rotations=["v", "u"],
        transform_block_size=block_size,
        transform_type="hadamard",
        ignore=["lm_head"],
    )

    # [FIX-2b] initialized 属性在部分版本的 modifier 中不存在，
    # 用 getattr 设默认值 False，保证向后兼容。
    if not getattr(modifier, "initialized", False):
        modifier.on_initialize(state=active_session().lifecycle)
        _ensure_quip_weights_materialized(layer_module)

    modifier.on_finalize(state=active_session().lifecycle)


def _ensure_quip_weights_materialized(module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, child in module.named_modules():
        if "quip" in name:
            for param_name, param in child.named_parameters(recurse=False):
                if param.device.type == "meta":
                    dim = param.shape[0]
                    H = get_hadamard_matrix(dim).to(dtype=torch.float16, device=device)
                    delattr(child, param_name)
                    child.register_parameter(param_name, torch.nn.Parameter(H))
            for buf_name, buf in child.named_buffers(recurse=False):
                if buf.device.type == "meta":
                    dim = buf.shape[0]
                    H = get_hadamard_matrix(dim).to(dtype=torch.float16, device=device)
                    setattr(child, buf_name, H)


def _hadamard_unscaled(n: int, device, dtype=None) -> torch.Tensor:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size n must be a power of 2, got n={n}")
    H = torch.ones((1, 1), device=device, dtype=dtype or torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def _hadamard_orthonormal(n: int, device, dtype) -> torch.Tensor:
    H = _hadamard_unscaled(n, device=device, dtype=dtype)
    return H / math.sqrt(n)


def _apply_rotation(weight, block_size=128):
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype
    H = _hadamard_orthonormal(block_size, device=device, dtype=dtype)
    pad_in = (block_size - (in_features % block_size)) % block_size
    w_padded_in = torch.nn.functional.pad(weight, (0, pad_in))
    w_v = torch.matmul(w_padded_in.view(out_features, -1, block_size), H)
    w_v = w_v.view(out_features, -1)
    pad_out = (block_size - (out_features % block_size)) % block_size
    if pad_out > 0:
        w_v = torch.nn.functional.pad(w_v, (0, 0, 0, pad_out))
    w_v_reshaped = w_v.view(-1, block_size, w_v.shape[1])
    w_u = torch.matmul(H, w_v_reshaped)
    w_final = w_u.view(-1, w_v.shape[1])
    return w_final, pad_in, pad_out, H


class QuIPWrapper(torch.nn.Module):
    def __init__(self, original_linear: torch.nn.Linear, block_size=128):
        super().__init__()
        self.block_size = block_size
        with torch.no_grad():
            w_rotated, pad_in, pad_out, H = _apply_rotation(original_linear.weight.data, block_size)
        self.pad_in = pad_in
        self.pad_out = pad_out
        self.register_buffer("H", H)
        out_features_padded, in_features_padded = w_rotated.shape
        self.linear = torch.nn.Linear(
            in_features_padded, out_features_padded, bias=original_linear.bias is not None
        )
        self.linear.weight.data = w_rotated
        if original_linear.bias is not None:
            b_padded = torch.nn.functional.pad(original_linear.bias.data, (0, pad_out))
            b_reshaped = b_padded.view(-1, block_size, 1)
            b_rotated = torch.matmul(self.H, b_reshaped).view(-1)
            self.linear.bias.data = b_rotated
        if hasattr(original_linear, "quantization_scheme"):
            self.linear.quantization_scheme = copy.deepcopy(original_linear.quantization_scheme)

    def forward(self, x):
        dtype = x.dtype
        x = x.to(self.H.dtype)
        if self.pad_in > 0:
            x = torch.nn.functional.pad(x, (0, self.pad_in))
        orig_shape = x.shape
        x_reshaped = x.view(*orig_shape[:-1], -1, self.block_size)
        x_rotated = torch.matmul(x_reshaped, self.H)
        x_rotated = x_rotated.view(*orig_shape)
        out = self.linear(x_rotated.to(dtype))
        out = out.to(self.H.dtype)
        out_shape = out.shape
        out_reshaped = out.view(*out_shape[:-1], -1, self.block_size)
        out_unrotated = torch.matmul(out_reshaped, self.H)
        out_final = out_unrotated.view(*out_shape)
        if self.pad_out > 0:
            out_final = out_final[..., : -self.pad_out]
        return out_final.to(dtype)


def _get_scale_and_zeropoint(weight, bits, symmetric=True, ch_axis=0):
    """
    [FIX] 统一qparam语义，避免ASYM手写公式与fake_quantize契约错配。
    - symmetric=True: 保持原实现（最小改动）
    - symmetric=False: 使用 torch.quantize_per_channel 生成同语义 qparams
    """
    if symmetric:
        # ===== 原逻辑保留 =====
        w_max = weight.abs().amax(dim=1, keepdim=True)
        qmax = 2 ** (bits - 1) - 1
        scale = torch.clamp(w_max / qmax, min=1e-5)
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
        return scale, zero_point
    else:
        # ===== [修改点-1] ASYM 不再手推公式，改为PyTorch原生qparam生成 =====
        # 说明：按每输出通道量化（Linear权重通常out_features在dim=0）
        assert bits == 4, "当前W4A16_ASYM路径预期4bit，可按需放宽"
        qmin, qmax = 0, 2**bits - 1

        w_f = weight.detach().float()
        # per-channel min/max（沿输入维归约）
        w_min = w_f.amin(dim=1)  # [out_features]
        w_max = w_f.amax(dim=1)  # [out_features]
        scales = torch.clamp((w_max - w_min) / float(qmax - qmin), min=1e-8)
        zps = torch.round(qmin - w_min / scales).clamp(qmin, qmax).to(torch.int32)

        # 用原生量化再反取qparams，确保语义闭环
        q = torch.quantize_per_channel(
            w_f, scales=scales, zero_points=zps, axis=ch_axis, dtype=torch.quint8
        )
        s_lib = q.q_per_channel_scales().to(w_f.device).view(-1, 1)
        zp_lib = q.q_per_channel_zero_points().to(w_f.device).to(torch.int32).view(-1, 1)

        zp_lib = zp_lib.to(torch.int32)  # 对齐 fake_quantize 预期 dtype
        s_lib = torch.clamp(s_lib, min=1e-8)  # 防极小 scale 数值不稳

        return s_lib.to(weight.dtype), zp_lib


def _torch_ref_qdq_asym_per_channel(weight, scale, zero_point, ch_axis=0):
    w_f = weight.detach().float()
    s = scale.detach().float().view(-1).cpu()
    zp = zero_point.detach().int().view(-1).cpu()
    q = torch.quantize_per_channel(
        w_f.cpu(), scales=s, zero_points=zp, axis=ch_axis, dtype=torch.quint8
    )
    return q.dequantize().to(weight.device, dtype=weight.dtype)


# [MOD-2] 增加 symmetric 参数（默认True兼容旧调用）
def _calculate_layer_metrics(layer, bits, use_quip=False, symmetric=True):
    """
    Calculates Cosine Similarity with STRICT filtering to avoid QuIP artifacts crash.
    """
    total_cos = 0.0
    total_bytes = 0
    count = 0
    func_used = "unknown"
    q_args = QuantizationArgs(num_bits=bits, symmetric=symmetric)  # [MOD-2]

    for name, submodule in layer.named_modules():
        if any(x in name for x in ["observer", "v_input", "u_output", "quip"]):
            continue
        if "Hadamard" in submodule.__class__.__name__:
            continue

        if not hasattr(submodule, "weight") or submodule.weight is None:
            continue

        if isinstance(submodule, torch.nn.Linear) or "proj" in name:
            hook_triggered = False
            if hasattr(submodule, "_hf_hook"):
                try:
                    submodule._hf_hook.pre_forward(submodule)
                    hook_triggered = True
                except Exception as e:
                    print(f"[WARN] hf_hook.pre_forward failed at {name}: {e}")

            weight = submodule.weight
            if weight.device.type == "meta":
                if hook_triggered:
                    submodule._hf_hook.post_forward(submodule, None)
                continue

            weight_data = weight.data
            total_bytes += weight_data.numel() * (bits / 8.0)

            try:
                w_to_quant = weight_data
                H_for_unrotate = None
                pad_in = 0
                pad_out = 0

                if use_quip:
                    if not isinstance(submodule, torch.nn.Linear):
                        continue
                    temp_wrapper = QuIPWrapper(submodule, block_size=128)
                    w_to_quant = temp_wrapper.linear.weight.data
                    H_for_unrotate = temp_wrapper.H

                    if hasattr(temp_wrapper, "pad_in"):
                        pad_in = temp_wrapper.pad_in
                    elif hasattr(temp_wrapper, "pad_len"):
                        pad_in = temp_wrapper.pad_len
                    else:
                        pad_in = 0
                    pad_out = getattr(temp_wrapper, "pad_out", 0)

                    func_used = f"QuIP_Real({bits}b)"
                else:
                    func_used = f"Std({bits}b)"

                # [MOD-2] 显式传入按输出通道量化轴，避免维度约定歧义
                scale, zero_point = _get_scale_and_zeropoint(
                    w_to_quant, bits, symmetric=symmetric, ch_axis=0
                )

                q_args.num_bits = bits
                q_args.symmetric = symmetric

                if not symmetric:
                    w_dq_rotated = _torch_ref_qdq_asym_per_channel(
                        w_to_quant, scale, zero_point, ch_axis=0
                    )
                else:
                    w_dq_rotated = fake_quantize(
                        x=w_to_quant, scale=scale, zero_point=zero_point, args=q_args
                    )

                mean_shift = (w_dq_rotated.mean() - w_to_quant.mean()).item()
                if (not symmetric) and abs(mean_shift) > 1e-2:
                    print(f"[WARN][{name}] ASYM mean_shift={mean_shift:.4f}")

                if use_quip:
                    rows_padded = w_dq_rotated.shape[0]
                    cols_padded = w_dq_rotated.shape[1]
                    block_size = 128

                    w_dq_reshaped_u = w_dq_rotated.view(-1, block_size, cols_padded)
                    w_u_inv = torch.matmul(H_for_unrotate, w_dq_reshaped_u)
                    w_u_inv = w_u_inv.view(rows_padded, cols_padded)
                    if pad_out > 0:
                        w_u_inv = w_u_inv[:-pad_out, :]

                    rows_orig = w_u_inv.shape[0]
                    w_dq_reshaped_v = w_u_inv.view(rows_orig, -1, block_size)
                    w_recon = torch.matmul(w_dq_reshaped_v, H_for_unrotate)
                    w_recon = w_recon.view(rows_orig, cols_padded)
                    if pad_in > 0:
                        w_recon = w_recon[:, :-pad_in]

                    w_dq = w_recon
                else:
                    w_dq = w_dq_rotated

                min_rows = min(weight_data.shape[0], w_dq.shape[0])
                min_cols = min(weight_data.shape[1], w_dq.shape[1])

                cos_sim = torch.nn.functional.cosine_similarity(
                    weight_data[:min_rows, :min_cols].flatten(),
                    w_dq[:min_rows, :min_cols].flatten(),
                    dim=0,
                    eps=1e-8,
                ).item()

                if cos_sim > 1.0:
                    cos_sim = 1.0
                total_cos += cos_sim
                count += 1
            except Exception as e:
                import traceback

                print(f"[WARN] fake_quant metric failed at {name}: {e}")
                traceback.print_exc()
            finally:
                if hook_triggered:
                    submodule._hf_hook.post_forward(submodule, None)

        if count == 0:
            return float("nan"), "none", 0
    return total_cos / count, func_used, total_bytes


def _extract_real_scheme_from_module(layer, target_bits):
    for name, submodule in layer.named_modules():
        if isinstance(submodule, torch.nn.Linear) or "proj" in name:
            if hasattr(submodule, "quantization_scheme") and submodule.quantization_scheme:
                scheme = submodule.quantization_scheme
                if hasattr(scheme, "weights") and scheme.weights:
                    w_config = None
                    if hasattr(scheme.weights, "dict"):
                        w_config = scheme.weights.dict()
                    elif isinstance(scheme.weights, dict):
                        w_config = copy.deepcopy(scheme.weights)
                    else:
                        try:
                            w_config = {
                                "num_bits": scheme.weights.num_bits,
                                "group_size": getattr(scheme.weights, "group_size", 128),
                                "symmetric": getattr(scheme.weights, "symmetric", True),
                                "strategy": getattr(scheme.weights, "strategy", "group"),
                                "observer": getattr(scheme.weights, "observer", "minmax"),
                                "type": "int",
                                "actorder": getattr(scheme.weights, "actorder", None),
                                "block_structure": getattr(scheme.weights, "block_structure", None),
                                "dynamic": getattr(scheme.weights, "dynamic", False),
                                "observer_kwargs": getattr(scheme.weights, "observer_kwargs", {}),
                            }
                        except:
                            pass
                    if w_config:
                        w_config["num_bits"] = target_bits
                        return w_config
    return None


# [MOD-5] 增加 group_id 与 symmetric
def _set_layer_quantization_bits(
    session, layer, layer_name, target_bits, group_id="group_0", symmetric=True
):
    for name, submodule in layer.named_modules():
        if ("gate" in name and "proj" not in name) or name.endswith(".gate"):
            continue
        if hasattr(submodule, "quantization_scheme") and submodule.quantization_scheme is not None:
            submodule.quantization_scheme = copy.deepcopy(submodule.quantization_scheme)
            if (
                hasattr(submodule.quantization_scheme, "weights")
                and submodule.quantization_scheme.weights is not None
            ):
                submodule.quantization_scheme.weights.num_bits = target_bits
                if hasattr(submodule.quantization_scheme.weights, "symmetric"):
                    submodule.quantization_scheme.weights.symmetric = symmetric  # [FIX-2]

    modifier = None
    for m in session.lifecycle.recipe.modifiers:
        if isinstance(m, QuantizationModifier):
            modifier = m
            break
    if not modifier:
        return
    if modifier.config_groups is None:
        modifier.config_groups = {}

    current_groups = {}
    for k, v in modifier.config_groups.items():
        if hasattr(v, "dict"):
            current_groups[k] = v.dict()
        elif isinstance(v, dict):
            current_groups[k] = copy.deepcopy(v)
        else:
            current_groups[k] = v

    target_group_key = group_id  # [MOD-6]
    if target_group_key in current_groups and "targets" in current_groups[target_group_key]:
        old_targets = current_groups[target_group_key]["targets"]
        prefix = f"re:{layer_name}."
        new_targets = [t for t in old_targets if not t.startswith(prefix)]
        current_groups[target_group_key]["targets"] = new_targets

    if "group_0" not in current_groups:
        try:

            def to_dict(obj):
                if hasattr(obj, "dict"):
                    return obj.dict()
                if isinstance(obj, dict):
                    return obj
                return obj

            flat_weights = to_dict(modifier.weights) if hasattr(modifier, "weights") else None
            if not flat_weights or (
                isinstance(flat_weights, dict) and all(v is None for v in flat_weights.values())
            ):
                extracted = _extract_real_scheme_from_module(layer, 8)
                if extracted:
                    flat_weights = extracted
            flat_targets = (
                modifier.targets
                if hasattr(modifier, "targets") and modifier.targets
                else ["Linear"]
            )
            group_0_dict = {
                "format": "pack-quantized",
                "input_activations": to_dict(modifier.input_activations)
                if hasattr(modifier, "input_activations")
                else None,
                "output_activations": to_dict(modifier.output_activations)
                if hasattr(modifier, "output_activations")
                else None,
                "targets": flat_targets,
                "weights": flat_weights,
            }
            if "ignore" in group_0_dict:
                del group_0_dict["ignore"]
            current_groups["group_0"] = group_0_dict
            if hasattr(modifier, "targets"):
                modifier.targets = []
            if hasattr(modifier, "weights"):
                modifier.weights = None
        except Exception as e:
            print(f"WARNING: Failed to auto-create group_0: {e}")

    modifier.config_groups = current_groups
    if target_bits == 8:
        return

    # [MOD-7] 显式写入symmetric，区分group_2和group_3
    if target_group_key not in current_groups:
        base_source = current_groups.get("group_0")
        if not base_source:
            base_source = {"weights": {"num_bits": 4}, "targets": []}

        new_group = copy.deepcopy(base_source)
        new_group["targets"] = []

        real_scheme = _extract_real_scheme_from_module(layer, target_bits)

        # 先确定weights容器
        if real_scheme:
            new_group["weights"] = real_scheme
        elif "weights" not in new_group or not new_group["weights"]:
            new_group["weights"] = {}

        # 再强制覆盖关键字段，保证group_2/group_3语义准确
        new_group["weights"]["num_bits"] = target_bits
        new_group["weights"]["symmetric"] = symmetric

        if "ignore" in new_group:
            del new_group["ignore"]
        if "transform" in new_group:
            del new_group["transform"]

        current_groups[target_group_key] = new_group

    target_group = current_groups[target_group_key]
    if "weights" not in target_group or not target_group["weights"]:
        target_group["weights"] = {}
    target_group["weights"]["num_bits"] = target_bits
    target_group["weights"]["symmetric"] = symmetric

    if "targets" not in target_group or target_group["targets"] is None:
        target_group["targets"] = []

    for name, submodule in layer.named_modules():
        if not (isinstance(submodule, torch.nn.Linear) or "proj" in name):
            continue
        if "observer" in name or "input" in name or "output" in name or "quip" in name:
            continue
        full_target_name = f"re:{layer_name}.{name}"
        if full_target_name not in target_group["targets"]:
            target_group["targets"].append(full_target_name)
    modifier.config_groups = current_groups


def _collapse_moe_targets(target_list):
    if not target_list:
        return []

    non_experts = [t for t in target_list if ".experts." not in t]

    collapsed_experts = set()
    for t in target_list:
        if ".experts." in t:
            new_t = re.sub(r"\.experts\.\d+\.", ".experts.*.", t)
            collapsed_experts.add(new_t)

    return sorted(non_experts + list(collapsed_experts))


def _ordered_weights_dict(weights: dict) -> dict:
    w = copy.deepcopy(weights or {})
    defaults = {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": 128,
        "num_bits": 8,
        "observer": "minmax",
        "observer_kwargs": {},
        "scale_dtype": None,
        "strategy": "group",
        "symmetric": True,
        "type": "int",
        "zp_dtype": None,
    }
    for k, v in defaults.items():
        if k not in w or w[k] is None:
            w[k] = v

    if int(w.get("num_bits", 8)) == 8:
        w["strategy"] = "channel"
        w["group_size"] = None  # [FIX-2c] 与官方 8-bit channel 配置一致
        w["zp_dtype"] = None
    else:
        w["strategy"] = "group"
        w["group_size"] = 128  # group 策略才需要 group_size
        if w.get("zp_dtype", None) is None and (w.get("symmetric", True) is False):
            w["zp_dtype"] = "torch.int8"

    return {
        "actorder": w.get("actorder"),
        "block_structure": w.get("block_structure"),
        "dynamic": w.get("dynamic"),
        "group_size": w.get("group_size"),
        "num_bits": w.get("num_bits"),
        "observer": w.get("observer"),
        "observer_kwargs": w.get("observer_kwargs"),
        "scale_dtype": w.get("scale_dtype"),
        "strategy": w.get("strategy"),
        "symmetric": w.get("symmetric"),
        "type": w.get("type"),
        "zp_dtype": w.get("zp_dtype"),
    }


def _build_transform_config(sorted_targets: list) -> dict:
    if not sorted_targets:
        return {}

    return {
        "config_groups": {
            "u": {
                "apply": [
                    {
                        "ignore": ["lm_head"],
                        "inverse": False,
                        "location": "weight_output",
                        "targets": sorted_targets,
                    },
                    {
                        "ignore": ["lm_head"],
                        "inverse": True,
                        "location": "output",
                        "targets": ["Linear"],
                    },
                ],
                "head_dim": 128,
                "precision": "torch.float64",
                "randomize": False,
                "requires_grad": False,
                "type": "hadamard",
            },
            "v": {
                "apply": [
                    {
                        "ignore": ["lm_head"],
                        "inverse": False,
                        "location": "input",
                        "targets": ["Linear"],
                    },
                    {
                        "ignore": ["lm_head"],
                        "inverse": True,
                        "location": "weight_input",
                        "targets": sorted_targets,
                    },
                ],
                "head_dim": 128,
                "precision": "torch.float64",
                "randomize": False,
                "requires_grad": False,
                "type": "hadamard",
            },
        }
    }


def _sync_modifier_config_to_model(session, model, quip_layers_list):
    modifier = None
    for m in session.lifecycle.recipe.modifiers:
        if isinstance(m, QuantizationModifier):
            modifier = m
            break
    if not modifier:
        return

    def layer_sort_key(s):
        match = re.search(r"\.layers\.(\d+)\.", s)
        if match:
            return int(match.group(1)), s
        return 999999, s

    final_groups = {}
    source_groups = modifier.config_groups or {}

    def to_dict_safe(obj):
        if hasattr(obj, "dict"):
            return obj.dict()
        if isinstance(obj, dict):
            return obj
        return obj

    g0_source = source_groups.get("group_0", {})
    g0_source = to_dict_safe(g0_source)
    default_weights = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
        "actorder": None,
    }
    final_weights_0 = default_weights.copy()
    if g0_source.get("weights"):
        source_w = to_dict_safe(g0_source["weights"])
        final_weights_0.update(source_w)
        final_weights_0["num_bits"] = 8
        final_weights_0["symmetric"] = True

    input_acts = (
        to_dict_safe(modifier.input_activations) if hasattr(modifier, "input_activations") else None
    )
    output_acts = (
        to_dict_safe(modifier.output_activations)
        if hasattr(modifier, "output_activations")
        else None
    )

    final_groups["group_0"] = {
        "format": "pack-quantized",
        "input_activations": input_acts,
        "output_activations": output_acts,
        "targets": ["Linear"],
        "weights": final_weights_0,
    }

    # [MOD-10] 其余组按“是否实际命中targets”决定是否写入
    for gk in ["group_1", "group_2", "group_3"]:
        if gk in source_groups:
            v = copy.deepcopy(to_dict_safe(source_groups[gk]))
            if v.get("targets"):
                v["targets"] = _collapse_moe_targets(v["targets"])
                v["targets"] = sorted(v["targets"], key=layer_sort_key)
            if v.get("targets"):  # 只有非空才保留
                final_groups[gk] = v

    for gk, gv in final_groups.items():
        if isinstance(gv, dict) and "weights" in gv:
            gv["weights"] = _ordered_weights_dict(gv.get("weights", {}))

    transform_config_dict = {}
    if quip_layers_list:
        unique_targets = list(set(quip_layers_list))
        collapsed_targets = _collapse_moe_targets(unique_targets)
        sorted_targets = sorted(collapsed_targets, key=layer_sort_key)
        transform_config_dict = _build_transform_config(sorted_targets)

    if not hasattr(model, "config"):
        model.config = type("Config", (), {})()
    if not hasattr(model.config, "quantization_config") or model.config.quantization_config is None:
        model.config.quantization_config = {}

    q_config_data = {"config_groups": final_groups, "quant_method": "compressed-tensors"}

    if isinstance(model.config.quantization_config, dict):
        model.config.quantization_config.update(q_config_data)
        if transform_config_dict:
            model.config.quantization_config["transform_config"] = transform_config_dict
        else:
            model.config.quantization_config.pop("transform_config", None)  # [FIX-3]
    else:
        model.config.quantization_config.config_groups = final_groups
        model.config.quantization_config.quant_method = "compressed-tensors"
        if transform_config_dict:
            model.config.quantization_config.transform_config = transform_config_dict
        elif hasattr(model.config.quantization_config, "transform_config"):
            delattr(model.config.quantization_config, "transform_config")  # [FIX-3]

    original_save = model.save_pretrained

    def new_save_pretrained(save_directory, *args, **kwargs):
        original_save(save_directory, *args, **kwargs)
        print(f"DEBUG: Force overwriting config.json in {save_directory}...")
        config_path = os.path.join(save_directory, "config.json")
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            if "quantization_config" not in data:
                data["quantization_config"] = {}
            data["quantization_config"]["config_groups"] = final_groups
            data["quantization_config"]["quant_method"] = "compressed-tensors"
            if transform_config_dict:
                data["quantization_config"]["transform_config"] = transform_config_dict
            else:
                data["quantization_config"].pop("transform_config", None)  # [FIX-3]

            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
            print("DEBUG: config.json overwritten with FULL STRUCTURE & SORTING!")
        except Exception as e:
            print(f"WARNING: Failed to overwrite config.json: {e}")

    model.save_pretrained = new_save_pretrained


def _create_dummy_input(
    layer: torch.nn.Module, model: torch.nn.Module
) -> torch.Tensor | dict[str, Any]:
    try:
        param = next(layer.parameters())
    except StopIteration:
        param = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)
    device = param.device
    dtype = param.dtype
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", param.shape[-1] if len(param.shape) > 0 else 4096)
    hidden_states = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)
    if hasattr(layer, "self_attn") or "DecoderLayer" in layer.__class__.__name__:
        return {
            "hidden_states": hidden_states,
            "attention_mask": torch.ones(1, 1, device=device, dtype=torch.long),
        }
    return hidden_states


def _simulate_and_verify(model, tokenizer=None):
    from functools import partial

    print("\n+++++++++++++++++++++++++++++++++++++++++++++")
    print(">>> [Simulation] Starting FakeQuant Inference Verification (Hook Method)...")

    if tokenizer is None:
        try:
            model_path = getattr(model.config, "_name_or_path", None)
            if model_path:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            pass

    def quantize_pre_hook(module, input, bits=8, group_size=128, symmetric=True):
        if not hasattr(module, "weight") or module.weight is None:
            return
        w = module.weight
        if w.device.type == "meta":
            return
        module._saved_weight_ref = w.data

        try:
            out_f, in_f = w.shape
            use_group = (group_size > 0) and (in_f % group_size == 0)
            w_float = w.data.float()

            # [修改点-3] 复用统一qparam入口，保证评估/模拟一致
            q_args = QuantizationArgs(num_bits=bits, symmetric=symmetric)

            if use_group:
                w_reshaped = w_float.view(out_f, -1, group_size)
                w_fake_groups = []
                for g in range(w_reshaped.shape[1]):
                    wg = w_reshaped[:, g, :]
                    s, zp = _get_scale_and_zeropoint(wg, bits, symmetric=symmetric, ch_axis=0)

                    if not symmetric:
                        wq = _torch_ref_qdq_asym_per_channel(wg.to(w.dtype), s, zp, ch_axis=0)
                    else:
                        wq = fake_quantize(
                            x=wg.to(w.dtype), scale=s.to(w.dtype), zero_point=zp, args=q_args
                        )

                    w_fake_groups.append(wq.unsqueeze(1))
                w_fake_reshaped = torch.cat(w_fake_groups, dim=1)
            else:
                w_reshaped = w_float
                s, zp = _get_scale_and_zeropoint(w_reshaped, bits, symmetric=symmetric, ch_axis=0)

                if not symmetric:
                    w_fake_reshaped = _torch_ref_qdq_asym_per_channel(
                        w_reshaped.to(w.dtype), s, zp, ch_axis=0
                    )
                else:
                    w_fake_reshaped = fake_quantize(
                        x=w_reshaped.to(w.dtype), scale=s.to(w.dtype), zero_point=zp, args=q_args
                    )

            if use_group:
                w_fake = w_fake_reshaped.view(out_f, in_f)
            else:
                w_fake = w_fake_reshaped

            module.weight.data = w_fake
        except Exception as e:
            print(f"[WARN] simulate quant hook failed: {e}")
            if hasattr(module, "_saved_weight_ref"):
                module.weight.data = module._saved_weight_ref

    def restore_post_hook(module, input, output):
        if hasattr(module, "_saved_weight_ref"):
            module.weight.data = module._saved_weight_ref
            del module._saved_weight_ref

    hooks = []
    skip_names = ["lm_head", "output"]
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or "proj" in name:
            if any(s in name for s in skip_names):
                continue
            if any(x in name for x in ["v_input", "u_output", "quip"]):
                continue

            if not hasattr(module, "weight") or module.weight is None:
                continue

            bits = 16
            symmetric = True
            if hasattr(module, "quantization_scheme") and module.quantization_scheme:
                _w = getattr(module.quantization_scheme, "weights", None)
                if _w is not None:
                    bits = getattr(_w, "num_bits", 16)
                    symmetric = getattr(_w, "symmetric", True)

            if bits < 16:
                h1 = module.register_forward_pre_hook(
                    partial(quantize_pre_hook, bits=bits, symmetric=symmetric)
                )
                h2 = module.register_forward_hook(restore_post_hook)
                hooks.extend([h1, h2])

    print(f"    Registered hooks on {len(hooks) // 2} layers.")

    test_cases = [
        {"prompt": "1 + 1 ="},
        {"prompt": "The capital of China is"},
        {"prompt": "Hello, my name is"},
    ]

    if tokenizer:
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            target_device = next(p.device for p in model.parameters() if p.device.type != "meta")
        except:
            target_device = "cuda:0"

        for i, case in enumerate(test_cases):
            prompt = case["prompt"]
            print(f"    --- Test {i + 1}: {prompt}")
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(target_device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                res = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt) :].strip()
                if "stop" in case and case["stop"] in res:
                    res = res.split(case["stop"])[0]

                print(f"    [Answer]: \033[92m{res}\033[0m")
            except Exception as e:
                print(f"    [Error]: {e}")
    else:
        print("    [Skip] No tokenizer available.")

    for h in hooks:
        h.remove()
    for module in model.modules():
        if hasattr(module, "_saved_weight_ref"):
            del module._saved_weight_ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("+++++++++++++++++++++++++++++++++++++++++++++\n")

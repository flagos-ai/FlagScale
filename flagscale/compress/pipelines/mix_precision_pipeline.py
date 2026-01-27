import torch
import re
import math
import copy
import json
import os
from typing import Optional, Dict, Any, List, Union
from torch.utils.data.dataloader import DataLoader

# === Core Imports from llmcompressor ===
from llmcompressor.core.session_functions import LifecycleCallbacks
from llmcompressor.core import active_session
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import get_sequential_targets
from llmcompressor.pipelines.layer_sequential.helpers import match_modules
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import QuIPModifier

# === Quantization Utilities ===
try:
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.quant_args import QuantizationArgs
except ImportError:
    raise ImportError("Could not import quantization functions. Please ensure llmcompressor/compressed-tensors is installed.")

# === Hadamard Matrix Utility ===
try:
    from llmcompressor.modifiers.transform.utils.hadamard import get_hadamard_matrix
except ImportError:
    def get_hadamard_matrix(n, dtype=torch.float32, device="cpu"):
        from scipy.linalg import hadamard
        H = torch.tensor(hadamard(n), dtype=dtype, device=device)
        return H / math.sqrt(n)

# =============================================================================
# Pipeline Class Registration
# =============================================================================

@CalibrationPipeline.register("mix_precision_search")
class MixPrecisionPipeline(CalibrationPipeline):
    
    @staticmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        dataset_args: Any,
        **kwargs
    ):
        session = active_session()
        modifiers = session.lifecycle.recipe.modifiers
        
        # 1. 获取目标层
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

        # 2. 自然排序 (Layer 0, Layer 1, ... Layer 10)
        def natural_keys(item):
            text = item[0]
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

        sorted_layers = sorted(layers_to_process, key=natural_keys)

        LifecycleCallbacks.calibration_epoch_start()
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(f"DEBUG: Processing {len(sorted_layers)} layers with Auto-Search (8-bit vs QuIP-4bit).")
        
        search_results = [] 
        global_quip_targets = [] 

        # 3. 逐层搜索
        for i, (layer_name, layer) in enumerate(sorted_layers):
            match = re.search(r"\.(\d+)(?:\.|$)", layer_name)
            real_layer_idx = int(match.group(1)) if match else i
            
            print(f"\nSearching Layer {real_layer_idx}: {layer_name}")
            
            candidate_configs = [
                {"name": "QuIP-4bit", "bits": 4, "quip": True},
                {"name": "Std-8bit",  "bits": 8, "quip": False}
            ]
            
            best_score = -1.0
            best_config = candidate_configs[1] # 默认回退到 8bit
            
            ACCEPTANCE_THRESHOLD = 0.009 
            layer_stats = {} 

            # 测试两种配置
            for config in candidate_configs:
                bit = config["bits"]
                use_quip = config["quip"]
                name = config["name"]
                
                # 设置当前层的量化位宽
                _set_layer_quantization_bits(session, layer, layer_name, bit)
                
                # 计算指标 (Cos Sim & Size)
                current_score, func_name, param_bytes = _calculate_layer_metrics(
                    layer, bit, use_quip=use_quip
                )
                
                layer_stats[name] = {
                    "score": current_score,
                    "size": param_bytes,
                    "config": config,
                    "func_name": func_name
                }
                
                print(f"  - Testing {name:<10} | Cos Sim: {current_score:.6f} | Size: {param_bytes/1024/1024:.2f} MB | Func: {func_name}")

            # 决策逻辑
            score_8bit = layer_stats["Std-8bit"]["score"]
            score_4bit = layer_stats["QuIP-4bit"]["score"]
            
            score_diff = score_8bit - score_4bit
            size_diff_mb = (layer_stats["Std-8bit"]["size"] - layer_stats["QuIP-4bit"]["size"]) / 1024 / 1024
            
            if score_diff <= ACCEPTANCE_THRESHOLD:
                best_config = layer_stats["QuIP-4bit"]["config"]
                best_score = score_4bit
                decision_reason = f"Accepted (Drop {score_diff:.4f} <= {ACCEPTANCE_THRESHOLD})"
            else:
                best_config = layer_stats["Std-8bit"]["config"]
                best_score = score_8bit
                decision_reason = f"Rejected (Drop {score_diff:.4f} > {ACCEPTANCE_THRESHOLD})"

            print(f"  >>> Decision: {best_config['name']} | {decision_reason}")
            print(f"  >>> Comparison: Saved {size_diff_mb:.2f} MB | Score Drop: {score_diff:.6f}")

            # 应用最终决策
            if best_config["quip"]:
                # 如果选中 QuIP，需要真正应用旋转矩阵
                _apply_official_quip_transform(model, layer_name, layer, block_size=128)
                _set_layer_quantization_bits(session, layer, layer_name, best_config["bits"])
                
                # 记录哪些层用了 QuIPconfig.json) (用
                for sub_name, sub_mod in layer.named_modules():
                    if "quip" in sub_name: continue 
                    if isinstance(sub_mod, torch.nn.Linear) or "proj" in sub_name:
                        if "observer" in sub_name: continue
                        if not hasattr(sub_mod, "weight"): continue
                        full_target_string = f"re:{layer_name}.{sub_name}"
                        global_quip_targets.append(full_target_string)
            else:
                # 如果选中 8bit，确保设置回 8bit
                _set_layer_quantization_bits(session, layer, layer_name, best_config["bits"])
            
            search_results.append({
                "layer": layer_name,
                "best_mode": best_config["name"],
                "score": best_score,
                "size_saved_mb": size_diff_mb if best_config["bits"] == 4 else 0,
                "score_drop": score_diff
            })
            
            # 清理缓存
            dummy_input = _create_dummy_input(layer, model)
            with torch.no_grad():
                try:
                    if isinstance(dummy_input, dict): _ = layer(**dummy_input)
                    else: _ = layer(dummy_input)
                except Exception: pass
                finally:
                    del dummy_input
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            LifecycleCallbacks.sequential_epoch_end()
        
        # 4. 输出摘要
        print("\n+++++++++++++++++++++++++++++++++++++++++++++")
        print("Auto-Search Summary (Standard 8-bit vs QuIP 4-bit):")
        print(f"{'Layer':<40} | {'Mode':<10} | {'Cos Sim':<10} | {'Save(MB)':<10} | {'Drop'}")
        for res in search_results:
            print(f"{res['layer']:<40} | {res['best_mode']:<10} | {res['score']:.6f}   | {res['size_saved_mb']:.2f}       | {res['score_drop']:.6f}")
        print("+++++++++++++++++++++++++++++++++++++++++++++\n")

        # 5. 同步配�config.json 并保存
        _sync_modifier_config_to_model(session, model, global_quip_targets)

        # 6. 最终模拟验证
        tokenizer = kwargs.get("tokenizer", None)
        _simulate_and_verify(model, tokenizer)
    
        LifecycleCallbacks.calibration_epoch_end()

# =============================================================================
# Helper Functions (QuIP Logic & Metrics)
# =============================================================================

def _apply_official_quip_transform(model, layer_name, layer_module, block_size=128):
    print(f"  >>> [QuIP Fix] Applying official QuIPModifier logic to {layer_name}...")
    current_layer_targets = []
    for name, submodule in layer_module.named_modules():
        if isinstance(submodule, torch.nn.Linear) or "proj" in name:
            if "observer" in name or "quip" in name: continue
            if not hasattr(submodule, "weight"): continue
            full_target = f"re:{layer_name}.{name}"
            current_layer_targets.append(full_target)
    
    if not current_layer_targets: return

    modifier = QuIPModifier(
        targets=current_layer_targets,
        rotations=["v", "u"],
        transform_block_size=block_size,
        transform_type="hadamard",
        ignore=["lm_head"]
    )

    if not modifier.initialized:
        modifier.on_initialize(state=active_session().lifecycle) 
        _ensure_quip_weights_materialized(layer_module)

    modifier.on_finalize(state=active_session().lifecycle)

def _ensure_quip_weights_materialized(module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, child in module.named_modules():
        if "quip" in name:
            for param_name, param in child.named_parameters(recurse=False):
                if param.device.type == 'meta':
                    # print(f"    [WARNING] Found meta parameter in {name}.{param_name}, materializing...")
                    dim = param.shape[0]
                    H = get_hadamard_matrix(dim).to(dtype=torch.float16, device=device)
                    delattr(child, param_name)
                    child.register_parameter(param_name, torch.nn.Parameter(H))
            for buf_name, buf in child.named_buffers(recurse=False):
                if buf.device.type == 'meta':
                    # print(f"    [WARNING] Found meta buffer in {name}.{buf_name}, materializing...")
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
    if pad_out > 0: w_v = torch.nn.functional.pad(w_v, (0, 0, 0, pad_out))
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
        self.register_buffer('H', H)
        out_features_padded, in_features_padded = w_rotated.shape
        self.linear = torch.nn.Linear(in_features_padded, out_features_padded, bias=original_linear.bias is not None)
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
        if self.pad_in > 0: x = torch.nn.functional.pad(x, (0, self.pad_in))
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
        if self.pad_out > 0: out_final = out_final[..., :-self.pad_out]
        return out_final.to(dtype)

def _get_scale_and_zeropoint(weight, bits):
    w_max = weight.abs().amax(dim=1, keepdim=True)
    max_q = 2**(bits - 1) - 1
    scale = w_max / max_q
    scale = torch.clamp(scale, min=1e-5) 
    zero_point = torch.zeros_like(scale, dtype=torch.int32)
    return scale, zero_point

def _calculate_layer_metrics(layer, bits, use_quip=False):
    """
    Calculates Cosine Similarity with STRICT filtering to avoid QuIP artifacts crash.
    """
    total_cos = 0.0
    total_bytes = 0
    count = 0
    func_used = "unknown"
    q_args = QuantizationArgs(num_bits=bits, symmetric=True)
    
    for name, submodule in layer.named_modules():
        # === [CRITICAL FIX] 严格过滤 QuIP 生成的子模块 ===
        if any(x in name for x in ["observer", "v_input", "u_output", "quip"]): 
            continue
        if "Hadamard" in submodule.__class__.__name__:
            continue

        if not hasattr(submodule, "weight") or submodule.weight is None: continue
        
        if isinstance(submodule, torch.nn.Linear) or "proj" in name:
            hook_triggered = False
            if hasattr(submodule, "_hf_hook"):
                try:
                    submodule._hf_hook.pre_forward(submodule)
                    hook_triggered = True
                except Exception: pass

            weight = submodule.weight
            if weight.device.type == 'meta':
                if hook_triggered: submodule._hf_hook.post_forward(submodule, None)
                continue 

            weight_data = weight.data
            total_bytes += weight_data.numel() * (bits / 8.0)

            try:
                w_to_quant = weight_data
                H_for_unrotate = None
                pad_in = 0
                pad_out = 0
                
                if use_quip:
                    if not hasattr(submodule, "bias"): continue
                    temp_wrapper = QuIPWrapper(submodule, block_size=128)
                    w_to_quant = temp_wrapper.linear.weight.data
                    H_for_unrotate = temp_wrapper.H
                    
                    if hasattr(temp_wrapper, 'pad_in'): pad_in = temp_wrapper.pad_in
                    elif hasattr(temp_wrapper, 'pad_len'): pad_in = temp_wrapper.pad_len
                    else: pad_in = 0
                    pad_out = getattr(temp_wrapper, 'pad_out', 0)
                    
                    func_used = f"QuIP_Real({bits}b)"
                else:
                    func_used = f"Std({bits}b)"

                scale, zero_point = _get_scale_and_zeropoint(w_to_quant, bits)
                q_args.num_bits = bits 
                w_dq_rotated = fake_quantize(x=w_to_quant, scale=scale, zero_point=zero_point, args=q_args)
                
                if use_quip:
                    rows_padded = w_dq_rotated.shape[0]
                    cols_padded = w_dq_rotated.shape[1]
                    block_size = 128
                    
                    w_dq_reshaped_u = w_dq_rotated.view(-1, block_size, cols_padded)
                    w_u_inv = torch.matmul(H_for_unrotate, w_dq_reshaped_u)
                    w_u_inv = w_u_inv.view(rows_padded, cols_padded)
                    if pad_out > 0: w_u_inv = w_u_inv[:-pad_out, :]
                    
                    rows_orig = w_u_inv.shape[0]
                    w_dq_reshaped_v = w_u_inv.view(rows_orig, -1, block_size)
                    w_recon = torch.matmul(w_dq_reshaped_v, H_for_unrotate)
                    w_recon = w_recon.view(rows_orig, cols_padded)
                    if pad_in > 0: w_recon = w_recon[:, :-pad_in]
                    
                    w_dq = w_recon
                else:
                    w_dq = w_dq_rotated

                min_rows = min(weight_data.shape[0], w_dq.shape[0])
                min_cols = min(weight_data.shape[1], w_dq.shape[1])
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    weight_data[:min_rows, :min_cols].flatten(), 
                    w_dq[:min_rows, :min_cols].flatten(), 
                    dim=0, eps=1e-8
                ).item()
                
                if cos_sim > 1.0: cos_sim = 1.0
                total_cos += cos_sim
                count += 1
            except Exception as e:
                pass
            finally:
                if hook_triggered: submodule._hf_hook.post_forward(submodule, None)
            
    if count == 0: return 0.0, "none", 0
    return total_cos / count, func_used, total_bytes

def _extract_real_scheme_from_module(layer, target_bits):
    for name, submodule in layer.named_modules():
        if isinstance(submodule, torch.nn.Linear) or "proj" in name:
            if hasattr(submodule, "quantization_scheme") and submodule.quantization_scheme:
                scheme = submodule.quantization_scheme
                if hasattr(scheme, "weights") and scheme.weights:
                    w_config = None
                    if hasattr(scheme.weights, "dict"): w_config = scheme.weights.dict()
                    elif isinstance(scheme.weights, dict): w_config = copy.deepcopy(scheme.weights)
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
                        except: pass
                    if w_config:
                        w_config["num_bits"] = target_bits
                        return w_config
    return None

def _set_layer_quantization_bits(session, layer, layer_name, target_bits, transform_scheme=None):
    for name, submodule in layer.named_modules():
        if ("gate" in name and "proj" not in name) or name.endswith(".gate"): continue 
        if hasattr(submodule, "quantization_scheme") and submodule.quantization_scheme is not None:
            submodule.quantization_scheme = copy.deepcopy(submodule.quantization_scheme)
            if hasattr(submodule.quantization_scheme, 'weights') and submodule.quantization_scheme.weights is not None:
                submodule.quantization_scheme.weights.num_bits = target_bits
    
    modifier = None
    for m in session.lifecycle.recipe.modifiers:
        if isinstance(m, QuantizationModifier):
            modifier = m
            break
    if not modifier: return
    if modifier.config_groups is None: modifier.config_groups = {}

    current_groups = {}
    for k, v in modifier.config_groups.items():
        if hasattr(v, 'dict'): current_groups[k] = v.dict()
        elif isinstance(v, dict): current_groups[k] = copy.deepcopy(v)
        else: current_groups[k] = v
    
    target_group_key = "group_1"
    if target_group_key in current_groups and "targets" in current_groups[target_group_key]:
        old_targets = current_groups[target_group_key]["targets"]
        prefix = f"re:{layer_name}."
        new_targets = [t for t in old_targets if not t.startswith(prefix)]
        current_groups[target_group_key]["targets"] = new_targets

    if "group_0" not in current_groups:
        try:
            def to_dict(obj):
                if hasattr(obj, 'dict'): return obj.dict()
                if isinstance(obj, dict): return obj
                return obj
            flat_weights = to_dict(modifier.weights) if hasattr(modifier, 'weights') else None
            if not flat_weights or (isinstance(flat_weights, dict) and all(v is None for v in flat_weights.values())):
                    extracted = _extract_real_scheme_from_module(layer, 8)
                    if extracted: flat_weights = extracted
            flat_targets = modifier.targets if hasattr(modifier, 'targets') and modifier.targets else ["Linear"]
            group_0_dict = {
                "format": "pack-quantized",
                "input_activations": to_dict(modifier.input_activations) if hasattr(modifier, 'input_activations') else None,
                "output_activations": to_dict(modifier.output_activations) if hasattr(modifier, 'output_activations') else None,
                "targets": flat_targets,
                "weights": flat_weights
            }
            if "ignore" in group_0_dict: del group_0_dict["ignore"]
            current_groups["group_0"] = group_0_dict
            if hasattr(modifier, 'targets'): modifier.targets = []
            if hasattr(modifier, 'weights'): modifier.weights = None
        except Exception as e: print(f"WARNING: Failed to auto-create group_0: {e}")
    
    modifier.config_groups = current_groups
    if target_bits == 8: return 

    if target_group_key not in current_groups:
        base_source = current_groups.get("group_0")
        if not base_source: base_source = {"weights": {"num_bits": 4}, "targets": []}
        new_group = copy.deepcopy(base_source)
        new_group['targets'] = []
        real_scheme = _extract_real_scheme_from_module(layer, target_bits)
        if real_scheme: new_group['weights'] = real_scheme
        else:
            if 'weights' in new_group and new_group['weights']: new_group['weights']['num_bits'] = target_bits
        if "ignore" in new_group: del new_group["ignore"]
        if "transform" in new_group: del new_group["transform"]
        current_groups[target_group_key] = new_group

    target_group = current_groups[target_group_key]
    if 'targets' not in target_group or target_group['targets'] is None: target_group['targets'] = []
    
    for name, submodule in layer.named_modules():
        if not (isinstance(submodule, torch.nn.Linear) or "proj" in name): continue
        if "observer" in name or "input" in name or "output" in name or "quip" in name: continue
        full_target_name = f"re:{layer_name}.{name}"
        if full_target_name not in target_group['targets']:
            target_group['targets'].append(full_target_name)
    modifier.config_groups = current_groups

def _sync_modifier_config_to_model(session, model, quip_layers_list):
    modifier = None
    for m in session.lifecycle.recipe.modifiers:
        if isinstance(m, QuantizationModifier):
            modifier = m
            break
    if not modifier: return

    def layer_sort_key(s):
        match = re.search(r"\.layers\.(\d+)\.", s)
        if match: return int(match.group(1)), s
        return 999999, s

    final_groups = {}
    source_groups = modifier.config_groups or {}
    
    def to_dict_safe(obj):
        if hasattr(obj, 'dict'): return obj.dict()
        if isinstance(obj, dict): return obj
        return obj

    g0_source = source_groups.get("group_0", {})
    g0_source = to_dict_safe(g0_source)
    default_weights = {
        "num_bits": 8, "type": "int", "symmetric": True, 
        "strategy": "tensor", "dynamic": False, "actorder": None
    }
    final_weights_0 = default_weights.copy()
    if "weights" in g0_source and g0_source["weights"]:
        source_w = to_dict_safe(g0_source["weights"])
        final_weights_0.update(source_w)
        final_weights_0["num_bits"] = 8 

    input_acts = to_dict_safe(modifier.input_activations) if hasattr(modifier, 'input_activations') else None
    output_acts = to_dict_safe(modifier.output_activations) if hasattr(modifier, 'output_activations') else None

    final_groups["group_0"] = {
        "format": "pack-quantized",
        "input_activations": input_acts,
        "output_activations": output_acts,
        "targets": ["Linear"],
        "weights": final_weights_0
    }

    if "group_1" in source_groups:
        v = source_groups["group_1"]
        v = to_dict_safe(v)
        v = copy.deepcopy(v)
        if "targets" in v and v["targets"]:
            v["targets"] = sorted(v["targets"], key=layer_sort_key)
        final_groups["group_1"] = v

    transform_config_dict = {}
    if quip_layers_list:
        unique_targets = list(set(quip_layers_list))
        sorted_targets = sorted(unique_targets, key=layer_sort_key)
        transform_config_dict = {
            "config_groups": {
                "u": {
                    "type": "hadamard", "head_dim": 128, "precision": "torch.float64",
                    "randomize": False, "requires_grad": False,
                    "apply": [
                        {"location": "weight_output", "inverse": False, "targets": sorted_targets, "ignore": ["lm_head"]},
                        {"location": "output", "inverse": True, "targets": sorted_targets, "ignore": ["lm_head"]}
                    ]
                },
                "v": {
                    "type": "hadamard", "head_dim": 128, "precision": "torch.float64",
                    "randomize": False, "requires_grad": False,
                    "apply": [
                        {"location": "input", "inverse": False, "targets": sorted_targets, "ignore": ["lm_head"]},
                        {"location": "weight_input", "inverse": True, "targets": sorted_targets, "ignore": ["lm_head"]}
                    ]
                }
            }
        }

    if not hasattr(model, 'config'): model.config = type('Config', (), {})()
    if not hasattr(model.config, 'quantization_config') or model.config.quantization_config is None:
        model.config.quantization_config = {}
    
    q_config_data = {"config_groups": final_groups, "quant_method": "compressed-tensors"}
    
    if isinstance(model.config.quantization_config, dict):
        model.config.quantization_config.update(q_config_data)
        if transform_config_dict:
            model.config.quantization_config['transform_config'] = transform_config_dict
    else:
        model.config.quantization_config.config_groups = final_groups
        model.config.quantization_config.quant_method = "compressed-tensors"
        if transform_config_dict:
            model.config.quantization_config.transform_config = transform_config_dict

    original_save = model.save_pretrained
    def new_save_pretrained(save_directory, *args, **kwargs):
        original_save(save_directory, *args, **kwargs)
        print(f"DEBUG: Force overwriting config.json in {save_directory}...")
        config_path = os.path.join(save_directory, "config.json")
        try:
            with open(config_path, 'r') as f: data = json.load(f)
            if "quantization_config" not in data: data["quantization_config"] = {}
            data["quantization_config"]["config_groups"] = final_groups
            data["quantization_config"]["quant_method"] = "compressed-tensors"
            if transform_config_dict: data["quantization_config"]["transform_config"] = transform_config_dict
            with open(config_path, 'w') as f: json.dump(data, f, indent=2)
            print("DEBUG: config.json overwritten with FULL STRUCTURE & SORTING!")
        except Exception as e: print(f"WARNING: Failed to overwrite config.json: {e}")
    model.save_pretrained = new_save_pretrained

def _create_dummy_input(layer: torch.nn.Module, model: torch.nn.Module) -> Union[torch.Tensor, Dict[str, Any]]:
    try: param = next(layer.parameters())
    except StopIteration: param = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)
    device = param.device
    dtype = param.dtype
    config = getattr(model, 'config', None)
    hidden_size = getattr(config, 'hidden_size', param.shape[-1] if len(param.shape) > 0 else 4096)
    hidden_states = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)
    if hasattr(layer, 'self_attn') or 'DecoderLayer' in layer.__class__.__name__:
        return {'hidden_states': hidden_states, 'attention_mask': torch.ones(1, 1, device=device, dtype=torch.long)}
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
        except Exception: pass

    def quantize_pre_hook(module, input, bits=8, group_size=128):
        if not hasattr(module, "weight") or module.weight is None: return
        w = module.weight
        if w.device.type == 'meta': return
        module._saved_weight_ref = w.data 
        
        try:
            out_f, in_f = w.shape
            use_group = (group_size > 0) and (in_f % group_size == 0)
            w_float = w.data.float()

            if use_group:
                w_reshaped = w_float.view(out_f, -1, group_size)
                w_max = w_reshaped.abs().amax(dim=-1, keepdim=True)
            else:
                w_reshaped = w_float
                w_max = w_float.abs().amax(dim=1, keepdim=True)

            max_q = 2**(bits - 1) - 1
            scale = torch.clamp(w_max / max_q, min=1e-5)
            zp = torch.zeros_like(scale, dtype=torch.int32)

            q_args = QuantizationArgs(num_bits=bits, symmetric=True)
            w_fake_reshaped = fake_quantize(w_reshaped.to(w.dtype), scale.to(w.dtype), zp, q_args)
            
            if use_group: w_fake = w_fake_reshaped.view(out_f, in_f)
            else: w_fake = w_fake_reshaped
            
            module.weight.data = w_fake
        except Exception:
            if hasattr(module, "_saved_weight_ref"): module.weight.data = module._saved_weight_ref

    def restore_post_hook(module, input, output):
        if hasattr(module, "_saved_weight_ref"):
            module.weight.data = module._saved_weight_ref
            del module._saved_weight_ref

    hooks = []
    skip_names = ["lm_head", "output"] 
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or "proj" in name:
            if any(s in name for s in skip_names): continue
            if any(x in name for x in ["v_input", "u_output", "quip"]): continue
            
            if not hasattr(module, "weight") or module.weight is None: continue
            
            bits = 16
            if hasattr(module, "quantization_scheme") and module.quantization_scheme:
                bits = module.quantization_scheme.weights.num_bits
            
            if bits < 16:
                h1 = module.register_forward_pre_hook(partial(quantize_pre_hook, bits=bits))
                h2 = module.register_forward_hook(restore_post_hook)
                hooks.extend([h1, h2])

    print(f"    Registered hooks on {len(hooks)//2} layers.")

    test_cases = [
        {"prompt": "1 + 1 ="},
        {"prompt": "The capital of China is"},
        {"prompt": "Hello, my name is"}
    ]

    if tokenizer:
        model.eval()
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        try:
            target_device = next(p.device for p in model.parameters() if p.device.type != 'meta')
        except: target_device = "cuda:0"

        for i, case in enumerate(test_cases):
            prompt = case["prompt"]
            print(f"    --- Test {i+1}: {prompt}")
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(target_device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=10, 
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                res = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
                if "stop" in case and case["stop"] in res:
                    res = res.split(case["stop"])[0]
                
                print(f"    [Answer]: \033[92m{res}\033[0m") 
            except Exception as e:
                print(f"    [Error]: {e}")
    else:
        print("    [Skip] No tokenizer available.")

    for h in hooks: h.remove()
    for module in model.modules():
        if hasattr(module, "_saved_weight_ref"): del module._saved_weight_ref
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("+++++++++++++++++++++++++++++++++++++++++++++\n")

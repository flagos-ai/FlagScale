import json
import os


def llama_cpp_rope_scaling_converter(v) -> dict:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string: {v}")
    if not isinstance(v, dict):
        raise ValueError(f"Expected a dictionary, got {type(v)}: {v}")
    rope_type = v.get("rope_type")
    factor = v.get("factor")
    if rope_type not in ["none", "linear", "yarn"]:
        raise ValueError(f"Invalid rope_type: {rope_type}")
    if factor is None:
        raise ValueError(f"Missing factor in the dictionary: {v}")
    return {"rope_scaling": rope_type, "rope_scale": factor}


def llama_cpp_kv_cache_dtype_converter(v) -> dict:
    # llama.cpp supports f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
    if v in ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]:
        return {"cache_type_k": v, "cache_type_v": v}
    # vllm supports fp8, fp8_e4m3, fp8_e5m2
    if v in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        return {"cache_type_k": "q8_0", "cache_type_v": "q8_0"}
    raise ValueError(f"Invalid kv_cache_dtype for llama.cpp: {v}")


def llama_cpp_reasoning_parser_converter(v) -> dict:
    if v in ["deepseek", "deepseek_r1", "deep-seek", "DeepSeek", "Deep-Seek"]:
        return {"reasoning_format": "deepseek"}
    raise ValueError(f"Invalid reasoning_parser for llama.cpp: {v}")


def llama_cpp_max_model_len_converter(v) -> int:
    if isinstance(v, int):
        return {"ctx_size": v}
    if not isinstance(v, str):
        raise ValueError(f"Invalid max_model_len: {v, type(v)}")

    if "k" in v:
        v = v.replace("k", "")
        try:
            v = int(v) * 1000
        except TypeError:
            raise TypeError(f"Invalid max_model_len: {v, type(v)}")
    elif "K" in v:
        v = v.replace("K", "")
        try:
            v = int(v) * 1024
        except TypeError:
            raise TypeError(f"Invalid max_model_len: {v, type(v)}")
    if not isinstance(v, int):
        raise ValueError(f"Expected an integer, got {type(v)}: {v}")
    return {"ctx_size": v}


def llama_cpp_uvicorn_log_level_converter(v) -> dict:
    if v in ["warning", "error", "critical", "trace", "WARNING", "ERROR", "CRITICAL", "TRACE"]:
        return {"log_verbosity": 0}
    elif v in ["debug", "info", "DEBUG", "INFO"]:
        return {"log_verbosity": 1}
    raise ValueError(f"Invalid uvicorn_log_level(log_verbosity) for llama.cpp: {v}")


def llama_cpp_model_converter(v) -> dict:
    if not os.path.exists(v):
        raise FileNotFoundError(f"Error: model_path '{v}' does not exist")
    if not os.path.isdir(v):
        raise NotADirectoryError(f"Error: model_path '{v}' is not a directory")
    for root, _, files in os.walk(v):
        for file in files:
            if file.lower().endswith(".gguf"):
                full_path = os.path.join(root, file)
                return {"model": full_path}
    return {"model": v}

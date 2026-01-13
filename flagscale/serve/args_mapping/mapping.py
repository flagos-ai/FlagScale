import importlib
from pathlib import Path

import yaml


def args2func(backend_name: str, args: str) -> str:
    """
    This function generates a function name based on the backend name and args.
    It creates a function name that follows the pattern "[backend_name]_[args]_converter".
    """
    return f"{backend_name}_{args}_converter"


def func2args(backend_name: str, func_name: str) -> str:
    """
    This function generates an argument name based on the backend name and function name.
    It resolves an argument name that follows the pattern "[backend_name]_[args]_converter".
    """
    if backend_name not in func_name:
        raise ValueError(f"Backend name {backend_name} not found in function name {func_name}")
    if not func_name.endswith("_converter"):
        raise ValueError(f"Function name {func_name} does not end with '_converter'")
    args_name = func_name[:-10]
    args_name = args_name[len(backend_name) + 1 :]
    return args_name


class ArgsConverter:
    """
    ArgsConverter is a class that handles the conversion of arguments
    between different backends. It loads the mapping configuration from a mapping.yaml
    and provides methods to convert arguments based on the mapping.
    """

    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        yaml_path = base_dir / "mapping.yaml"
        with open(yaml_path, "r") as f:
            self.mapping = yaml.safe_load(f)
            for backend_name in self.mapping:
                self.mapping[backend_name]["kv_mapping_func_loaded"] = {}
                self.load_funcs(backend_name)

    def load_funcs(self, backend_name: str):
        if backend_name not in self.mapping:
            raise ValueError(f"Backend name {backend_name} not found in mapping.yaml")
        mapping_conf = self.mapping[backend_name]
        kv_mapping_func = mapping_conf.get("kv_mapping_func", [])
        if not kv_mapping_func:
            return
        for args_name in kv_mapping_func:
            func_module = importlib.import_module(
                f"flagscale.serve.args_mapping.mapping_funcs.{backend_name}"
            )
            func_name = args2func(backend_name=backend_name, args=args_name)
            func = getattr(func_module, func_name)
            self.mapping[backend_name]["kv_mapping_func_loaded"][args_name] = func

    def convert(self, backend_name: str, src_map: dict) -> dict:
        if backend_name not in self.mapping:
            raise ValueError(f"Backend name {backend_name} not found in mapping.yaml")
        mapping_conf = self.mapping[backend_name]
        key_mapping = mapping_conf.get("key_mapping", {})
        kv_mapping_func = mapping_conf.get("kv_mapping_func", [])

        dst_map = {}
        for key, value in src_map.items():
            if key in key_mapping:
                dst_key = key_mapping[key]
                dst_map[dst_key] = value
            elif key in kv_mapping_func:
                dst_func = mapping_conf["kv_mapping_func_loaded"][key]
                new_kvs = dst_func(value)
                dst_map.update(new_kvs)
            else:
                dst_map[key] = value
        return dst_map


ARGS_CONVERTER = ArgsConverter()

if __name__ == "__main__":
    src_map = {
        "model": "llama-7b",
        "max_model_len": "1K",
        "max_num_seqs": 4,
        "kv_cache_dtype": "f16",
        "uvicorn_log_level": "warning",
    }
    print(ARGS_CONVERTER.convert("llama_cpp", src_map))

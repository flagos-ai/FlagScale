# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# It creates a py file named "template_funcs.py" with functions that are
# responsible for converting the key-value to the desired format
import argparse

import yaml
from mapping import args2func


def gen_single_func(func_name):
    """
    This function generates a single function definition for the given function name.
    It creates a function that takes a value, checks its type, and returns a dictionary.
    """
    lines = []
    lines.append(f"def {func_name}(v) -> dict:\n")
    lines.append("     # Do mapping here: a vllm style kv -> new backend style kvs\n")
    lines.append('    return {"NEW_KEY": "NEW_VALUE"}\n')
    return lines


def gen_template_funcs(backend_name):
    """
    This function generates a template functions file.
    The configuration is saved to a file named "template_funcs.py".
    """
    lines = ["# This file is auto-generated, edit it properly.\n"]
    with open("flagscale/serve/args_mapping/mapping.yaml", "r") as f:
        conf = yaml.safe_load(f)
        if backend_name not in conf:
            raise ValueError(f"Backend name {backend_name} not found in mapping.yaml")
        conf = conf[backend_name]
        if "kv_mapping_func" not in conf:
            raise ValueError(
                f"kv_mapping_func not found in mapping.yaml for backend {backend_name}"
            )
        kv_mapping_func = conf["kv_mapping_func"]
        if not kv_mapping_func:
            raise ValueError(f"kv_mapping_func is empty in mapping.yaml for backend {backend_name}")
        for i in kv_mapping_func:
            lines.append("\n")
            lines.extend(gen_single_func(args2func(backend_name=backend_name, args=i)))
            lines.append("\n")

    with open("template_funcs.py", "w") as f:
        f.writelines(lines)
        print("./template_funcs.py generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Generates a template functions file according to the mapping.yaml file and --backend-name.

Example usage:
`python template_funcs.py --backend-name llama_cpp`
This will generate a template functions file for the llama_cpp backend.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--backend-name", type=str, required=True, help="The mlp weights with hf format"
    )
    args = parser.parse_args()

    gen_template_funcs(backend_name=args.backend_name)

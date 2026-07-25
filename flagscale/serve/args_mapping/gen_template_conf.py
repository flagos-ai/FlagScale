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

# It creates a mapping conf of argument names to placeholder values and saves
# the configuration to a file named "template_conf.yaml".
import yaml


def gen_template_conf():
    """
    This function generates a template mapping config file .
    It extracts the argument names from ./common_args creates a mapping configuration.
    The configuration is saved to a file named "template_conf.yaml".
    """
    serve_args = []
    with open("flagscale/serve/args_mapping/common_args.yaml", "r") as f:
        common_args = yaml.safe_load(f)
        serve_args = common_args["common_args"]

    print("len of common serve args: ", len(serve_args))
    conf = {"NEW_BACKEND_NAME": {"key_mapping": {}, "kv_mapping_func": []}}
    for i in serve_args:
        conf["NEW_BACKEND_NAME"]["key_mapping"][i] = "TODO"
    conf["NEW_BACKEND_NAME"]["kv_mapping_func"].append("TODO")

    with open("template_conf.yaml", "w") as f:
        yaml.dump(conf, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    gen_template_conf()
    print("./template_conf.yaml generated")
    print("Please update the keys with TODO values in the template_conf.yaml file.")

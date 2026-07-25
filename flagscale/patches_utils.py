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

# this file is used for adding tools func  to processing patches


def add_patches_module(path: str, module_dict: dict):
    if len(module_dict) == 0:
        raise Exception("module dict is None")
    import sys

    print(f"{path} is being instead, using module {module_dict}")
    for k in sys.modules:
        if k.startswith(path):
            for module_name, module_ in module_dict.items():
                import re

                class_pattern = re.compile("\w*\.w*")
                if not re.match(class_pattern, module_name):
                    try:
                        if getattr(sys.modules[k], module_name, None):
                            setattr(sys.modules[k], module_name, module_)
                    except:
                        raise RuntimeError("module_name format must be right!")
                else:
                    class_name, fuc_name = module_name.split(".")
                    class_obj = getattr(sys.modules[k], class_name, None)
                    if class_obj and getattr(class_obj, fuc_name, None):
                        setattr(class_obj, fuc_name, module_)

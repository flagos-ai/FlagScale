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

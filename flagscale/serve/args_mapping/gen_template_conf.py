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

import sys

import yaml


def parse_config(config_file, backend, subset):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

        backend_config = config.get(backend, {})
        config_str = ""
        config_str += ";".join(backend_config.get("set_environment", [])) + "|"
        config_str += f"{backend_config.get('root', '')}" + "|"
        config_str += f"{backend_config.get('coverage_fold', '')}" + "|"

        if subset not in backend_config.get("subset", {}):
            print(config_str + "|||")
            return

    if subset not in backend_config.get("subset", {}):
        print(config_str + "|||")
        return

    config = backend_config.get("subset", {}).get(subset, {})

    config_str += f"{config.get('type', [])}|" if "type" in config else "|"
    config_str += f"{config.get('depth', [])}|" if "depth" in config else "|"
    config_str += f"{config.get('ignore', [])}|" if "ignore" in config else "|"
    config_str += f"{config.get('deselect', [])}|" if "deselect" in config else "|"

    # Output the set_environment, root, and ignore as separate values, separated by "|"
    print(config_str)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: parse_config.py <config_file> <backend> <subset>")
        sys.exit(1)

    config_file = sys.argv[1]
    backend = sys.argv[2]
    subset = sys.argv[3]

    parse_config(config_file, backend, subset)

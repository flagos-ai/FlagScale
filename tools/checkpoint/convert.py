import argparse
import copy
import importlib
import sys

import torch.multiprocessing as mp
from utils import validate_args


def load_plugin(plugin_type, name):
    module_name = f"{plugin_type}_{name}"
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        print(e)
        module_name = name
        try:
            plugin = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            print(e)
            sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, "add_arguments"):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    print(f"Loaded {module_name} as the {plugin_type}.")
    return plugin


def main():
    parser = argparse.ArgumentParser(
        description="Convert checkpoint", allow_abbrev=False, conflict_handler="resolve"
    )
    # convert args
    parser.add_argument(
        "--model-type",
        type=str,
        default=[],
        nargs="+",
        required=True,
        choices=["aquila", "mistral", "mixtral", "llama", "deepseek_v3", "qwen3"],
        help="Type of the model.",
    )
    parser.add_argument(
        "--loader",
        type=str,
        default="mcore",
        choices=["mcore", "transformers"],
        help="Module name to load checkpoint, should be on python path",
    )
    parser.add_argument(
        "--saver",
        type=str,
        default="mcore",
        choices=["mcore", "transformers"],
        help="Module name to save checkpoint, shdoul be on python path",
    )
    parser.add_argument(
        "--load-dir", type=str, required=True, help="Directory to load model checkpoint from"
    )
    parser.add_argument(
        "--save-dir", type=str, required=True, help="Directory to save model checkpoint to"
    )
    parser.add_argument(
        "--max-queue-size", type=int, default=50, help="Maximum number of tensors in the queue"
    )

    extend_cases = [["mistral", "mixtral"]]

    known_args, _ = parser.parse_known_args()
    loader = load_plugin("loader", known_args.loader)
    saver = load_plugin("saver", known_args.saver)

    loader.add_arguments(parser)
    saver.add_arguments(parser)

    args = parser.parse_args()
    validate_args(args)

    queue = mp.Queue(maxsize=args.max_queue_size)

    print("Starting saver...")
    saver_args = copy.deepcopy(args)
    if len(args.model_type) == 1:
        saver_args.model_type = args.model_type[0]
    elif len(args.model_type) == 2:
        assert args.model_type in extend_cases, f"Only support extend cases are {extend_cases}"
        saver_args.model_type = args.model_type[1]
    else:
        raise ValueError("")
    saver_proc = mp.Process(target=saver.save_checkpoint, args=(queue, saver_args))
    saver_proc.start()

    print("Starting loader...")
    loader_args = copy.deepcopy(args)
    if len(args.model_type) == 1:
        loader_args.model_type = args.model_type[0]
    elif len(args.model_type) == 2:
        assert args.model_type in extend_cases, f"Only support extend cases are {extend_cases}"
        loader_args.model_type = args.model_type[0]
    else:
        raise ValueError("")
    loader.load_checkpoint(queue, loader_args)

    print("Waiting for saver to complete...")
    saver_proc.join()


if __name__ == "__main__":
    main()

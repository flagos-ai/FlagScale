"""Select a rank-local Hugging Face dynamic-module cache in CI workers."""

import importlib.util
import os
import sys


def _configure_rank_cache():
    cache_root = os.environ.get("HF_MODULES_CACHE_ROOT")
    local_rank = os.environ.get("LOCAL_RANK")
    if not cache_root or local_rank is None or not local_rank.isdigit():
        return

    node_rank = os.environ.get("GROUP_RANK", "0")
    os.environ["HF_MODULES_CACHE"] = os.path.join(cache_root, f"node_{node_rank}_rank_{local_rank}")


_configure_rank_cache()


def _load_previous_sitecustomize():
    """Preserve sitecustomize hooks supplied by the runtime image."""
    current_file = os.path.realpath(__file__)
    for entry in sys.path:
        candidate = os.path.realpath(os.path.join(entry or os.curdir, "sitecustomize.py"))
        if candidate == current_file or not os.path.isfile(candidate):
            continue

        spec = importlib.util.spec_from_file_location(
            "_flagscale_previous_sitecustomize", candidate
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        break


_load_previous_sitecustomize()

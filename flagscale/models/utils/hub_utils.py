import hashlib
import os
import tempfile
from pathlib import Path

import filelock

from flagscale.logger import logger

_lock_dir = tempfile.gettempdir()


def use_modelscope() -> bool:
    return os.environ.get("FLAGSCALE_USE_MODELSCOPE", "false").lower() == "true"


# Copied from https://github.com/vllm-project/vllm/blob/1fc69f59bb0838c2ff6efc416dd8875c3e210d04/vllm/model_executor/model_loader/weight_utils.py
def _get_lock(model_name_or_path: str, cache_dir: str | None = None) -> filelock.FileLock:
    lock_dir = cache_dir or _lock_dir
    os.makedirs(lock_dir, exist_ok=True)
    model_name = str(model_name_or_path).replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()[:16]
    # add hash to avoid conflict with old users' lock files
    lock_file = os.path.join(lock_dir, f"{hash_name}-{model_name}.lock")
    # mode 0o666 is required for the filelock to be shared across users
    return filelock.FileLock(lock_file, mode=0o666)


def _resolve_hub_path(
    name_or_path: str,
    repo_type: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
) -> str:
    """Resolve a local path or HF/ModelScope repo ID to a local directory path."""
    if use_modelscope():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if Path(name_or_path).is_dir():
        logger.info(f"{repo_type.capitalize()} path is local directory: {name_or_path}")
        return name_or_path

    with _get_lock(f"{repo_type}:{name_or_path}", cache_dir):
        if use_modelscope():
            logger.info(f"Downloading {repo_type} from ModelScope: {name_or_path}")
            from modelscope.hub.snapshot_download import snapshot_download

            local_path = snapshot_download(
                model_id=name_or_path,
                repo_type=repo_type,
                cache_dir=cache_dir,
                revision=revision,
                ignore_file_pattern=ignore_patterns,
                allow_patterns=allow_patterns,
            )
        else:
            logger.info(f"Downloading {repo_type} from HuggingFace Hub: {name_or_path}")
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(
                name_or_path,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

    if not local_path:
        raise RuntimeError(
            f"Failed to download {repo_type} '{name_or_path}': "
            "snapshot_download returned an empty path"
        )

    logger.info(f"{repo_type.capitalize()} resolved to: {local_path}")
    return local_path


def resolve_model_path(
    model_name_or_path: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
) -> str:
    """Resolve a model name or HF/ModelScope repo ID to a local directory path.

    If ``model_name_or_path`` is already a local directory, returns it as-is.
    Otherwise downloads the model repo and returns the local cache path.
    """
    return _resolve_hub_path(
        model_name_or_path,
        repo_type="model",
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def resolve_dataset_path(
    dataset_name_or_path: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
) -> str:
    """Resolve a dataset name or HF/ModelScope repo ID to a local directory path.

    If ``dataset_name_or_path`` is already a local directory, returns it as-is.
    Otherwise downloads the dataset repo and returns the local cache path.
    """
    return _resolve_hub_path(
        dataset_name_or_path,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

import argparse
import os
import shutil
import tempfile

import git
import yaml

from encryption_utils import decrypt_file
from git.repo import Repo
from logger_utils import get_unpatch_logger
from patch import normalize_backend

FLAGSCALE_BACKEND = "FlagScale"
logger = get_unpatch_logger()


def apply_patches_from_directory(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        logger.warning(f"Patch directory '{src_dir}' does not exist. Nothing to apply.")
        return

    try:
        repo = Repo(dst_dir)
        for root, _, files in os.walk(src_dir):
            for file in sorted(files):
                if file.endswith(".patch"):
                    patch_file_path = os.path.join(root, file)
                    logger.info(f"Applying patch: {patch_file_path}")
                    try:
                        repo.git.apply(patch_file_path, check=True)
                        repo.git.apply(patch_file_path)
                    except git.exc.GitCommandError as e:
                        logger.error(
                            f"Failed to apply patch '{patch_file_path}'. Error: {e.stderr}"
                        )
                        raise e

    except Exception as e:
        logger.error(f"An error occurred while setting up patch application for '{dst_dir}': {e}")
        raise e


def unpatch(main_path, src, dst, submodule_name, force=False, backend_commit={}, fs_extension=True):
    """Unpatch the backend with patches."""
    if submodule_name != FLAGSCALE_BACKEND:
        logger.info(f"Unpatching backend {submodule_name}...")
        submodule_commit = None
        if backend_commit and backend_commit[submodule_name] is not None:
            submodule_commit = backend_commit[submodule_name]
        init_submodule(main_path, dst, submodule_name, force=force, commit=submodule_commit)
        if fs_extension:
            apply_patches_from_directory(src, dst)
            logger.info(f"Successfully applied all patches to backend {submodule_name}")
        else:
            logger.info(
                f"FlagScale extension for {submodule_name} is disabled, skipping unpatching..."
            )


def init_submodule(main_path, dst, submodule_name, force=False, commit=None):
    if os.path.lexists(dst) and len(os.listdir(dst)) > 0 and not force:
        logger.info(f"Skipping {submodule_name} initialization, as it already lexists.")
        return
    logger.info(f"Initializing submodule {submodule_name}...")
    logger.warning(
        "When you perform unpatch, the specified submodule will be fully restored to its initial state, regardless of any modifications you may have made within the submodule."
    )
    repo = Repo(main_path)
    submodule_name = os.path.join("third_party", submodule_name)
    submodule = repo.submodule(submodule_name)
    retry_times = 2
    for _ in range(retry_times):
        try:
            git_modules_path = os.path.join(main_path, ".git", "modules", submodule_name)
            if os.path.exists(git_modules_path):
                shutil.rmtree(git_modules_path)
            submodule_worktree_path = os.path.join(main_path, submodule_name)
            if os.path.exists(submodule_worktree_path):
                shutil.rmtree(submodule_worktree_path)
            submodule.update(init=True, force=force)
            if commit:
                sub_repo = submodule.module()
                sub_repo.git.reset('--hard', commit)
                logger.info(f"Reset {submodule_name} to commit {commit}.")
            logger.info(f"Initialized {submodule_name} submodule.")
            break

        except Exception as e:
            logger.error(f"Exception occurred: {e}", exc_info=True)
            logger.info(f"Retrying to initialize submodule {submodule_name}...")


def commit_to_checkout(main_path, device_type=None, tasks=None, backends=None, commit=None):
    if commit:
        return commit

    newest_flagscale_commit = None
    main_repo = Repo(main_path)
    if device_type and tasks:
        # Check if device_type is in the format xxx_yyy
        if device_type.count("_") != 1 or len(device_type.split("_")) != 2:
            raise ValueError("Invalid format. Device type must be in the format xxx_yyy.")

        assert backends
        history_yaml = os.path.join(main_path, "hardware", "patch_history.yaml")
        if not os.path.exists(history_yaml):
            logger.warning(
                f"Yaml {history_yaml} does not exist. Please check the hardware/patch_history.yaml."
            )
            logger.warning("Try to use the current commit to unpatch.")
            return main_repo.head.commit.hexsha

        # Backend key
        backends_key = "+".join(sorted(backends))
        # Newest flagscale commit to checkout and unpatch
        newest_flagscale_commit = None
        # Find newest flagscale commit
        with open(history_yaml, 'r') as f:
            history = yaml.safe_load(f)
            if device_type not in history:
                logger.warning(f"Device type {device_type} not found in {history_yaml}.")
                logger.warning("Try to use the current commit to unpatch.")
                return main_repo.head.commit.hexsha

            # Find the newest flagscale commit in the history
            for task in tasks:
                if task not in history[device_type]:
                    continue
                if backends_key not in history[device_type][task]:
                    continue
                if (
                    not isinstance(history[device_type][task][backends_key], list)
                    or not history[device_type][task][backends_key]
                ):
                    continue
                newest_flagscale_commit = history[device_type][task][backends_key][-1]
                try:
                    main_repo.commit(newest_flagscale_commit)
                    break
                except ValueError:
                    raise ValueError(
                        f"The commit ID {newest_flagscale_commit} does not exist in the FlagScale. Please check the {history_yaml}"
                    )
                    newest_flagscale_commit = None
        if not newest_flagscale_commit:
            logger.warning(
                f"No valid commit found for device type {device_type}, task {task} in {history_yaml}. Try to use the current commit to unpatch."
            )
            return main_repo.head.commit.hexsha
    return newest_flagscale_commit


def apply_hardware_patch(
    device_type, backends, commit, main_path, need_init_submodule, key_path=None
):
    build_path = os.path.join(main_path, "build", device_type)
    final_path = os.path.join(build_path, os.path.basename(main_path))

    try:
        # Remove existing build directory if present.
        if os.path.exists(build_path):
            logger.info(f"Removing existing build path: {build_path}")
            shutil.rmtree(build_path)

        temp_path = tempfile.mkdtemp()
        logger.info(f"Step 1: Copying {main_path} to temp path {temp_path}")
        shutil.copytree(main_path, temp_path, dirs_exist_ok=True)

        repo = Repo(temp_path)
        # Stash firstly to prevent checkout failed
        repo.git.stash("push", "--include-untracked")
        logger.info(f"Step 2: Checking out {commit} in temp path {temp_path}")
        repo.git.checkout(commit)

        # Check device path
        device_path = os.path.join(temp_path, "hardware", device_type)
        if not os.path.exists(device_path):
            raise ValueError(f"{device_path} is not found.")

        # Check backend path and patch file path
        all_base_commit_id = set()
        patch_files = []
        patch_backends = []
        backends_commit = {}

        for backend in backends:
            backend_path = os.path.join(device_path, backend)
            if not os.path.exists(backend_path):
                raise ValueError(f"{backend_path} is not found.")

            yaml_file = os.path.join(backend_path, "diff.yaml")
            if not os.path.isfile(yaml_file):
                raise ValueError(f"Missing diff.yaml in {backend_path}")

            with open(yaml_file, "r") as f:
                info = yaml.safe_load(f)
                base_commit_id = info["commit"]
                if "backends_commit" in info and backend in info["backends_commit"]:
                    backends_commit[backend] = info["backends_commit"][backend]
                assert base_commit_id
            all_base_commit_id.add(base_commit_id)

            backend_patch_files = []
            for root, _, files in os.walk(backend_path):
                for file in files:
                    if file.endswith(".patch") or file.endswith(".patch.encrypted"):
                        backend_patch_files.append(os.path.join(root, file))
            if not backend_patch_files:
                raise ValueError(f"No patch files found in {backend_path}")
            patch_files.extend(backend_patch_files)
            patch_backends.extend([backend] * len(backend_patch_files))

        all_base_commit_id = list(all_base_commit_id)

        # Sort the commit by appearance order
        position = {}
        rev_list = repo.git.rev_list('--topo-order', 'HEAD').splitlines()
        for idx, commit in enumerate(rev_list):
            if commit in all_base_commit_id:
                position[commit] = idx

        # Check if all commits were found
        missing = set(all_base_commit_id) - set(position.keys())
        if missing:
            raise ValueError(f"The following commits were not found in rev-list: {missing}")

        sorted_commits = sorted(all_base_commit_id, key=lambda x: position[x])
        # Get the neweset base_commit_id
        base_commit_id = sorted_commits[0]
        logger.info(f"Step 3: Finding the newset base commit {base_commit_id} to checkout.")

        temp_unpatch_path = tempfile.mkdtemp()
        logger.info(f"Step 4: Copying {temp_path} to temp unpatch path {temp_unpatch_path}")
        shutil.copytree(temp_path, temp_unpatch_path, dirs_exist_ok=True)
        repo = Repo(temp_unpatch_path)
        repo.git.checkout(base_commit_id)

        logger.info(f"Step 5: Applying patch:")
        initialized_backends = set()
        for idx, patch_file in enumerate(patch_files):
            # Check if the patch file is encrypted
            new_patch_file = patch_file
            if patch_file.endswith(".encrypted"):
                if key_path is not None:
                    private_key_path = os.path.join(key_path, "private_key.pem")
                    new_patch_file = decrypt_file(patch_file, private_key_path)
                else:
                    raise ValueError(
                        f"Patch file {patch_file} is encrypted, but no key path provided."
                    )
            backend = patch_backends[idx]
            if backend != FLAGSCALE_BACKEND:
                # init submodule
                if need_init_submodule:
                    if backend not in initialized_backends:
                        logger.info(
                            f"    Initializing submodule {backend} in temp unpatch path {temp_unpatch_path}..."
                        )
                        dst = os.path.join(temp_unpatch_path, "third_party", backend)
                        src = os.path.join(temp_unpatch_path, "flagscale", "backends", backend)
                        # Initialize the submodule

                        submodule_commit = None
                        if backends_commit and backend in backends_commit:
                            submodule_commit = backends_commit[backend]
                        init_submodule(
                            temp_unpatch_path, dst, backend, force=True, commit=submodule_commit
                        )
                        initialized_backends.add(backend)
            submodule_path = (
                os.path.join(temp_unpatch_path, "third_party", backend)
                if backend != FLAGSCALE_BACKEND
                else temp_unpatch_path
            )

            repo = Repo(submodule_path)
            try:
                repo.git.apply("--whitespace", "warn", new_patch_file)
            except Exception as e:
                logger.warning(
                    f"Failed to apply patch cleanly, and error is {e.stderr}. Retrying with --whitespace=fix."
                )

                repo.git.apply("--whitespace", "fix", new_patch_file)
            logger.info(f"Patch {new_patch_file} has been applied.")

        logger.info(f"Step 6: Moving patched temp path {temp_unpatch_path} to {final_path}")
        os.makedirs(build_path, exist_ok=True)
        shutil.move(temp_unpatch_path, final_path)
        logger.info(f"Unpatch Ended.")

    except Exception as e:
        logger.error(f"Exception occurred: {e}", exc_info=True)

        # Clean up temp directory
        if "temp_path" in locals() and os.path.exists(temp_path):
            logger.info(f"Cleaning up temp path: {temp_path}")
            shutil.rmtree(temp_path, ignore_errors=True)

        # Clean up temp directory
        if "temp_unpatch_path" in locals() and os.path.exists(temp_unpatch_path):
            logger.info(f"Cleaning up temp path: {temp_unpatch_path}")
            shutil.rmtree(temp_unpatch_path, ignore_errors=True)

        # Clean up build directory
        if os.path.exists(build_path):
            logger.info(f"Cleaning up build path: {build_path}")
            shutil.rmtree(build_path, ignore_errors=True)

        raise ValueError("Error occurred during unpatching.")

    finally:
        # Clean up temp directory
        if "temp_path" in locals() and os.path.exists(temp_path):
            logger.info(f"Cleaning up temp path: {temp_path}")
            shutil.rmtree(temp_path, ignore_errors=True)

        # Clean up temp directory
        if "temp_unpatch_path" in locals() and os.path.exists(temp_unpatch_path):
            logger.info(f"Cleaning up temp path: {temp_unpatch_path}")
            shutil.rmtree(temp_unpatch_path, ignore_errors=True)

    return final_path


def validate_unpatch_args(device_type, tasks, commit, main_path):
    main_repo = Repo(main_path)
    if commit:
        # Check if the commit exists in the FlagScale
        try:
            main_repo.commit(commit)
        except ValueError:
            raise ValueError(f"Commit {commit} does not exist in the FlagScale.")
    if device_type:
        if (
            device_type.count("_") != 1
            or len(device_type.split("_")) != 2
            or not device_type.split("_")[0]
            or not device_type.split("_")[0][0].isupper()
        ):
            raise ValueError("Device type is invalid!")

    if device_type or tasks:
        assert device_type and tasks, "The args device_type, task must not be None."


def backend_commit_mapping(backends, backends_commit):
    backend_commit = {}
    for idx, backend in enumerate(backends):
        if backend == FLAGSCALE_BACKEND:
            assert backends_commit == [None], "FlagScale backend commit must be None."
        else:
            if idx >= len(backends_commit):
                backend_commit[backend] = None
            else:
                backend_commit[backend] = backends_commit[idx]

    return backend_commit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch or unpatch backend with patch files.")
    parser.add_argument(
        "--backend",
        nargs="+",
        type=normalize_backend,
        default=["Megatron-LM"],
        help="Backend to unpatch (default: Megatron-LM)",
    )

    parser.add_argument(
        "--device-type", type=str, default=None, help="Device type. Default is None."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=None,
        choices=["train", "inference", "post_train"],
        help="Task. Default is None",
    )
    parser.add_argument(
        "--commit", type=str, default=None, help="Unpatch based on this commit. Default is None."
    )
    parser.add_argument(
        '--no-force', dest='force', action='store_false', help='Do not force update the backend.'
    )
    parser.add_argument(
        "--no-init-submodule",
        action="store_false",
        dest="init_submodule",
        help="Do not initialize and update submodules. Default is True.",
    )
    parser.add_argument(
        "--key-path",
        type=str,
        default=None,
        help="The path for storing public and private keys. Be careful not to upload to the Git repository.",
    )
    parser.add_argument(
        "--no-fs-extension",
        action="store_false",
        dest="fs_extension",
        help="Disable fs extension. Default is True.",
    )
    parser.add_argument(
        "--backend-commit", nargs="+", default=[None], help="The backend commit to checkout."
    )

    args = parser.parse_args()
    backends = args.backend
    device_type = args.device_type
    tasks = args.task
    commit = args.commit
    key_path = args.key_path
    backends_commit = args.backend_commit
    fs_extension = args.fs_extension

    if not isinstance(backends, list):
        backends = [backends]

    if not isinstance(backends_commit, list):
        backends_commit = [backends_commit]

    if tasks is not None and not isinstance(tasks, list):
        tasks = [tasks]

    # FlagScale/tools/patch
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # FlagScale/tools
    script_dir = os.path.dirname(script_dir)
    # FlagScale
    main_path = os.path.dirname(script_dir)

    validate_unpatch_args(device_type, tasks, commit, main_path)
    backend_commit = backend_commit_mapping(backends, backends_commit)

    if FLAGSCALE_BACKEND in backends:
        assert (
            device_type is not None
        ), "FlagScale unpatch only can be applied with hardware unpatch."

    # Check patch exist
    commit = commit_to_checkout(main_path, device_type, tasks, backends, commit)
    if commit is not None:
        # Checkout to the commit and apply the patch to build FlagScale
        apply_hardware_patch(
            device_type, backends, commit, main_path, args.init_submodule, key_path=key_path
        )

    else:
        for backend in backends:
            dst = os.path.join(main_path, "third_party", backend)
            src = os.path.join(main_path, "flagscale", "backends", backend)
            unpatch(
                main_path,
                src,
                dst,
                backend,
                force=args.force,
                backend_commit=backend_commit,
                fs_extension=fs_extension,
            )

# -*- coding: utf-8 -*-
import argparse
import copy
import os
import re
import shutil
import tempfile

import git
import yaml

from encryption_utils import encrypt_file
from git.repo import Repo
from git_utils import (
    check_git_user_info,
    get_diff_between_commit_and_now,
    get_file_statuses_for_staged_or_unstaged,
    get_file_statuses_for_untracked,
    get_submodule_commit,
)
from logger_utils import get_patch_logger

FLAGSCALE_BACKEND = "FlagScale"
logger = get_patch_logger()


def generate_and_save_patch(sub_repo, base_commit, file_path, status, src_dir):

    patch_content = ""
    try:
        if status in ['A', 'UT']:
            patch_content = sub_repo.git.diff('--no-index', '/dev/null', file_path)

        elif status in ['M', 'T', 'D']:
            patch_content = sub_repo.git.diff(base_commit, '--', file_path)
    except git.exc.GitCommandError as e:
        if e.status == 1:
            raw_output = str(e.stdout)
            start_marker = "diff --git"
            start_index = raw_output.find(start_marker)
            end_index = raw_output.rfind("'")
            patch_content = raw_output[start_index:end_index]
        else:
            raise e

    if patch_content:
        target_patch_path = os.path.join(src_dir, file_path + ".patch")
        os.makedirs(os.path.dirname(target_patch_path), exist_ok=True)

        with open(target_patch_path, 'w', encoding='utf-8') as f:
            content = patch_content if patch_content else ""
            if content and not content.endswith('\n'):
                content += '\n'
            f.write(content)
        logger.info(f"Generated patch for '{file_path}' (Status: {status})")
    else:
        logger.warning(f"No patch content generated for '{file_path}' (Status: {status})")


def patch(main_path, submodule_name, src, dst):
    """
    Sync the submodule modifications to the corresponding backend in FlagScale.
    Args:
        main_path (str): The path to the repository.
        submodule_name (str): The name of the submodule to be patched, e.g., "Mgeatron-LM".
        src (str): The source directory of the submodule, e.g., "flagscale/backends/Megatron-LM".
        dst (str): The destination directory of the submodule, e.g., "third_party/Megatron-LM".
    """

    submodule_path = os.path.join("third_party", submodule_name)
    logger.info(f"Patching backend {submodule_path}...")

    # Get the submodule repo and the commit in FlagScale.
    main_repo = Repo(main_path)
    submodule = main_repo.submodule(submodule_path)
    sub_repo = submodule.module()
    # Get the submodule commit in FlagScale by FlagScale HEAD instead of the submodule HEAD.
    submodule_commit_in_fs = main_repo.head.commit.tree[submodule_path].hexsha
    logger.info(f"Base commit hash of submodule {submodule_path} is {submodule_commit_in_fs}.")

    # Get all differences between the submodule specified commit and now.
    # The differences include staged, working directory, and untracked files.
    staged_diff, unstaged_diff, untracked_files = get_diff_between_commit_and_now(
        sub_repo, submodule_commit_in_fs
    )

    file_statuses = {}
    # Process the diff between the staged and the base commit
    staged_file_statuses = get_file_statuses_for_staged_or_unstaged(staged_diff)
    file_statuses.update(staged_file_statuses)
    # Process the diff between the working directory and the staged
    unstaged_file_statuses = get_file_statuses_for_staged_or_unstaged(unstaged_diff)
    file_statuses.update(unstaged_file_statuses)
    # Process the untracked files
    untracked_file_statuses = get_file_statuses_for_untracked(untracked_files)
    file_statuses.update(untracked_file_statuses)

    try:
        # create temporary path
        if os.path.exists(src):
            temp_path = tempfile.mkdtemp()
            shutil.copytree(src, temp_path, dirs_exist_ok=True)
            logger.info(f"Created a temporary backup of '{src}' at '{temp_path}'")

        logger.info(f"Cleaning up old patch directory: {src}")
        shutil.rmtree(src, ignore_errors=True)
        os.makedirs(src)

        if not file_statuses:
            logger.info("No file changes detected. Nothing to patch.")

        else:
            logger.info(f"Found {len(file_statuses)} file change(s). Generating patches...")
            for file_path, status_info in file_statuses.items():
                status = status_info[0]
                generate_and_save_patch(sub_repo, submodule_commit_in_fs, file_path, status, src)
            logger.info("Patch generation completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred during patch generation: {e}", exc_info=True)
        shutil.rmtree(src, ignore_errors=True)
        shutil.copytree(temp_path, src, dirs_exist_ok=True)

    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            logger.info(f"Cleaning up temp path: {temp_path}")
            shutil.rmtree(temp_path, ignore_errors=True)


def patch_hardware(main_path, commit, backends, device_type, tasks, key_path=None):
    assert commit is not None, "The commit hash must be specified for hardware patch."
    assert backends is not None, "The backends must be specified for hardware patch."
    assert device_type is not None, "The device type must be specified for hardware patch."
    assert tasks is not None, "The tasks must be specified for hardware patch."

    patch_info = prompt_info(main_path, backends, device_type, tasks)
    generate_patch_file(main_path, commit, patch_info, key_path=key_path)


def prompt_info(main_path, backends, device_type, tasks):
    logger.info("Prompting for patch information: ")

    backends_version = {}
    logger.info("1. Please enter backends version: ")
    for backend in backends:
        version = input(f"    {backend} version: ").strip()
        while not version:
            logger.info(f"Version for {backend} cannot be empty.")
            version = input(f"    {backend} version: ").strip()
        backends_version[backend] = version

    backends_commit = {}
    logger.info("2. Please enter backends commit: ")
    logger.info(
        "If a specific submodule commit is provided, it will be used to generate the diff and apply the patch. By default, the commit defined by FlagScale will be used."
    )
    for backend in backends:
        if backend == FLAGSCALE_BACKEND:
            continue
        commit = input(f"    {backend} commit (optional): ").strip()
        commit = get_submodule_commit(commit, backend, main_path)
        backends_commit[backend] = commit

    # FlagScale-compatible models
    model_input = input(
        "3. Please enter FlagScale-compatible models (separated with commas): "
    ).strip()
    models = [m.strip() for m in model_input.split(",") if m.strip()]
    while not models:
        logger.info("At least one FlagScale-compatible model must be provided.")
        model_input = input(
            "3. Please enter FlagScale-compatible models (separated with commas): "
        ).strip()
        models = [m.strip() for m in model_input.split(",") if m.strip()]

    # 3. Commit message
    commit_msg = input("4. Please enter commit message: ").strip()
    while not commit_msg:
        logger.info("Commit message cannot be empty.")
        commit_msg = input("4. Please enter commit message: ").strip()

    # 4. Contact (optional)
    contact_prompt = "5. Please enter email (optional): "
    contact = input(contact_prompt).strip()

    return {
        "task": tasks,
        "backends_version": backends_version,
        "device_type": device_type,
        "models": models,
        "contact": contact,
        "commit_msg": commit_msg,
        "backends_commit": backends_commit,
    }


def _generate_patch_file_for_backend(
    main_path: str,
    commit: str,
    backend: str,
    patch_info: dict,
    key_path=None,
    flagscale_commit=None,
):
    """
    Generate patch file for a specific backend.
    Args:
        main_path (str): The path to FlagScale.
        commit (str): The commit hash based to patch (default: None).
        backend (str): The backend to patch.
        patch_info (dict): The patch information.
        key_path (str): The path for public and private keys (default: None).
        flagscale_commit (str): The commit hash in FlagScale (default: None).
    """
    repo = Repo(main_path)
    assert not repo.bare
    try:
        patch_dir = os.path.join(main_path, "hardware", patch_info["device_type"], backend)
        repo_path = (
            os.path.join(main_path, "third_party", backend)
            if backend != FLAGSCALE_BACKEND
            else main_path
        )
        repo = Repo(repo_path)
        current_branch = repo.active_branch.name
        if repo.bare:
            raise Exception("Repository is bare. Cannot proceed.")

        logger.info(f"Generating the patch file for {repo_path}:")
        # Step 1: Stash all, including untracked, and create temp_branch_for_hardware_patch
        logger.info(
            "Step 1: Stashing current changes (including untracked) and create the temp_branch_for_hardware_patch..."
        )
        temp_branch = "temp_branch_for_hardware_patch"
        repo.git.stash("push", "--include-untracked")
        stash_pop = False
        if temp_branch in repo.heads:
            logger.info(
                "Temporary branch 'temp_branch_for_hardware_patch' already exists, deleting..."
            )
            repo.git.branch("-D", temp_branch)
        repo.git.checkout("-b", temp_branch)

        # Step2: Apply stash on the temp_branch_for_hardware_patch and add
        logger.info(
            "Step 2: Applying stashed changes and add changes on the temp_branch_for_hardware_patch..."
        )
        repo.git.stash("apply")
        (
            repo.git.add(all=True)
            if backend != FLAGSCALE_BACKEND
            else repo.git.add('--all', f':!third_party')
        )

        # Step3: Commit with message "Patch for {commit}".
        logger.info("Step 3: Committing changes on the temp_branch_for_hardware_patch...")
        repo.git.commit("--no-verify", "-m", f"Patch for {commit}")

        # Step4: Generate patch diff between commit and HEAD, writing into temp_file.
        logger.info(f"Step 4: Generating diff patch from {commit} to HEAD...")

        # Diff excludes the submodules
        flagscale_diff_args = None
        if backend == FLAGSCALE_BACKEND:
            backends = copy.deepcopy(list(patch_info["backends_version"].keys()))
            flagscale_diff_args = [commit, "HEAD", "--binary", "--ignore-submodules=all", "--"]
            for item in backends:
                if item != FLAGSCALE_BACKEND:
                    backend_dir = os.path.join(main_path, "flagscale", "backends", item)
                    flagscale_diff_args.append(f':(exclude){backend_dir}')

        diff_data = (
            repo.git.diff(commit, "HEAD", "--binary")
            if backend != FLAGSCALE_BACKEND
            else repo.git.diff(*flagscale_diff_args)
        )
        if not diff_data:
            raise ValueError(f"No changes in backend {backend}.")

        splits = re.split(r'(?=^diff --git a/)', diff_data, flags=re.MULTILINE)

        temp_patch_files_with_relpath = []
        for file_diff in splits:
            if not file_diff.strip():
                continue

            m = re.match(r'diff --git a/.+ b/(.+)', file_diff.splitlines()[0])
            if not m:
                continue
            filepath = m.group(1)  # relative path like megatron/train.py

            temp_file = tempfile.NamedTemporaryFile(
                delete=False, mode="w", encoding="utf-8", suffix=".patch"
            )
            temp_file.write(file_diff)

            temp_file.write("\n")
            temp_file.flush()
            temp_file.close()

            temp_patch_files_with_relpath.append((temp_file.name, filepath))

        if key_path is not None:
            encrypted_files_with_relpath = []
            for patch_path, rel_path in temp_patch_files_with_relpath:
                encrypted_path = encrypt_file(patch_path, key_path)
                encrypted_files_with_relpath.append((encrypted_path, rel_path))
            temp_patch_files_with_relpath = encrypted_files_with_relpath

        temp_yaml_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", encoding="utf-8", suffix=".yaml"
        )
        temp_yaml_path = temp_yaml_file.name
        data = copy.deepcopy(patch_info)
        assert flagscale_commit is not None, "FlagScale commit must be specified."
        data["commit"] = flagscale_commit
        if "commit_msg" in data:
            del data["commit_msg"]
        yaml.dump(data, temp_yaml_file, sort_keys=True, allow_unicode=True)
        temp_yaml_file.flush()

        # Step5: Checkout to current branch and pop the stash.
        logger.info(f"Step 5: Checkouting to current branch and pop the stash...")
        repo.git.checkout(current_branch)
        repo.git.stash("pop")
        stash_pop = True

    except Exception as e:
        logger.error(f"{e}", exc_info=True)
        logger.info(f"Rolling back to current branch...")
        repo.git.checkout(current_branch)
        if "stash_pop" in locals() and not stash_pop:
            repo.git.stash("pop")
            stash_pop = True

    finally:
        try:
            # Clean up the temporary branch
            if "temp_branch" in locals() and temp_branch in repo.heads:
                repo.git.branch("-D", temp_branch)

        except Exception as cleanup_error:
            logger.error(f"Failed to roll back: {cleanup_error}", exc_info=True)
            raise cleanup_error

    return patch_dir, temp_patch_files_with_relpath, temp_yaml_path


def generate_patch_file(main_path: str, commit: str, patch_info: dict, key_path=None):
    repo = Repo(main_path)
    assert not repo.bare
    temp_patch_files = []

    """
    This function performs the following steps.
    """
    try:
        backends = copy.deepcopy(list(patch_info["backends_version"].keys()))
        patches = {}
        for backend in backends:
            # Generate patch file for each backend
            if backend != FLAGSCALE_BACKEND:
                # Get the submodule repo and the commit in FlagScale.
                main_repo = Repo(main_path)
                submodule_path = os.path.join("third_party", backend)
                submodule = main_repo.submodule(submodule_path)
                submodule_commit_in_fs = repo.head.commit.tree[submodule_path].hexsha
                if backend in patch_info["backends_commit"]:
                    submodule_commit_in_fs = patch_info["backends_commit"][backend]
                patch_dir, patch_files_with_relpath, temp_yaml_path = (
                    _generate_patch_file_for_backend(
                        main_path,
                        submodule_commit_in_fs,
                        backend,
                        patch_info,
                        key_path=key_path,
                        flagscale_commit=commit,
                    )
                )
            else:
                patch_dir, patch_files_with_relpath, temp_yaml_path = (
                    _generate_patch_file_for_backend(
                        main_path,
                        commit,
                        backend,
                        patch_info,
                        key_path=key_path,
                        flagscale_commit=commit,
                    )
                )
            patches[backend] = [patch_dir, patch_files_with_relpath, temp_yaml_path]

        logger.info(f"Checking out {commit}...")
        repo.git.checkout(commit)

        logger.info("Staging the generated patch files...")
        patch_dir_need_to_clean = []
        for backend in patches:
            patch_dir, patch_files_with_relpath, temp_yaml_path = patches[backend]
            patch_dir_exist = os.path.exists(patch_dir)

            if not patch_dir_exist:
                patch_dir_need_to_clean.append(patch_dir)

            # Remove the backend
            if os.path.exists(patch_dir):
                shutil.rmtree(patch_dir)
                repo.git.rm('-r', patch_dir, ignore_unmatch=True)
            os.makedirs(patch_dir, exist_ok=True)

            # Copy each patch file preserving relative path inside patch_dir
            for temp_patch_path, rel_path in patch_files_with_relpath:
                dst_patch_path = (
                    os.path.join(patch_dir, f"{rel_path}.patch")
                    if key_path is None
                    else os.path.join(patch_dir, f"{rel_path}.patch.encrypted")
                )
                os.makedirs(os.path.dirname(dst_patch_path), exist_ok=True)
                shutil.copy(temp_patch_path, dst_patch_path)
                repo.git.add(dst_patch_path)
                temp_patch_files.append(temp_patch_path)

            # Copy yaml file
            yaml_file_name = "diff.yaml"
            dst_yaml_path = os.path.join(patch_dir, yaml_file_name)
            shutil.copy(temp_yaml_path, dst_yaml_path)
            repo.git.add(dst_yaml_path)
            temp_patch_files.append(temp_yaml_path)

        # Commit the patch file with the same message.
        logger.info("Committing the patch file...")
        commit_msg = patch_info["commit_msg"]
        repo.git.commit("--no-verify", "-m", commit_msg)

        logger.info(
            "Commit successfully! If you want to push, try 'git push origin HEAD:refs/heads/(your branch)' or  'git push --force origin HEAD:refs/heads/(your branch)'"
        )

    except Exception as e:
        logger.error(f"{e}", exc_info=True)

    finally:
        try:
            # Clean up the temp files
            if "temp_patch_files" in locals():
                for temp_patch_path in temp_patch_files:
                    if os.path.exists(temp_patch_path):
                        os.remove(temp_patch_path)
                        logger.debug(f"Temporary patch file {temp_patch_path} deleted.")

            # Clean up the patch dir
            if "patch_dir_need_to_clean" in locals():
                for patch_dir in patch_dir_need_to_clean:
                    if os.path.exists(patch_dir):
                        shutil.rmtree(patch_dir)
                        logger.debug(f"Temporary patch dir {patch_dir} deleted.")

        except Exception as cleanup_error:
            logger.error(f"Failed to delete temporary: {cleanup_error}", exc_info=True)
            raise cleanup_error


def validate_patch_args(device_type, task, commit, main_path):
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

    if commit or device_type or task:
        assert (
            commit and device_type and task
        ), "The args commit, device_type, task must not be None."


def domain_to_backends(domain) -> str:
    """
    Convert domain to backends
    Args:
        domain (str): Domain name provided by the user.
    Returns:
        str: Backend names separated by commas.
    """
    input_lower = domain.lower()

    if input_lower in ["robo", "robotics"]:
        return ",".join(["Megatron-LM", "Megatron-Energon"])
    else:
        raise ValueError(f'Unsupported domain {domain}')


def normalize_backend(backend):
    """
    Normalize backend to standard backend names

    Args:
        backend (str): Backend name provided by the user.

    Returns:
        str: Standardized backend name.
    """

    input_lower = backend.lower()

    if input_lower in ["megatron", "megatron-lm"]:
        return "Megatron-LM"
    elif input_lower in ["energon", "megatron-energon"]:
        return "Megatron-Energon"
    elif input_lower in ["fs", "flagscale"]:
        return "FlagScale"
    elif input_lower == "vllm":
        return "vllm"
    elif input_lower == "sglang":
        return "sglang"
    elif input_lower in ["llama.cpp", "llama_cpp"]:
        return "llama.cpp"
    elif input_lower in ["omniinfer", "omni_infer", "OmniInfer"]:
        return "omniinfer"
    elif input_lower in ["verl"]:
        return "verl"

    raise ValueError(f'Unsupported backend {backend}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync submodule modifications to the corresponding backend in FlagScale."
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        type=normalize_backend,
        default=["Megatron-LM"],
        help="Backend to patch (default: Megatron-LM)",
    )
    parser.add_argument(
        "--commit", type=str, default=None, help="Patch based on this commit. Default is None."
    )
    parser.add_argument(
        "--device-type", type=str, default=None, help="Device type. Default is None."
    )
    parser.add_argument("--task", nargs="+", default=None, help="Task. Default is None")
    parser.add_argument(
        "--key-path",
        type=str,
        default=None,
        help="The path for storing public and private keys. Be careful not to upload to the Git repository.",
    )

    args = parser.parse_args()
    backends = args.backend
    commit = args.commit
    tasks = args.task
    device_type = args.device_type
    key_path = args.key_path

    if not isinstance(backends, list):
        backends = [backends]

    if tasks is not None and not isinstance(tasks, list):
        tasks = [tasks]

    # FlagScale/tools/patch
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # FlagScale/tools
    script_dir = os.path.dirname(script_dir)
    # FlagScale
    main_path = os.path.dirname(script_dir)

    check_git_user_info(main_path)

    validate_patch_args(device_type, tasks, commit, main_path)

    if FLAGSCALE_BACKEND in backends:
        assert commit is not None, "FlagScale patch only can be generated with hardware."

    if commit:
        # hardware patch
        patch_hardware(main_path, commit, backends, device_type, tasks, key_path=key_path)
    else:
        for backend in backends:
            dst = os.path.join(main_path, "third_party", backend)
            src = os.path.join(main_path, "flagscale", "backends", backend)
            patch(main_path, backend, src, dst)

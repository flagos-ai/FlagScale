import os
import subprocess
import sys

from urllib.parse import quote

import click
import yaml

from packaging.version import Version

VERSION = "0.8.0"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(VERSION, "-v", "--version", message=f"flagscale version {VERSION}")
def flagscale():
    """
    FlagScale is a comprehensive toolkit designed to support the entire lifecycle of large models,
    developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI).
    """
    pass


@flagscale.command()
@click.argument("model_name", type=str)
@click.argument("yaml_path", type=click.Path(exists=True), required=False)
def train(model_name, yaml_path=None):
    """
    Train model from yaml.
    """
    from flag_scale.run import main as run_main

    if yaml_path:
        if os.path.isabs(yaml_path):
            yaml_path = yaml_path
        else:
            yaml_path = os.path.join(os.getcwd(), yaml_path)
        if not os.path.exists(yaml_path):
            click.echo(f"Error: The yaml {yaml_path} does not exist.", err=True)
            return
        click.echo(f"Using configuration yaml: {yaml_path}")
    else:
        default_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        # Default is train.yaml
        yaml_path = os.path.join(default_dir, "examples", model_name, "conf", f"train.yaml")
        if not os.path.exists(yaml_path):
            click.echo(f"Error: The yaml {yaml_path} does not exist.", err=True)
            return
        click.echo(f"Using default configuration yaml: {yaml_path}")
    click.echo(f"Start training from the yaml {yaml_path}...")
    yaml_path = os.path.abspath(yaml_path)
    config_path = os.path.dirname(yaml_path)
    config_name = os.path.splitext(os.path.basename(yaml_path))[0]
    click.echo(f"config_path: {config_path}")
    click.echo(f"config_name: {config_name}")

    sys.argv = ["run.py", f"--config-path={config_path}", f"--config-name={config_name}"]
    run_main()


@flagscale.command()
@click.argument("model_name", type=str)
@click.argument("yaml_path", type=click.Path(exists=True), required=False)
@click.option(
    "--model-path", "model_path", required=False, type=str, help="The weight path of model"
)
@click.option("--port", "port", required=False, type=int, help="The port of serve")
@click.option(
    "--engine-args",
    "engine_args",
    required=False,
    type=str,
    help="Model config as JSON string, e.g. '{\"a\":1, \"b\":2}'",
)
def serve(model_name, yaml_path=None, model_path=None, port=None, engine_args=None):
    """
    Serve model from yaml.
    """
    from flag_scale.run import main as run_main

    if yaml_path:
        if os.path.isabs(yaml_path):
            yaml_path = yaml_path
        else:
            yaml_path = os.path.join(os.getcwd(), yaml_path)
        if not os.path.exists(yaml_path):
            click.echo(f"Error: The yaml {yaml_path} does not exist.", err=True)
            return
        click.echo(f"Using configuration yaml: {yaml_path}")
    else:
        default_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        # Default is serve.yaml
        yaml_path = os.path.join(default_dir, "examples", model_name, "conf", f"serve.yaml")
        if not os.path.exists(yaml_path):
            click.echo(f"Error: The yaml {yaml_path} does not exist.", err=True)
            return
        click.echo(f"Using default configuration yaml: {yaml_path}")
    click.secho(
        "Warning: When serving, please specify the relevant environment variables. When serving on multiple machines, ensure that the necessary parameters, such as hostfile, are set correctly. For details, refer to the following link: https://github.com/FlagOpen/FlagScale/blob/main/flagscale/serve/README.md",
        fg="yellow",
    )
    click.echo(f"Start serving from the yaml {yaml_path}...")
    yaml_path = os.path.abspath(yaml_path)
    config_path = os.path.dirname(yaml_path)
    config_name = os.path.splitext(os.path.basename(yaml_path))[0]
    click.echo(f"config_path: {config_path}")
    click.echo(f"config_name: {config_name}")
    args = ["run.py", f"--config-path={config_path}", f"--config-name={config_name}"]
    if model_path:
        args.append(f"+experiment.runner.cli_args.model_path={model_path}")
    if port:
        args.append(f"+experiment.runner.cli_args.port={port}")
    if engine_args:
        args.append(f"+experiment.runner.cli_args.engine_args='{engine_args}'")
    sys.argv = args
    run_main()


@flagscale.command()
# Input the name of the Docker image (required)
@click.option("--image", "image_name", required=True, type=str, help="The name of the Docker image")
# Input the address of the Git repository (required)
@click.option(
    "--ckpt", "ckpt_name", required=True, type=str, help="The address of the ckpt's git repository"
)
# Input the address of the local directory (optional)
@click.option(
    "--ckpt-path", "ckpt_path", type=click.Path(), required=False, help="The path to save ckpt"
)
def pull(image_name, ckpt_name, ckpt_path):
    "Docker pull image and git clone ckpt."
    # If ckpt_path is not provided, use the default download directory
    if ckpt_path is None:
        ckpt_path = os.path.join(os.getcwd(), "model_download")

    # Check and create the directory
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        print(f"Directory {ckpt_path} created.")

    # Pull the Docker image
    try:
        print(f"Pulling Docker image: {image_name}...")
        subprocess.run(["docker", "pull", image_name], check=True)
        print(f"Successfully pulled Docker image: {image_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to pull Docker image: {image_name}")
        return

    # Clone the Git repository
    try:
        print(f"Cloning Git repository: {ckpt_name} into {ckpt_path}...")
        subprocess.run(["git", "clone", ckpt_name, ckpt_path], check=True)
        print(f"Successfully cloned Git repository: {ckpt_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to clone Git repository: {ckpt_name}")
        return

    # Pull large files using Git LFS
    print("Pulling Git LFS files...")
    try:
        subprocess.run(["git", "lfs", "pull"], cwd=ckpt_path, check=True)
        print("Successfully pulled Git LFS files")
    except subprocess.CalledProcessError:
        print("Failed to pull Git LFS files")
        return


def change_to_flagscale():
    flagscale_path = os.path.dirname(os.path.abspath(__file__))
    flag_scale_path = flagscale_path + "/../"
    os.chdir(flag_scale_path)

    return flag_scale_path


def _install_from_source(backend, device):
    """Install backend from source code using builder functions."""
    import os
    import sys

    # need run in source directory
    flagscale_path = os.path.dirname(os.path.abspath(__file__))
    if 'site-packages' in flagscale_path or 'dist-packages' in flagscale_path:
        root_dir = os.getcwd()
        if not os.path.exists(os.path.join(root_dir, "setup.py")):
            raise click.ClickException(
                "Cannot find FlagScale source directory. Please:\n"
                "  Run this command from FlagScale source directory"
            )
        click.echo(f"[install] Using FlagScale source directory: {root_dir}")
    else:
        root_dir = os.path.abspath(os.path.join(flagscale_path, ".."))

    # Import builder functions
    sys.path.insert(0, root_dir)
    from install.builder import build_and_install_backend, check_device
    from tools.patch.patch import normalize_backend

    # Validate device
    try:
        check_device(device)
    except ValueError as e:
        raise click.ClickException(str(e))

    # Normalize backend name
    backend_normalized = normalize_backend(backend)

    # Build and install the backend (includes unpatch step)
    try:
        build_and_install_backend(backend_normalized, device, root_dir)
        click.echo(f"[install] Successfully installed {backend_normalized} from source.")
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Failed to build {backend_normalized}: {e}")
    except Exception as e:
        raise click.ClickException(f"Error during installation: {e}")


def _install_from_whl(backend, device, version=None):
    from flag_scale.version import get_whl_version

    versions, compatible_versions = get_whl_version(backend, device)
    if compatible_versions:
        install_version = None
        if version:
            for compatible_version in compatible_versions:
                if version in compatible_version:
                    install_version = compatible_version
                    break
        else:
            versions = [Version(version) for version in compatible_versions]
            install_version = max(versions)
            for idx, ver in enumerate(versions):
                if ver == install_version:
                    install_version = compatible_versions[idx]
                    break

        if install_version:
            install_version = f"{backend}" + "-" + install_version + ".whl"
            click.echo(f"Installing {install_version} for {backend} on {device}.")
            install_version = quote(install_version)
            ks3_path = f"https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/whl/{backend}/{device}/{install_version}"
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    ks3_path,
                    "--no-build-isolation",
                    "--verbose",
                ]
            )
            click.echo(f"Successfully installed {backend} from whl package.")
    else:
        raise ValueError(f"No compatible versions found for {backend} on {device}.")


def get_valid_backends_subsets(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    VALID_BACKENDS_SUBSETS = {}

    for backend, subset_config in config.items():
        subsets = list(subset_config["subset"].keys())
        VALID_BACKENDS_SUBSETS[backend] = subsets

    print(VALID_BACKENDS_SUBSETS)
    return VALID_BACKENDS_SUBSETS


def get_valid_types_tasks_cases(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    VALID_TYPES_TASKS_CASES = {}

    for test_type in config:
        VALID_TYPES_TASKS_CASES[test_type] = {}

        for task_name in config[test_type]:
            VALID_TYPES_TASKS_CASES[test_type][task_name] = []
            cases = config[test_type][task_name].strip().split()
            for case in cases:
                VALID_TYPES_TASKS_CASES[test_type][task_name].append(case.lstrip("-").strip())

    print(VALID_TYPES_TASKS_CASES)
    return VALID_TYPES_TASKS_CASES


def unit_test(backend_name, subset_name):
    change_to_flagscale()

    VALID_BACKENDS_SUBSETS = get_valid_backends_subsets("tests/scripts/unit_tests/config.yml")

    if backend_name not in VALID_BACKENDS_SUBSETS:
        click.echo(
            f"Unsupported backend: {backend_name}. Supported backends: {', '.join(VALID_BACKENDS_SUBSETS.keys())}"
        )
        return

    if subset_name not in VALID_BACKENDS_SUBSETS[backend_name]:
        valid_combinations = [
            f"{backend} -> {subset}"
            for backend, subsets in VALID_BACKENDS_SUBSETS.items()
            for subset in subsets
        ]
        click.echo(
            f"Unsupported subset: {subset_name} for backend: {backend_name}. Supported combinations: {', '.join(valid_combinations)}"
        )
        return

    subprocess.run(
        [
            "tests/scripts/unit_tests/test_subset.sh",
            "--backend",
            backend_name,
            "--subset",
            subset_name,
            "--coverage",
            "False",
        ],
        check=True,
    )


def unit_test_all():

    change_to_flagscale()

    VALID_BACKENDS_SUBSETS = get_valid_backends_subsets("tests/scripts/unit_tests/config.yml")

    for backend in VALID_BACKENDS_SUBSETS:
        for subset in VALID_BACKENDS_SUBSETS[backend]:
            subprocess.run(
                [
                    "tests/scripts/unit_tests/test_subset.sh",
                    "--backend",
                    backend,
                    "--subset",
                    subset,
                    "--coverage",
                    "False",
                ],
                check=True,
            )


def functional_test(type_name, task_name):

    flag_scale_path = change_to_flagscale()

    VALID_TYPES_TASKS_CASES = get_valid_types_tasks_cases(
        "tests/scripts/functional_tests/config.yml"
    )

    if type_name not in VALID_TYPES_TASKS_CASES:
        click.echo(
            f"Unsupported type: {type_name}. Supported types: {', '.join(VALID_TYPES_TASKS_CASES.keys())}"
        )
        return

    if task_name not in VALID_TYPES_TASKS_CASES[type_name]:
        valid_combinations = [
            f"{type_name} -> {task_name}"
            for type_name, tasks_name in VALID_TYPES_TASKS_CASES.items()
            for task_name in tasks_name
        ]
        click.echo(
            f"Unsupported task: {task_name} for type: {type_name}. Supported combinations: {', '.join(valid_combinations)}"
        )
        return

    try:
        subprocess.run(
            [
                "tests/scripts/functional_tests/test_task.sh",
                "--type",
                type_name,
                "--task",
                task_name,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: {e} \n")
        print("*" * 200)
        print(
            f"Also, please check the configuration file in path'{flag_scale_path}/tests/functional_tests/test_cases/{type_name}/{task_name}/conf' to ensure that the dependent files already exist."
        )
        print("*" * 200)
        raise


def functional_test_all():

    flag_scale_path = change_to_flagscale()

    VALID_TYPES_TASKS_CASES = get_valid_types_tasks_cases(
        "tests/scripts/functional_tests/config.yml"
    )

    for type_name in VALID_TYPES_TASKS_CASES:
        for task_name in VALID_TYPES_TASKS_CASES[type_name]:
            for case_name in VALID_TYPES_TASKS_CASES[type_name][task_name]:
                try:
                    subprocess.run(
                        [
                            "tests/scripts/functional_tests/test_task.sh",
                            "--type",
                            type_name,
                            "--task",
                            task_name,
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e} \n")
                    print("*" * 200)
                    print(
                        f"Also, please check the configuration file in path'{flag_scale_path}/tests/functional_tests/test_cases/{type_name}/{task_name}/conf' to ensure that the dependent files already exist."
                    )
                    print("*" * 200)
                    raise


@flagscale.command()
@click.option(
    "--unit", is_flag=True, help="Run specific unit test (requires --backend and --subset)"
)
@click.option(
    "--unit-all", is_flag=True, help="Run all unit tests (no additional parameters needed)"
)
@click.option(
    "--functional", is_flag=True, help="Run specific functional test (requires --type and --task)"
)
@click.option(
    "--functional-all",
    is_flag=True,
    help="Run all functional tests (no additional parameters needed)",
)
@click.option("--backend", "backend_name", type=str, help="Backend name for unit testing")
@click.option("--subset", "subset_name", type=str, help="Subset name for unit testing")
@click.option("--type", "type_name", type=str, help="Task classification for functional testing")
@click.option("--task", "task_name", type=str, help="Testing tasks that match the type")
def test(
    unit, unit_all, functional, functional_all, backend_name, subset_name, type_name, task_name
):
    """Execute test command with flexible parameter requirements"""

    print("unit, unit_all, functional, functional_all", unit, unit_all, functional, functional_all)

    # Validate mutual exclusivity
    if unit and unit_all:
        raise click.UsageError("Cannot use both --unit and --unit-all")
    if functional and functional_all:
        raise click.UsageError("Cannot use both --functional and --functional-all")
    if (unit or unit_all) and (functional or functional_all):
        raise click.UsageError(
            "Cannot use both --unit/--unit-all with --functional/--functional-all"
        )

    # Unit test validation
    if unit:
        unit_test(backend_name, subset_name)
    elif unit_all:
        unit_test_all()

    # Functional test validation
    if functional:
        functional_test(type_name, task_name)
    elif functional_all:
        functional_test_all()

    if (not unit) and (not functional) and (not unit_all) and (not functional_all):
        unit_test_all()
        functional_test_all()


@flagscale.command()
@click.argument("backend", type=str)
@click.option("--device", default="gpu", help="Device type (e.g., gpu, cpu)")
def show(backend, device="gpu"):
    """
    Show the whl version of backend.
    """
    from flag_scale.version import get_whl_version

    versions, compatible_versions = get_whl_version(backend, device)
    version_list = '\n'.join(versions)
    click.echo(f"All versions for {backend} on {device}:\n{version_list}")
    if not compatible_versions:
        click.echo(f"Warning: No compatible versions found for {backend} on {device}.")
    else:
        version_list = '\n'.join(compatible_versions)
        click.echo(f"All compatible versions for {backend} on your {device}:\n{version_list}")


@flagscale.command()
@click.option("--backend", type=str, default=None, help="Backend name for installation")
@click.option("--domain", type=str, default=None, help="Domain name for installation")
@click.option("--device", default="gpu", help="Device type (e.g., gpu, cpu)")
@click.option("--version", default=None)
@click.option("--from-source", is_flag=True, help="Force install from source code")
def install(backend=None, domain=None, device="gpu", version=None, from_source=False):
    """
    Install the whl version of backend. If whl is not available, install from source.
    """

    if "metax" in device.lower():
        device = "metax"
    assert backend or domain, "Either backend or domain must be specified"
    backends = []
    if backend:
        backends = backend.split(",")
    if domain:
        if domain == "robotics":
            backends = ["Megatron-LM", "Megatron-Energon"]
        else:
            raise ValueError(f"Unsupported domain: {domain}")

    # Try to install from whl first unless --from-source is specified
    for backend in backends:
        if not from_source:
            try:
                _install_from_whl(backend, device, version)
            except Exception as e:
                click.echo(f"Error installing whl package: {e}", err=True)
                click.echo(f"Falling back to source installation for {backend}...")
                # Install from source
                _install_from_source(backend, device)
        else:
            _install_from_source(backend, device)


if __name__ == "__main__":
    flagscale()

import os
import subprocess
import sys

import click

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
        yaml_path = os.path.join(default_dir, "examples", model_name, "conf", "train.yaml")
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
    help='Model config as JSON string, e.g. \'{"a":1, "b":2}\'',
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
        yaml_path = os.path.join(default_dir, "examples", model_name, "conf", "serve.yaml")
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


if __name__ == "__main__":
    flagscale()

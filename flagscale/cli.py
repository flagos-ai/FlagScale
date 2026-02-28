import subprocess
import sys
from enum import Enum
from pathlib import Path

import typer

from flagscale import __version__ as FLAGSCALE_VERSION

app = typer.Typer(
    name="flagscale",
    help="FlagScale CLI - comprehensive toolkit for large model lifecycle.",
    add_completion=False,
)

# ============================================================================
# Helper Functions
# ============================================================================


def resolve_config(model_name: str, yaml_path: Path | None, task: str) -> tuple[str, str]:
    """Resolve config path and name from model_name or yaml_path.

    When no explicit ``yaml_path`` is given, the config is looked up at
    ``<cwd>/examples/<model>/conf/<task>.yaml``.  The user is expected to
    run CLI commands from the repo root — ``examples/`` is not part of the
    installed package.
    """
    if yaml_path:
        yaml_path = yaml_path.resolve()
        if not yaml_path.exists():
            typer.echo(f"Error: {yaml_path} does not exist", err=True)
            raise typer.Exit(1)
        return str(yaml_path.parent), yaml_path.stem

    yaml_path = Path.cwd() / "examples" / model_name / "conf" / f"{task}.yaml"
    if not yaml_path.exists():
        typer.echo(f"Error: {yaml_path} does not exist", err=True)
        raise typer.Exit(1)
    return str(yaml_path.parent), yaml_path.stem


def _handle_config_error(error: Exception, config_path: str, config_name: str) -> None:
    """Format config errors with detailed context and exit."""
    config_file = f"{config_path}/{config_name}.yaml"
    typer.secho(f"Config error in: {config_file}", fg="red", err=True)
    typer.secho(f"Error: {error}", fg="red", err=True)
    typer.secho(
        "Hint: Check that the config file exists, all defaults are resolvable, "
        "and required fields are present.",
        fg="yellow",
        err=True,
    )
    raise typer.Exit(1)


def run_task(config_path: str, config_name: str, action: str, extra_args: list | None = None):
    """Execute task via flagscale.run.

    The task type is determined from the config at runtime by Hydra,
    not passed explicitly here.
    """
    from flagscale.run import main as run_main

    args = [
        "run.py",
        f"--config-path={config_path}",
        f"--config-name={config_name}",
        f"action={action}",
    ]
    if extra_args:
        args.extend(extra_args)
    sys.argv = args
    try:
        run_main()
    except SystemExit as e:
        if e.code and e.code != 0:
            raise
    except Exception as e:
        _handle_config_error(e, config_path, config_name)


def get_action(stop: bool, dryrun: bool, test: bool, query: bool, tune: bool) -> str:
    """Determine action from flags (mutually exclusive)"""
    flags = [
        ("stop", stop),
        ("dryrun", dryrun),
        ("test", test),
        ("query", query),
        ("auto_tune", tune),
    ]
    set_flags = [name for name, value in flags if value]

    if len(set_flags) > 1:
        typer.echo(f"Error: Flags are mutually exclusive: --{', --'.join(set_flags)}", err=True)
        raise typer.Exit(1)

    if set_flags:
        return set_flags[0]
    return "run"  # default


# ============================================================================
# Run Command: flagscale run --config-path <path> --config-name <name> [options]
# This replaces the old `python run.py` interface
# ============================================================================


class Action(str, Enum):
    run = "run"
    dryrun = "dryrun"
    test = "test"
    stop = "stop"
    query = "query"
    auto_tune = "auto_tune"


@app.command("run")
def run_cmd(
    config_path: Path = typer.Option(..., "--config-path", "-p", help="Path to config directory"),
    config_name: str = typer.Option(
        ..., "--config-name", "-n", help="Config file name (without .yaml)"
    ),
    action: Action = typer.Option(Action.run, "--action", "-a", help="Action to perform"),
    overrides: list[str] | None = typer.Argument(
        None, help="Additional Hydra overrides (e.g., key=value)"
    ),
):
    """Run task with explicit config path and name (replaces python run.py)

    Example:
        flagscale run --config-path ./examples/qwen3/conf --config-name train --action run
        flagscale run -p ./examples/qwen3/conf -n train -a stop
    """
    config_path = config_path.resolve()
    if not config_path.exists():
        typer.echo(f"Error: Config path does not exist: {config_path}", err=True)
        raise typer.Exit(1)

    config_file = config_path / f"{config_name}.yaml"
    if not config_file.exists():
        typer.echo(f"Error: Config file does not exist: {config_file}", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"Running: flagscale --config-path={config_path} "
        f"--config-name={config_name} action={action.value}"
    )
    run_task(
        str(config_path),
        config_name,
        action.value,
        extra_args=list(overrides) if overrides else None,
    )


# ============================================================================
# Task Commands: flagscale <task> [--flags] <model>
# ============================================================================


@app.command()
def train(
    model: str = typer.Argument(..., help="Model name (e.g., aquila, llama)"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config YAML path"),
    stop: bool = typer.Option(False, "--stop", help="Stop training"),
    dryrun: bool = typer.Option(False, "--dryrun", help="Validate config only"),
    test: bool = typer.Option(False, "--test", help="Run with test"),
    query: bool = typer.Option(False, "--query", help="Query status"),
    tune: bool = typer.Option(False, "--tune", help="Auto-tune"),
):
    """Train a model"""
    action = get_action(stop, dryrun, test, query, tune)
    cfg_path, cfg_name = resolve_config(model, config, "train")
    typer.echo(f"Train {model} [{action}]")
    typer.echo(f"config_path: {cfg_path}")
    typer.echo(f"config_name: {cfg_name}")
    run_task(cfg_path, cfg_name, action)


@app.command()
def serve(
    model: str = typer.Argument(..., help="Model name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config YAML path"),
    stop: bool = typer.Option(False, "--stop", help="Stop serving"),
    test: bool = typer.Option(False, "--test", help="Test serving"),
    tune: bool = typer.Option(False, "--tune", help="Auto-tune"),
    port: int | None = typer.Option(None, "--port", help="Server port"),
    model_path: str | None = typer.Option(None, "--model-path", help="Model weights path"),
    engine_args: str | None = typer.Option(
        None, "--engine-args", help="Engine args as JSON string, e.g. '{\"a\":1}'"
    ),
):
    """Serve a model"""
    action = "stop" if stop else ("test" if test else ("auto_tune" if tune else "run"))
    cfg_path, cfg_name = resolve_config(model, config, "serve")
    extra = []
    if port:
        extra.append(f"+experiment.runner.cli_args.port={port}")
    if model_path:
        extra.append(f"+experiment.runner.cli_args.model_path={model_path}")
    if engine_args:
        extra.append(f"+experiment.runner.cli_args.engine_args='{engine_args}'")
    typer.echo(f"Serve {model} [{action}]")
    typer.echo(f"config_path: {cfg_path}")
    typer.echo(f"config_name: {cfg_name}")
    if action == "run":
        typer.secho(
            "Warning: When serving, please specify the relevant environment variables. "
            "When serving on multiple machines, ensure that the necessary parameters, "
            "such as hostfile, are set correctly. For details, refer to: "
            "https://github.com/FlagOpen/FlagScale/blob/main/flagscale/serve/README.md",
            fg="yellow",
        )
    run_task(cfg_path, cfg_name, action, extra)


@app.command()
def inference(
    model: str = typer.Argument(..., help="Model name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config YAML path"),
    stop: bool = typer.Option(False, "--stop", help="Stop inference"),
    dryrun: bool = typer.Option(False, "--dryrun", help="Validate config only"),
    test: bool = typer.Option(False, "--test", help="Run with test"),
):
    """Run inference"""
    action = get_action(stop, dryrun, test, False, False)
    cfg_path, cfg_name = resolve_config(model, config, "inference")
    typer.echo(f"Inference {model} [{action}]")
    run_task(cfg_path, cfg_name, action)


@app.command()
def rl(
    model: str = typer.Argument(..., help="Model name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config YAML path"),
    stop: bool = typer.Option(False, "--stop", help="Stop RL training"),
    dryrun: bool = typer.Option(False, "--dryrun", help="Validate config only"),
    test: bool = typer.Option(False, "--test", help="Run with test"),
):
    """Run RL training"""
    action = get_action(stop, dryrun, test, False, False)
    cfg_path, cfg_name = resolve_config(model, config, "rl")
    typer.echo(f"RL {model} [{action}]")
    run_task(cfg_path, cfg_name, action)


@app.command()
def compress(
    model: str = typer.Argument(..., help="Model name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config YAML path"),
    stop: bool = typer.Option(False, "--stop", help="Stop compression"),
    dryrun: bool = typer.Option(False, "--dryrun", help="Validate config only"),
):
    """Compress a model"""
    action = "stop" if stop else ("dryrun" if dryrun else "run")
    cfg_path, cfg_name = resolve_config(model, config, "compress")
    typer.echo(f"Compress {model} [{action}]")
    run_task(cfg_path, cfg_name, action)


# ============================================================================
# Install Command (delegates to tools/install)
# ============================================================================


class Platform(str, Enum):
    cuda = "cuda"
    default = "default"


class PkgManager(str, Enum):
    uv = "uv"
    pip = "pip"
    conda = "conda"


@app.command()
def install(
    platform: Platform = typer.Option(Platform.cuda, "--platform", "-p", help="Platform"),
    task: str = typer.Option(
        "train", "--task", "-t", help="Task (train, inference, serve, rl, all)"
    ),
    pkg_mgr: PkgManager = typer.Option(PkgManager.uv, "--pkg-mgr", help="Package manager"),
    no_system: bool = typer.Option(False, "--no-system", help="Skip system packages"),
    no_dev: bool = typer.Option(False, "--no-dev", help="Skip dev phase"),
    no_base: bool = typer.Option(False, "--no-base", help="Skip base phase"),
    no_task: bool = typer.Option(False, "--no-task", help="Skip task phase"),
    only_pip: bool = typer.Option(
        False, "--only-pip", help="Only install pip packages (skip apt and source builds)"
    ),
    debug: bool = typer.Option(False, "--debug", help="Dry-run mode"),
):
    """Install dependencies via tools/install/install.sh"""
    install_script = Path.cwd() / "tools" / "install" / "install.sh"

    if not install_script.exists():
        typer.echo(f"Error: Install script not found: {install_script}", err=True)
        raise typer.Exit(1)

    cmd = [
        str(install_script),
        "--platform",
        platform.value,
        "--task",
        task,
        "--pkg-mgr",
        pkg_mgr.value,
    ]
    if no_system:
        cmd.append("--no-system")
    if only_pip:
        cmd.append("--only-pip")
    if no_dev:
        cmd.append("--no-dev")
    if no_base:
        cmd.append("--no-base")
    if no_task:
        cmd.append("--no-task")
    if debug:
        cmd.append("--debug")

    typer.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


# ============================================================================
# Test Command
# ============================================================================


@app.command("test")
def run_tests(
    platform: Platform = typer.Option(Platform.cuda, "--platform", help="Platform"),
    device: str = typer.Option("gpu", "--device", help="Device (gpu, cpu)"),
    test_type: str = typer.Option("unit", "--type", help="Test type (unit, functional)"),
    task: str | None = typer.Option(None, "--task", help="Task to test"),
    model: str | None = typer.Option(None, "--model", help="Model to test"),
    test_list: str | None = typer.Option(None, "--list", help="Specific tests to run"),
):
    """Run tests"""
    test_script = Path.cwd() / "tests" / "test_utils" / "runners" / "run_tests.sh"

    if not test_script.exists():
        typer.echo(f"Error: Test script not found: {test_script}", err=True)
        raise typer.Exit(1)

    cmd = [
        str(test_script),
        "--platform",
        platform.value,
        "--device",
        device,
        "--type",
        test_type,
    ]
    if task:
        cmd.extend(["--task", task])
    if model:
        cmd.extend(["--model", model])
    if test_list:
        cmd.extend(["--list", test_list])

    typer.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


# ============================================================================
# Pull Command
# ============================================================================


@app.command()
def pull(
    image: str = typer.Option(..., "--image", help="Docker image name"),
    ckpt: str = typer.Option(..., "--ckpt", help="Checkpoint git repository"),
    ckpt_path: Path | None = typer.Option(None, "--ckpt-path", help="Path to save checkpoint"),
):
    """Pull Docker image and clone checkpoint repository."""
    # If ckpt_path is not provided, use the default download directory
    if ckpt_path is None:
        ckpt_path = Path.cwd() / "model_download"

    # Check and create the directory
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True)
        typer.echo(f"Directory {ckpt_path} created.")

    # Pull the Docker image
    try:
        typer.echo(f"Pulling Docker image: {image}...")
        subprocess.run(["docker", "pull", image], check=True)
        typer.echo(f"Successfully pulled Docker image: {image}")
    except subprocess.CalledProcessError:
        typer.echo(f"Failed to pull Docker image: {image}", err=True)
        raise typer.Exit(1)

    # Clone the Git repository
    try:
        typer.echo(f"Cloning Git repository: {ckpt} into {ckpt_path}...")
        subprocess.run(["git", "clone", ckpt, str(ckpt_path)], check=True)
        typer.echo(f"Successfully cloned Git repository: {ckpt}")
    except subprocess.CalledProcessError:
        typer.echo(f"Failed to clone Git repository: {ckpt}", err=True)
        raise typer.Exit(1)

    # Pull large files using Git LFS
    typer.echo("Pulling Git LFS files...")
    try:
        subprocess.run(["git", "lfs", "pull"], cwd=str(ckpt_path), check=True)
        typer.echo("Successfully pulled Git LFS files")
    except subprocess.CalledProcessError:
        typer.echo("Failed to pull Git LFS files", err=True)
        raise typer.Exit(1)


# ============================================================================
# Version
# ============================================================================


def version_callback(value: bool):
    if value:
        typer.echo(f"flagscale version {FLAGSCALE_VERSION}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version"
    ),
):
    """FlagScale CLI - comprehensive toolkit for large model lifecycle."""
    pass


# Entry point
def flagscale():
    app()


if __name__ == "__main__":
    flagscale()

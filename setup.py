import os
import subprocess

from setuptools import setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version() -> str:
    """Read version from pyproject.toml (single source of truth, Python 3.11+)."""
    try:
        import tomllib

        pyproject_path = os.path.join(SCRIPT_DIR, "pyproject.toml")
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version", "0.0.0")
    except Exception:
        return "0.0.0"


FLAGSCALE_VERSION = _get_version()
INSTALL_SCRIPT = os.path.join(SCRIPT_DIR, "tools", "install", "install.sh")


def is_pip_isolated_build():
    """Check if we're running in pip's isolated build environment.

    Prefer detection via pip/PEP 517 specific environment variables rather than
    fragile filesystem heuristics that can misclassify user code directories.
    """
    # Common environment markers set by pip during isolated/PEP 517 builds.
    pip_env_markers = (
        "PEP517_BUILD_BACKEND",
        "PIP_BUILD_TRACKER",
        "PIP_REQ_TRACKER",
        "PIP_ISOLATED_ENV",
    )
    for var in pip_env_markers:
        if os.environ.get(var):
            return True
    return False


def run_install_script():
    """Invoke tools/install/install.sh for dependency installation."""
    if not os.path.exists(INSTALL_SCRIPT):
        print(f"[flagscale] Warning: Install script not found: {INSTALL_SCRIPT}")
        return

    platform = os.environ.get("FLAGSCALE_PLATFORM", "cuda")
    task = os.environ.get("FLAGSCALE_TASK", "all")

    cmd = [
        INSTALL_SCRIPT,
        "--platform",
        platform,
        "--task",
        task,
        "--pkg-mgr",
        "pip",
        "--no-system",
        "--only-pip",
        "--src-deps",
        "megatron-lm",
    ]

    print(f"[flagscale] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Installation failed with exit code {exc.returncode}") from exc


# Run install.sh when using: pip install --no-build-isolation .
# Control via env vars: FLAGSCALE_PLATFORM (default: cuda), FLAGSCALE_TASK (default: all)
if not is_pip_isolated_build():
    run_install_script()

# NOTE: Installation methods:
# 1. pip install .                      -> Installs flagscale CLI only (isolated build)
# 2. pip install --no-build-isolation . -> Installs CLI + pip deps + megatron-lm (no apt)
#    Control with: FLAGSCALE_PLATFORM=cuda FLAGSCALE_TASK=train pip install --no-build-isolation .
# 3. flagscale install                  -> Full installation (apt + pip + all source deps)

setup(
    name="flagscale",
    version=FLAGSCALE_VERSION,
    description="FlagScale is a comprehensive toolkit designed to support the entire lifecycle of large models, developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI).",
    url="https://github.com/FlagOpen/FlagScale",
    packages=["flagscale"],
    package_dir={"flagscale": "flagscale"},
    install_requires=["typer>=0.9.0"],
    entry_points={"console_scripts": ["flagscale=flagscale.cli:flagscale"]},
)

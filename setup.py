import os
import platform
import re
import subprocess
import sys

from setuptools import setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared requirements parser (also used by tools/install shell scripts via CLI).
sys.path.insert(0, os.path.join(SCRIPT_DIR, "tools", "install", "utils"))
from parse_requirements import parse_requirements


def build_extras():
    """Build extras_require by scanning requirements/ directory.

    Auto-discovers platforms (cuda, rocm, ...) and tasks (train, serve, ...).
    Maps: requirements/<platform>/<task>.txt -> extra "<platform>-<task>"
    Special: base.txt -> extra "<platform>", dev.txt -> extra "dev"

    Returns (extras, pip_options, pkg_options) where:
      - extras: dict mapping extra name -> list of PEP 508 specifiers
        (excludes packages with per-package options — those are
        auto-installed after setup() via _auto_install_annotated_packages())
      - pip_options: dict mapping extra name -> list of pip option strings
      - pkg_options: dict mapping extra name -> dict of package -> list of options
        (these packages are excluded from extras and auto-installed after setup())
    """
    extras = {}
    extra_pip_options = {}
    extra_pkg_options = {}

    def _register(name, req_file):
        deps, opts, pkg_opts = parse_requirements(req_file)
        # Exclude annotated packages from extras_require — they need
        # special pip flags and are auto-installed after setup().
        normal_deps = [d for d in deps if d not in pkg_opts]
        if normal_deps or pkg_opts:
            extras[name] = normal_deps
        if opts:
            extra_pip_options[name] = list(dict.fromkeys(opts))
        if pkg_opts:
            extra_pkg_options[name] = pkg_opts

    req_dir = os.path.join(SCRIPT_DIR, "requirements")
    # Platform directories (cuda, rocm, ...)
    for entry in sorted(os.listdir(req_dir)):
        entry_path = os.path.join(req_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        for filename in sorted(os.listdir(entry_path)):
            if not filename.endswith(".txt"):
                continue
            task = filename[:-4]  # strip .txt
            extra_name = entry if task == "base" else f"{entry}-{task}"
            _register(extra_name, os.path.join(entry_path, filename))
    # Dev extras (platform-independent)
    dev_path = os.path.join(req_dir, "dev.txt")
    if os.path.isfile(dev_path):
        _register("dev", dev_path)

    return extras, extra_pip_options, extra_pkg_options


EXTRAS, PIP_OPTIONS, PKG_OPTIONS = build_extras()


# ---------------------------------------------------------------------------
# Auto-install helpers (run after setup() when invoked by pip)
# ---------------------------------------------------------------------------

_SYSTEM = platform.system()  # "Linux", "Darwin", or "Windows"

_BUILD_ISOLATION_VARS = (
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PEP517_BUILD_BACKEND",
    "PIP_BUILD_TRACKER",
    "PIP_REQ_TRACKER",
)


def _get_pip_verbosity():
    """Detect pip's verbosity level from environment.

    pip maps ``-v`` / ``--verbose`` flags to the ``PIP_VERBOSE``
    environment variable (standard pip config-via-env convention).
    Returns 0 when quiet, 1+ for increasing verbosity.
    """
    try:
        return int(os.environ.get("PIP_VERBOSE", "0"))
    except ValueError:
        return 0


def _get_clean_env():
    """Return a copy of os.environ with pip's build-isolation variables removed.

    pip's isolated build sets PYTHONPATH and PYTHONNOUSERSITE to sandbox the
    build, which prevents subprocesses from finding packages (including pip
    itself) in the user's real environment.  Removing these lets the
    subprocess use the original conda/venv site-packages.
    """
    env = os.environ.copy()
    for var in _BUILD_ISOLATION_VARS:
        env.pop(var, None)
    return env


def _get_cmdline(pid):
    """Get the command line string of a process by PID (cross-platform).

    Returns the command line as a string, or ``None`` on failure.
    """
    try:
        if _SYSTEM == "Linux":
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                return f.read().decode("utf-8", errors="replace")
        elif _SYSTEM == "Darwin":
            output = subprocess.check_output(
                ["ps", "-o", "args=", "-p", str(pid)],
                stderr=subprocess.DEVNULL,
            )
            return output.decode("utf-8", errors="replace").strip()
        elif _SYSTEM == "Windows":
            output = subprocess.check_output(
                ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine", "/value"],
                stderr=subprocess.DEVNULL,
            )
            for line in output.decode("utf-8", errors="replace").splitlines():
                if line.startswith("CommandLine="):
                    return line[len("CommandLine=") :]
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


def _get_ppid(pid):
    """Get the parent PID of a process (cross-platform).

    Returns the parent PID as an ``int``, or ``None`` on failure.
    """
    try:
        if _SYSTEM == "Linux":
            with open(f"/proc/{pid}/stat") as f:
                stat_content = f.read()
            # Format: pid (comm) state ppid ... — split after last ')' for spaces in comm
            return int(stat_content.split(")")[1].split()[1])
        elif _SYSTEM == "Darwin":
            output = subprocess.check_output(
                ["ps", "-o", "ppid=", "-p", str(pid)],
                stderr=subprocess.DEVNULL,
            )
            return int(output.strip())
        elif _SYSTEM == "Windows":
            output = subprocess.check_output(
                [
                    "wmic",
                    "process",
                    "where",
                    f"ProcessId={pid}",
                    "get",
                    "ParentProcessId",
                    "/value",
                ],
                stderr=subprocess.DEVNULL,
            )
            for line in output.decode("utf-8", errors="replace").splitlines():
                if line.startswith("ParentProcessId="):
                    return int(line[len("ParentProcessId=") :].strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return None


def _get_requested_extras():
    """Auto-detect which extras were requested by inspecting the process tree.

    When the user runs ``pip install ".[cuda-train]"``, pip spawns a
    subprocess to build the wheel.  This function walks up the process tree
    looking for a pip install argument matching ``.[<extras>]`` and returns
    the parsed list of extra names.

    Works on Linux (``/proc``), macOS (``ps``), and Windows (``wmic``).
    Returns ``None`` if no extras specifier is found (e.g. ``pip install .``).
    """
    # Match .[extras] at word boundary — the dot may be preceded by a path
    # separator, NUL (Linux /proc cmdline delimiter), or space (macOS/Windows).
    extras_re = re.compile(r"(?:^|/|\\|\x00|\s)\.\[([^\]]+)\]")
    pid = os.getpid()
    for _ in range(10):  # walk up at most 10 levels
        cmdline = _get_cmdline(pid)
        if cmdline is None:
            break
        m = extras_re.search(cmdline)
        if m:
            return [e.strip() for e in m.group(1).split(",") if e.strip()]
        ppid = _get_ppid(pid)
        if ppid is None or ppid <= 1 or ppid == pid:
            break
        pid = ppid
    return None


def _auto_install_annotated_packages():
    """Auto-install packages that need special pip flags (e.g. --no-build-isolation).

    Detects which extras were requested from the parent pip process (e.g.
    ``pip install ".[cuda-train]"``), then installs only annotated packages
    belonging to those extras.

    These packages are excluded from extras_require because pip can't pass
    per-package flags.  Assumes build dependencies (e.g. torch) are already
    installed in the environment.
    """
    requested = _get_requested_extras()
    if not requested:
        return

    # Filter to extras that actually have annotated packages.
    requested = [e for e in requested if e in PKG_OPTIONS]
    if not requested:
        return

    verbose = _get_pip_verbosity()
    clean_env = _get_clean_env()

    if verbose:
        print("[flagscale] Auto-installing annotated packages...", file=sys.stderr)
        print(f"[flagscale]   requested extras: {', '.join(requested)}", file=sys.stderr)
        print(f"[flagscale]   verbosity level: {verbose}", file=sys.stderr)
        print(
            f"[flagscale]   cleaned env vars: {', '.join(_BUILD_ISOLATION_VARS)}",
            file=sys.stderr,
        )

    seen = set()
    for extra_name in sorted(requested):
        pkg_opts = PKG_OPTIONS.get(extra_name, {})
        pip_opt_list = PIP_OPTIONS.get(extra_name, [])
        for pkg, opts in pkg_opts.items():
            if pkg in seen:
                continue
            seen.add(pkg)
            pkg_name = pkg.split("@")[0].strip()
            opt_str = " ".join(opts)
            cmd = [sys.executable, "-m", "pip", "install"]
            cmd.extend(opts)
            if verbose:
                cmd.append("-" + "v" * verbose)
            for pip_opt in pip_opt_list:
                cmd.extend(pip_opt.split())
            cmd.append(pkg)

            if verbose:
                print(f"[flagscale]   command: {' '.join(cmd)}", file=sys.stderr)
            else:
                print(
                    f"[flagscale] Installing {pkg_name} with {opt_str}...",
                    file=sys.stderr,
                )

            rc = subprocess.call(cmd, env=clean_env)
            if rc != 0:
                full_opts = f"{opt_str} {' '.join(pip_opt_list)}".strip()
                print(
                    f"[flagscale] Warning: auto-install of {pkg_name} failed (exit {rc}).",
                    file=sys.stderr,
                )
                print(
                    f'[flagscale] Install manually: pip install {full_opts} "{pkg}"',
                    file=sys.stderr,
                )


# ---------------------------------------------------------------------------
# FlagCX build (native communication library + torch plugin)
# ---------------------------------------------------------------------------

_ADAPTOR_TO_MAKE_FLAG = {
    "nvidia": "USE_NVIDIA",
    "iluvatar_corex": "USE_ILUVATAR_COREX",
    "cambricon": "USE_CAMBRICON",
    "metax": "USE_METAX",
    "du": "USE_DU",
    "klx": "USE_KUNLUNXIN",
    "ascend": "USE_ASCEND",
    "musa": "USE_MUSA",
    "amd": "USE_AMD",
    "tsm": "USE_TSM",
    "enflame": "USE_ENFLAME",
}


def _build_flagcx():
    """Build FlagCX native library and install its torch plugin.

    Controlled by environment variables:
      - ``FLAGSCALE_BUILD_FLAGCX=1`` — enable the build (default: skip)
      - ``FLAGCX_ADAPTOR`` — hardware adaptor name (default: ``nvidia``)

    The function auto-initializes the git submodule if the source tree is
    missing, runs ``make`` with the appropriate ``USE_*`` flag, and then
    pip-installs the torch plugin in editable mode.
    """
    if os.environ.get("FLAGSCALE_BUILD_FLAGCX") != "1":
        return

    adaptor = os.environ.get("FLAGCX_ADAPTOR", "nvidia")
    if adaptor not in _ADAPTOR_TO_MAKE_FLAG:
        raise ValueError(
            f"Unknown FLAGCX_ADAPTOR={adaptor!r}. "
            f"Valid values: {', '.join(sorted(_ADAPTOR_TO_MAKE_FLAG))}"
        )

    flagcx_dir = os.path.join(SCRIPT_DIR, "third_party", "FlagCX")

    # Auto-initialize the submodule (and its nested submodules like
    # nlohmann/json, googletest) if the source directory is empty/missing.
    # Always run with --recursive to ensure nested submodules are present
    # even when the top-level FlagCX directory was cloned without them.
    if not os.path.isdir(flagcx_dir) or not os.listdir(flagcx_dir):
        print("[flagscale] Initializing FlagCX submodule...", file=sys.stderr)
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive", "third_party/FlagCX"],
            cwd=SCRIPT_DIR,
        )
    else:
        print("[flagscale] Initializing FlagCX nested submodules...", file=sys.stderr)
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=flagcx_dir,
        )

    make_flag = _ADAPTOR_TO_MAKE_FLAG[adaptor]
    nproc = os.cpu_count() or 1

    # Build the native library.
    print(f"[flagscale] Building FlagCX ({adaptor})...", file=sys.stderr)
    subprocess.check_call(
        ["make", f"{make_flag}=1", f"-j{nproc}"],
        cwd=flagcx_dir,
        env=_get_clean_env(),
    )

    # Install the torch plugin.
    print("[flagscale] Installing FlagCX torch plugin...", file=sys.stderr)
    plugin_dir = os.path.join(flagcx_dir, "plugin", "torch")
    clean_env = _get_clean_env()
    clean_env["FLAGCX_ADAPTOR"] = adaptor
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", plugin_dir],
        env=clean_env,
    )


# ---------------------------------------------------------------------------
# NOTE: Installation methods:
# 1. pip install .                    -> CLI only (typer)
# 2. pip install ".[cuda-train]"      -> CLI + pip deps + auto-install annotated packages
#    Annotated packages (e.g. megatron-core with --no-build-isolation) are excluded from
#    extras_require and auto-installed by detecting ".[cuda-train]" from the parent pip
#    process tree (cross-platform: /proc on Linux, ps on macOS, wmic on Windows).
#    Requires torch to be pre-installed. Use -v/-vvv for detail.
# 3. pip install ".[cuda-all,dev]"    -> CLI + all CUDA pip deps + dev tools
# 4. pip install -r requirements/cuda/train.txt  -> pip deps with index URLs (handled natively)
#    Packages annotated with "# [--option ...]" need separate install with those options.
#    The shell installer (tools/install) handles this via parse_pkg_annotations().
# 5. flagscale install                -> Full installation (apt + pip + ALL source deps)
#
# FlagCX (optional native communication library):
#   Set FLAGSCALE_BUILD_FLAGCX=1 to auto-build the FlagCX submodule (third_party/FlagCX)
#   and install its torch plugin.  Use FLAGCX_ADAPTOR to select the hardware backend
#   (default: nvidia).  Example:
#     FLAGSCALE_BUILD_FLAGCX=1 FLAGCX_ADAPTOR=nvidia pip install .
# ---------------------------------------------------------------------------

# Only extras_require is dynamic — everything else comes from pyproject.toml.
setup(extras_require=EXTRAS)

# Only auto-install when setup.py is executed directly (pip install, python setup.py ...),
# not when imported by tests or other modules.
if __name__ == "__main__":
    _build_flagcx()
    if PKG_OPTIONS:
        _auto_install_annotated_packages()

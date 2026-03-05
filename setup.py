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

    Produces flat extras: platform names (cuda, rocm, ...) and task names
    (train, serve, ...) are separate extras with empty deps.  The actual
    dependencies are installed via subprocess at install time because the
    correct requirements file depends on the combination of platform + task,
    which pip's static extras_require can't express.

    Returns (extras, platforms, tasks) where:
      - extras: dict mapping extra name -> list of PEP 508 specifiers
        (empty for platform/task markers; populated for ``dev``)
      - platforms: set of discovered platform names
      - tasks: set of discovered task names (excluding ``base``)
    """
    extras = {}
    platforms = set()
    tasks = set()

    req_dir = os.path.join(SCRIPT_DIR, "requirements")
    # Platform directories (cuda, rocm, ...)
    for entry in sorted(os.listdir(req_dir)):
        entry_path = os.path.join(req_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        platforms.add(entry)
        extras[entry] = []  # platform marker — empty deps
        for filename in sorted(os.listdir(entry_path)):
            if not filename.endswith(".txt"):
                continue
            task = filename[:-4]  # strip .txt
            if task not in ("base", "all"):
                tasks.add(task)

    # Task markers — empty deps (deps installed via subprocess)
    for task in sorted(tasks):
        extras[task] = []

    # Special markers
    extras["all"] = []  # installs all tasks for the platform
    extras["flagcx"] = []  # triggers native FlagCX build

    # Dev extras (platform-independent) — has actual deps
    dev_path = os.path.join(req_dir, "dev.txt")
    if os.path.isfile(dev_path):
        deps, _, _ = parse_requirements(dev_path)
        extras["dev"] = deps

    return extras, platforms, tasks


EXTRAS, PLATFORMS, TASKS = build_extras()


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


def _normalize_extra(name):
    """PEP 685: lowercase, replace [-_.] runs with single hyphen."""
    return re.sub(r"[-_.]+", "-", name).lower()


_PLATFORM_TO_ADAPTOR = {
    "cuda": "nvidia",
    # Future: "rocm": "amd", "ascend": "ascend", etc.
}


def _get_flagcx_adaptor(requested_extras, req_platforms):
    """Determine FlagCX adaptor from requested extras and platform.

    Returns adaptor name (e.g. ``"nvidia"``) or ``None`` if ``flagcx``
    was not requested.  Raises ``ValueError`` if ``flagcx`` is requested
    without a platform that maps to an adaptor.
    """
    normalized = {_normalize_extra(e) for e in requested_extras}
    if "flagcx" not in normalized:
        return None

    if not req_platforms:
        raise ValueError(
            "The 'flagcx' extra requires a platform extra (e.g. 'cuda') "
            "to determine the hardware backend."
        )

    plat = next(iter(req_platforms))
    adaptor = _PLATFORM_TO_ADAPTOR.get(plat)
    if adaptor is None:
        raise ValueError(
            f"No FlagCX adaptor mapping for platform '{plat}'. "
            f"Known mappings: {_PLATFORM_TO_ADAPTOR}"
        )
    return adaptor


def _install_platform_task_deps():
    """Install platform+task dependencies via subprocess.

    Detects requested extras from the parent pip process, validates the
    combination, installs dependencies from the matching requirements
    files, and triggers FlagCX build if requested.
    """
    requested = _get_requested_extras()
    if not requested:
        return

    normalized = {_normalize_extra(e) for e in requested}

    # Separate into categories.
    req_platforms = normalized & PLATFORMS
    req_tasks = normalized & TASKS
    has_all = "all" in normalized
    has_flagcx = "flagcx" in normalized

    # Validation: tasks require exactly one platform.
    if (req_tasks or has_all) and len(req_platforms) == 0:
        raise ValueError(
            f"Task extras {sorted(req_tasks or {'all'})} require a platform extra "
            f"(one of: {sorted(PLATFORMS)}). "
            f'Example: pip install ".[cuda,train]"'
        )
    if len(req_platforms) > 1:
        raise ValueError(
            f"Multiple platform extras requested: {sorted(req_platforms)}. "
            "Only one platform can be specified at a time."
        )

    if not req_platforms:
        # Only non-platform extras (dev, flagcx without tasks) — nothing to install.
        if has_flagcx:
            raise ValueError(
                "The 'flagcx' extra requires a platform extra (e.g. 'cuda') "
                "to determine the hardware backend."
            )
        return

    plat = next(iter(req_platforms))
    req_dir = os.path.join(SCRIPT_DIR, "requirements")

    # Determine which task files to install.
    if has_all:
        # Use all.txt which contains -r includes for the intended tasks.
        # parse_requirements() resolves -r includes recursively, so the
        # exact set of tasks is governed by all.txt, not by directory listing.
        all_file = os.path.join(req_dir, plat, "all.txt")
        task_files = [all_file] if os.path.isfile(all_file) else []
    elif req_tasks:
        task_files = []
        # Always include base.txt for the platform.
        base_file = os.path.join(req_dir, plat, "base.txt")
        if os.path.isfile(base_file):
            task_files.append(base_file)
        for task in sorted(req_tasks):
            task_file = os.path.join(req_dir, plat, f"{task}.txt")
            if os.path.isfile(task_file):
                task_files.append(task_file)
            else:
                print(
                    f"[flagscale] Warning: no requirements file for {plat}/{task}.txt",
                    file=sys.stderr,
                )
    else:
        # Platform only, no tasks — install base.txt.
        base_file = os.path.join(req_dir, plat, "base.txt")
        task_files = [base_file] if os.path.isfile(base_file) else []

    verbose = _get_pip_verbosity()
    clean_env = _get_clean_env()

    if verbose:
        print("[flagscale] Installing platform+task dependencies...", file=sys.stderr)
        print(f"[flagscale]   platform: {plat}", file=sys.stderr)
        print(
            f"[flagscale]   tasks: {sorted(req_tasks) if req_tasks else 'all' if has_all else 'base'}",
            file=sys.stderr,
        )
        print(f"[flagscale]   verbosity level: {verbose}", file=sys.stderr)

    seen_deps = set()
    for req_file in task_files:
        deps, pip_opts, pkg_opts = parse_requirements(req_file)

        # Install normal deps (non-annotated).
        normal_deps = [d for d in deps if d not in pkg_opts and d not in seen_deps]
        if normal_deps:
            cmd = [sys.executable, "-m", "pip", "install"]
            if verbose:
                cmd.append("-" + "v" * verbose)
            for pip_opt in pip_opts:
                cmd.extend(pip_opt.split())
            cmd.extend(normal_deps)

            if verbose:
                print(f"[flagscale]   command: {' '.join(cmd)}", file=sys.stderr)
            else:
                basename = os.path.basename(req_file)
                print(f"[flagscale] Installing deps from {plat}/{basename}...", file=sys.stderr)

            rc = subprocess.call(cmd, env=clean_env)
            if rc != 0:
                print(
                    f"[flagscale] Warning: install from {req_file} failed (exit {rc}).",
                    file=sys.stderr,
                )

        seen_deps.update(d for d in deps if d not in pkg_opts)

        # Install annotated packages (need special pip flags).
        for pkg, opts in pkg_opts.items():
            if pkg in seen_deps:
                continue
            seen_deps.add(pkg)
            pkg_name = pkg.split("@")[0].strip()
            opt_str = " ".join(opts)
            cmd = [sys.executable, "-m", "pip", "install"]
            cmd.extend(opts)
            if verbose:
                cmd.append("-" + "v" * verbose)
            for pip_opt in pip_opts:
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
                full_opts = f"{opt_str} {' '.join(pip_opts)}".strip()
                print(
                    f"[flagscale] Warning: auto-install of {pkg_name} failed (exit {rc}).",
                    file=sys.stderr,
                )
                print(
                    f'[flagscale] Install manually: pip install {full_opts} "{pkg}"',
                    file=sys.stderr,
                )

    # FlagCX build.
    adaptor = _get_flagcx_adaptor(requested, req_platforms)
    if adaptor is not None:
        _build_flagcx(adaptor)


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


def _build_flagcx(adaptor):
    """Build FlagCX native library and install its torch plugin.

    ``adaptor`` selects the hardware backend (must be a key in
    ``_ADAPTOR_TO_MAKE_FLAG``).  The function auto-initializes the git
    submodule if the source tree is missing, runs ``make`` with the
    appropriate ``USE_*`` flag, and then pip-installs the torch plugin.
    """
    if adaptor not in _ADAPTOR_TO_MAKE_FLAG:
        raise ValueError(
            f"Unknown FlagCX adaptor {adaptor!r}. "
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
# 1. pip install .                       -> CLI only (typer)
# 2. pip install ".[cuda,train]"         -> CLI + platform/task deps (subprocess)
#    Platform + task extras are flat and comma-separated.  The correct
#    requirements file is determined by the combination (e.g. cuda + train ->
#    requirements/cuda/train.txt).  Annotated packages (e.g. megatron-core
#    with --no-build-isolation) are auto-installed via subprocess.
#    Requires torch to be pre-installed. Use -v/-vvv for detail.
# 3. pip install ".[cuda,all,dev]"       -> CLI + all CUDA deps + dev tools
# 4. pip install -r requirements/cuda/train.txt  -> pip deps with index URLs
# 5. flagscale install                   -> Full installation (apt + pip + ALL)
#
# FlagCX (optional native communication library):
#   pip install ".[cuda,flagcx]"         -> Build FlagCX for NVIDIA (adaptor
#                                           inferred from platform)
#   pip install ".[cuda,train,flagcx]"   -> Combined with task extras
# ---------------------------------------------------------------------------

# Only extras_require is dynamic — everything else comes from pyproject.toml.
setup(extras_require=EXTRAS)

# Setuptools commands that only collect metadata — no building needed.
# pip's PEP 517 build workflow invokes setup.py three times:
#   1. egg_info  — "Getting requirements to build wheel" (metadata only)
#   2. dist_info — "Preparing metadata (pyproject.toml)" (metadata only)
#   3. bdist_wheel — actual wheel build
# Without this guard, _install_platform_task_deps() would run all three times.
# Skipping during metadata phases ensures a single build during bdist_wheel.
_METADATA_COMMANDS = frozenset({"egg_info", "dist_info"})

# Only auto-install when setup.py is executed directly (pip install, python setup.py ...),
# not when imported by tests or other modules.
if __name__ == "__main__":
    if not (_METADATA_COMMANDS & set(sys.argv)):
        _install_platform_task_deps()

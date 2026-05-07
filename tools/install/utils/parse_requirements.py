#!/usr/bin/env python3
"""Parse requirements files with per-package annotation support.

Shared by setup.py (Python API) and tools/install shell scripts (CLI).

Annotation syntax — a comment line before a package applies options to that
package only, then resets.  Multiple annotations stack::

    # [--no-build-isolation]
    megatron-core @ git+https://github.com/...

The annotations are plain comments, so the file remains valid for
``pip install -r``.
"""

import os
import re
import sys


def parse_requirements(req_file):
    """Parse a requirements file, recursively resolving ``-r`` includes.

    Parameters
    ----------
    req_file : str
        **Absolute** path to the requirements file.

    Returns
    -------
    (deps, pip_options, pkg_options)
        - *deps*: list of PEP 508 dependency specifiers (normal packages)
        - *pip_options*: list of pip option strings (e.g.
          ``'--extra-index-url https://...'``) preserved as-is
        - *pkg_options*: dict mapping package specifier → list of per-package
          pip options (e.g. ``{"megatron-core @ git+...": ["--no-build-isolation"]}``)
    """
    if not os.path.isfile(req_file):
        return [], [], {}

    deps = []
    pip_options = []
    pkg_options = {}
    pending_options = []
    base_dir = os.path.dirname(req_file)

    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = re.match(r"^#\s*\[([^\]]+)\]\s*$", line)
                if m:
                    opts = m.group(1).split()
                    if opts and opts[0].startswith("-"):
                        pending_options.extend(opts)
                continue
            if line.startswith("-r "):
                included = line[3:].strip()
                included_path = os.path.normpath(os.path.join(base_dir, included))
                sub_deps, sub_opts, sub_pkg_opts = parse_requirements(included_path)
                deps.extend(sub_deps)
                pip_options.extend(sub_opts)
                pkg_options.update(sub_pkg_opts)
            elif line.startswith("-"):
                pip_options.append(line)
            else:
                deps.append(line)
                if pending_options:
                    pkg_options[line] = list(pending_options)
                    pending_options = []

    return deps, pip_options, pkg_options


# ---------------------------------------------------------------------------
# CLI — called by shell scripts in tools/install/
# ---------------------------------------------------------------------------


def _cmd_annotations(req_file):
    """Print PKG_SPEC<TAB>OPTIONS for each annotated package."""
    _, _, pkg_options = parse_requirements(os.path.abspath(req_file))
    for pkg, opts in pkg_options.items():
        print(f"{pkg}\t{' '.join(opts)}")


def _cmd_filter(req_file, output_file):
    """Write a copy of *req_file* with annotated packages commented out."""
    abs_req = os.path.abspath(req_file)
    _, _, pkg_options = parse_requirements(abs_req)
    if not pkg_options:
        # No annotations — just copy the file.
        with open(abs_req) as src, open(output_file, "w") as dst:
            dst.write(src.read())
        return

    # Build set of annotated package lines for fast lookup.
    annotated = set(pkg_options)
    pending = False

    with open(abs_req) as src, open(output_file, "w") as dst:
        for line in src:
            stripped = line.strip()
            if stripped.startswith("#"):
                m = re.match(r"^#\s*\[([^\]]+)\]\s*$", stripped)
                if m:
                    opts = m.group(1).split()
                    if opts and opts[0].startswith("-"):
                        pending = True
                        continue  # skip annotation comment
                dst.write(line)
                continue
            if not stripped:
                dst.write(line)
                continue
            if pending and stripped in annotated:
                dst.write(f"# [skipped by installer] {stripped}\n")
                pending = False
            else:
                dst.write(line)
                pending = False


def _usage():
    prog = os.path.basename(sys.argv[0])
    print(f"Usage: {prog} <command> <args>", file=sys.stderr)
    print(f"  {prog} annotations <req_file>", file=sys.stderr)
    print(f"  {prog} filter <req_file> <output_file>", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _usage()
    cmd = sys.argv[1]
    if cmd == "annotations":
        _cmd_annotations(sys.argv[2])
    elif cmd == "filter":
        if len(sys.argv) < 4:
            _usage()
        _cmd_filter(sys.argv[2], sys.argv[3])
    else:
        _usage()

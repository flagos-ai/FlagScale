import os
import sys

from setuptools import setup
from setuptools._distutils._log import log

# Add current directory to path to import version
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from version import FLAGSCALE_VERSION


def _read_requirements_file(requirements_path):
    """Read the requirements file and return the dependency list"""
    requirements = []
    try:
        with open(requirements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                requirements.append(line)
    except FileNotFoundError:
        print(f"[WARNING] Requirements file not found: {requirements_path}")
        return []
    return requirements


def deduplicate_dependencies(dependencies):
    """Deduplicate the dependencies"""
    seen = set()
    result = []
    for dep in dependencies:
        pkg_name = (
            dep.split("==")[0]
            .split(">=")[0]
            .split("<=")[0]
            .split(">")[0]
            .split("<")[0]
            .split("!=")[0]
            .strip()
        )
        pkg_name_lower = pkg_name.lower()
        if pkg_name_lower not in seen:
            seen.add(pkg_name_lower)
            result.append(dep)
    return result


def _get_install_requires():
    """get install_requires list"""
    install_requires = []

    install_requires.extend(_read_requirements_file("requirements/requirements-base.txt"))
    install_requires.extend(_read_requirements_file("requirements/requirements-common.txt"))
    core_deps = ["setuptools==79.0.1", "packaging>=24.2", "importlib_metadata>=8.5.0"]

    all_deps = install_requires + core_deps
    result = deduplicate_dependencies(all_deps)
    log.info(f"[build] install_requires Unique dependencies: {result}")

    return result


setup(
    name="flag_scale",
    version=FLAGSCALE_VERSION,
    description="FlagScale is a comprehensive toolkit designed to support the entire lifecycle of large models, developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI). ",
    url="https://github.com/FlagOpen/FlagScale",
    packages=[
        "flag_scale",
        "flag_scale.flagscale",
        "flag_scale.examples",
        "flag_scale.tools",
        "flag_scale.tests",
    ],
    package_dir={
        "flag_scale": "",
        "flag_scale.flagscale": "flagscale",
        "flag_scale.examples": "examples",
        "flag_scale.tools": "tools",
        "flag_scale.tests": "tests",
    },
    package_data={
        "flag_scale.flagscale": ["**/*"],
        "flag_scale.examples": ["**/*"],
        "flag_scale.tools": ["**/*"],
        "flag_scale.tests": ["**/*"],
    },
    install_requires=_get_install_requires(),
    entry_points={"console_scripts": ["flagscale=flag_scale.flagscale.cli:flagscale"]},
)

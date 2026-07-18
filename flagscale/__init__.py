# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def _get_version() -> str:
    """Get version from importlib.metadata or parse pyproject.toml as fallback."""
    try:
        from importlib.metadata import version

        return version("flagscale")
    except Exception:
        pass

    # Fallback: parse pyproject.toml for development mode (Python 3.11+)
    try:
        from pathlib import Path

        import tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                return data.get("project", {}).get("version", "0.0.0")
    except Exception:
        pass

    return "0.0.0"


__version__ = _get_version()

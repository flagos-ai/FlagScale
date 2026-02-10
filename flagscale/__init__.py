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

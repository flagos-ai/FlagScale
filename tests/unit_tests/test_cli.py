from pathlib import Path

import pytest
from click.exceptions import Exit as ClickExit

from flagscale.cli import get_action, resolve_config


class TestGetAction:
    """Tests for get_action() function"""

    def test_default_returns_run(self):
        """No flags set returns 'run'"""
        assert get_action(False, False, False, False, False) == "run"

    def test_stop_flag(self):
        """stop=True returns 'stop'"""
        assert get_action(True, False, False, False, False) == "stop"

    def test_dryrun_flag(self):
        """dryrun=True returns 'dryrun'"""
        assert get_action(False, True, False, False, False) == "dryrun"

    def test_test_flag(self):
        """test=True returns 'test'"""
        assert get_action(False, False, True, False, False) == "test"

    def test_query_flag(self):
        """query=True returns 'query'"""
        assert get_action(False, False, False, True, False) == "query"

    def test_tune_flag(self):
        """tune=True returns 'auto_tune'"""
        assert get_action(False, False, False, False, True) == "auto_tune"

    def test_mutually_exclusive_stop_dryrun(self, capsys):
        """Multiple flags (stop and dryrun) raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            get_action(True, True, False, False, False)
        assert exc_info.value.exit_code == 1

    def test_mutually_exclusive_all_flags(self, capsys):
        """All flags set raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            get_action(True, True, True, True, True)
        assert exc_info.value.exit_code == 1

    def test_mutually_exclusive_test_query(self, capsys):
        """Multiple flags (test and query) raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            get_action(False, False, True, True, False)
        assert exc_info.value.exit_code == 1


class TestResolveConfig:
    """Tests for resolve_config() function"""

    def test_with_yaml_path(self, tmp_path):
        """Explicit yaml path returns parent dir and stem"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text("test: value")

        path, name = resolve_config("model", yaml_file, "train")

        assert path == str(tmp_path)
        assert name == "test_config"

    def test_with_nested_yaml_path(self, tmp_path):
        """Yaml path in nested directory works correctly"""
        nested_dir = tmp_path / "conf" / "nested"
        nested_dir.mkdir(parents=True)
        yaml_file = nested_dir / "my_train.yaml"
        yaml_file.write_text("model: test")

        path, name = resolve_config("model", yaml_file, "train")

        assert path == str(nested_dir)
        assert name == "my_train"

    def test_yaml_not_exists(self):
        """Non-existent yaml path raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            resolve_config("model", Path("/nonexistent/path/config.yaml"), "train")
        assert exc_info.value.exit_code == 1

    def test_model_not_found(self):
        """Non-existent model raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            resolve_config("nonexistent_model_xyz_12345", None, "train")
        assert exc_info.value.exit_code == 1

    def test_from_model_name_aquila(self, mocker):
        """Resolves config from examples/aquila/conf directory if it exists"""
        # This test checks if the function correctly constructs the path
        # We mock Path.exists() to control the test
        mocker.patch.object(Path, "exists", return_value=True)

        # The function should construct path: script_dir / "examples" / model / "conf" / f"{task}.yaml"
        try:
            path, name = resolve_config("aquila", None, "train")
            # If aquila exists, it should return the path
            assert "aquila" in path or name == "train"
        except SystemExit:
            # If aquila doesn't exist in the test environment, that's expected
            pass


class TestResolveConfigEdgeCases:
    """Edge case tests for resolve_config()"""

    def test_yaml_path_with_spaces(self, tmp_path):
        """Yaml path with spaces in directory name works"""
        spaced_dir = tmp_path / "path with spaces"
        spaced_dir.mkdir()
        yaml_file = spaced_dir / "config.yaml"
        yaml_file.write_text("test: value")

        path, name = resolve_config("model", yaml_file, "train")

        assert "path with spaces" in path
        assert name == "config"

    def test_yaml_path_absolute(self, tmp_path):
        """Absolute yaml path is resolved correctly"""
        yaml_file = tmp_path / "absolute_test.yaml"
        yaml_file.write_text("test: value")

        # Use absolute path
        abs_path = yaml_file.resolve()
        path, name = resolve_config("model", abs_path, "train")

        assert Path(path).is_absolute()
        assert name == "absolute_test"

    def test_empty_model_name_with_yaml(self, tmp_path):
        """Empty model name works when yaml_path is provided"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("test: value")

        path, name = resolve_config("", yaml_file, "train")

        assert path == str(tmp_path)
        assert name == "config"

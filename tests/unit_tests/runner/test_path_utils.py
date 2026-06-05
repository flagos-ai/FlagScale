import os

import pytest
from omegaconf import OmegaConf

from flagscale.runner.utils import (
    get_cwd_dir,
    get_pkg_dir,
    resolve_path,
    setup_exp_dir,
    setup_logging_dirs,
)

# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_absolute_path_unchanged(self):
        assert resolve_path("/tmp/some/path") == "/tmp/some/path"

    def test_relative_path_resolved_against_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = resolve_path("data/train")
        assert result == os.path.join(str(tmp_path), "data", "train")

    def test_dot_relative_path(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = resolve_path("./models/checkpoint")
        assert result == os.path.join(str(tmp_path), "models", "checkpoint")

    def test_parent_relative_path(self, monkeypatch, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        result = resolve_path("../sibling")
        assert result == os.path.join(str(tmp_path), "sibling")

    def test_result_is_always_absolute(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert os.path.isabs(resolve_path("relative"))

    def test_check_exists_warns_on_missing(self, mocker):
        mock_warn = mocker.patch("flagscale.runner.utils.logger.warning")
        resolve_path("/nonexistent/path/xyzzy", config_key="test.key", check_exists=True)
        mock_warn.assert_called_once()
        assert "test.key" in mock_warn.call_args[0][0]
        assert "/nonexistent/path/xyzzy" in mock_warn.call_args[0][0]

    def test_check_exists_no_warning_for_existing(self, tmp_path, mocker):
        mock_warn = mocker.patch("flagscale.runner.utils.logger.warning")
        existing = tmp_path / "real_file.txt"
        existing.write_text("hello")
        resolve_path(str(existing), config_key="test.key", check_exists=True)
        mock_warn.assert_not_called()

    def test_check_exists_false_does_not_warn_on_missing(self, mocker):
        mock_warn = mocker.patch("flagscale.runner.utils.logger.warning")
        resolve_path("/nonexistent/path/xyzzy", config_key="test.key", check_exists=False)
        mock_warn.assert_not_called()

    def test_default_config_key_is_empty(self, mocker):
        mock_warn = mocker.patch("flagscale.runner.utils.logger.warning")
        resolve_path("/nonexistent/abc", check_exists=True)
        assert "Config ''" in mock_warn.call_args[0][0]

    def test_raise_missing_on_nonexistent(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_path("/nonexistent/path/xyzzy", config_key="test.key", raise_missing=True)

    def test_raise_missing_includes_config_key(self):
        with pytest.raises(FileNotFoundError, match="test.key"):
            resolve_path("/nonexistent/path/xyzzy", config_key="test.key", raise_missing=True)

    def test_raise_missing_no_error_for_existing(self, tmp_path):
        existing = tmp_path / "real_file.txt"
        existing.write_text("hello")
        result = resolve_path(str(existing), config_key="test.key", raise_missing=True)
        assert result == str(existing)

    def test_raise_missing_takes_precedence_over_check_exists(self):
        with pytest.raises(FileNotFoundError):
            resolve_path("/nonexistent/xyz", check_exists=True, raise_missing=True)


# ---------------------------------------------------------------------------
# setup_exp_dir
# ---------------------------------------------------------------------------


class TestSetupExpDir:
    def _make_config(self, exp_dir):
        return OmegaConf.create({"experiment": {"exp_dir": exp_dir}})

    def test_creates_directory(self, tmp_path):
        target = str(tmp_path / "experiment_output")
        config = self._make_config(target)
        result = setup_exp_dir(config)
        assert os.path.isdir(target)
        assert result == target

    def test_returns_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = self._make_config("rel_exp_dir")
        result = setup_exp_dir(config)
        assert os.path.isabs(result)
        assert result == os.path.join(str(tmp_path), "rel_exp_dir")

    def test_idempotent_on_existing_dir(self, tmp_path):
        target = str(tmp_path / "existing")
        os.makedirs(target)
        marker = os.path.join(target, "marker.txt")
        with open(marker, "w") as f:
            f.write("keep")
        config = self._make_config(target)
        setup_exp_dir(config)
        assert os.path.isfile(marker)

    def test_creates_nested_directories(self, tmp_path):
        target = str(tmp_path / "a" / "b" / "c")
        config = self._make_config(target)
        result = setup_exp_dir(config)
        assert os.path.isdir(target)
        assert result == target


# ---------------------------------------------------------------------------
# get_pkg_dir
# ---------------------------------------------------------------------------


class TestGetPkgDir:
    def test_returns_absolute_path(self):
        assert os.path.isabs(get_pkg_dir())

    def test_points_to_parent_of_flagscale_package(self):
        pkg_dir = get_pkg_dir()
        assert os.path.isdir(os.path.join(pkg_dir, "flagscale"))

    def test_is_stable_regardless_of_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert os.path.isdir(os.path.join(get_pkg_dir(), "flagscale"))


# ---------------------------------------------------------------------------
# get_cwd_dir
# ---------------------------------------------------------------------------


class TestGetCwdDir:
    def test_none_returns_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert get_cwd_dir(None) == str(tmp_path)

    def test_no_arg_returns_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert get_cwd_dir() == str(tmp_path)

    def test_returns_absolute_path(self):
        assert os.path.isabs(get_cwd_dir())

    def test_override_returns_resolved_path(self, tmp_path):
        override = str(tmp_path / "custom_root")
        os.makedirs(override)
        result = get_cwd_dir(override)
        assert result == override

    def test_override_relative_resolved(self, monkeypatch, tmp_path):
        custom = tmp_path / "my_root"
        custom.mkdir()
        monkeypatch.chdir(tmp_path)
        result = get_cwd_dir("my_root")
        assert os.path.isabs(result)
        assert result == str(custom)

    def test_override_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="root_dir"):
            get_cwd_dir("/nonexistent/root_dir_xyz")


# ---------------------------------------------------------------------------
# setup_logging_dirs
# ---------------------------------------------------------------------------


class TestSetupLoggingDirs:
    def _make_logging_config(self, **kwargs):
        cfg = OmegaConf.create(kwargs)
        OmegaConf.set_struct(cfg, False)
        return cfg

    def test_default_subdir(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        os.makedirs(exp_dir)
        logging_config = self._make_logging_config()
        result = setup_logging_dirs(logging_config, exp_dir)
        expected_log_dir = os.path.join(exp_dir, "logs")
        assert result == expected_log_dir
        assert logging_config.log_dir == expected_log_dir
        assert logging_config.scripts_dir == os.path.join(expected_log_dir, "scripts")
        assert logging_config.pids_dir == os.path.join(expected_log_dir, "pids")

    def test_custom_subdir(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        os.makedirs(exp_dir)
        logging_config = self._make_logging_config()
        result = setup_logging_dirs(logging_config, exp_dir, log_subdir="serve_logs")
        expected_log_dir = os.path.join(exp_dir, "serve_logs")
        assert result == expected_log_dir
        assert logging_config.log_dir == expected_log_dir

    def test_existing_log_dir_in_config(self, tmp_path):
        custom_log_dir = str(tmp_path / "custom_logs")
        os.makedirs(custom_log_dir)
        logging_config = self._make_logging_config(log_dir=custom_log_dir)
        result = setup_logging_dirs(logging_config, str(tmp_path))
        assert result == custom_log_dir
        assert logging_config.log_dir == custom_log_dir
        assert logging_config.scripts_dir == os.path.join(custom_log_dir, "scripts")
        assert logging_config.pids_dir == os.path.join(custom_log_dir, "pids")

    def test_relative_log_dir_in_config_resolved(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        logging_config = self._make_logging_config(log_dir="relative_logs")
        result = setup_logging_dirs(logging_config, str(tmp_path))
        assert os.path.isabs(result)
        assert result == os.path.join(str(tmp_path), "relative_logs")

    def test_scripts_and_pids_are_subdirs_of_log_dir(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        os.makedirs(exp_dir)
        logging_config = self._make_logging_config()
        setup_logging_dirs(logging_config, exp_dir)
        assert logging_config.scripts_dir.startswith(logging_config.log_dir)
        assert logging_config.pids_dir.startswith(logging_config.log_dir)

    def test_none_log_dir_uses_exp_dir(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        os.makedirs(exp_dir)
        logging_config = self._make_logging_config(log_dir=None)
        result = setup_logging_dirs(logging_config, exp_dir)
        assert result == os.path.join(exp_dir, "logs")

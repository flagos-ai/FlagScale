import os
import sys
import unittest.mock

import pytest

# Mock setuptools.setup before importing setup module to prevent it from
# running at import time and interfering with pytest.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with unittest.mock.patch("setuptools.setup"):
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from setup import (
        EXTRAS,
        _build_flagcx,
        _get_flagcx_adaptor,
        _install_platform_task_deps,
        build_extras,
        parse_requirements,
    )


# --- Dynamic discovery ---


def _discover_platforms():
    """Discover platform directories under requirements/."""
    req_dir = os.path.join(PROJECT_ROOT, "requirements")
    return sorted(e for e in os.listdir(req_dir) if os.path.isdir(os.path.join(req_dir, e)))


DISCOVERED_PLATFORMS = _discover_platforms()


# --- Fixture ---


@pytest.fixture
def req_tree(tmp_path):
    """Temporary directory for isolated requirements file tests."""
    return tmp_path


# --- TestParseRequirements: unit tests with tmp files ---


class TestParseRequirements:
    """Tests for parse_requirements() function"""

    def test_nonexistent_file(self, tmp_path):
        """Non-existent file returns empty tuples/dict"""
        deps, opts, pkg_opts = parse_requirements(str(tmp_path / "nonexistent" / "file.txt"))
        assert deps == []
        assert opts == []
        assert pkg_opts == {}

    def test_simple_requirements(self, req_tree):
        """Parses simple package specifiers"""
        req_file = req_tree / "req.txt"
        req_file.write_text("numpy==1.26.4\nscipy==1.14.1\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["numpy==1.26.4", "scipy==1.14.1"]
        assert opts == []
        assert pkg_opts == {}

    def test_skips_comments_and_blanks(self, req_tree):
        """Skips comment lines and blank lines"""
        req_file = req_tree / "req.txt"
        req_file.write_text("# comment\nnumpy==1.26.4\n\n# another comment\nscipy==1.14.1\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["numpy==1.26.4", "scipy==1.14.1"]
        assert opts == []
        assert pkg_opts == {}

    def test_collects_find_links(self, req_tree):
        """Collects --find-links as a pip option"""
        req_file = req_tree / "req.txt"
        req_file.write_text("--find-links /some/path\nnumpy==1.26.4\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["numpy==1.26.4"]
        assert opts == ["--find-links /some/path"]
        assert pkg_opts == {}

    def test_collects_extra_index_url(self, req_tree):
        """Collects --extra-index-url as a pip option"""
        req_file = req_tree / "req.txt"
        req_file.write_text(
            "--extra-index-url https://download.pytorch.org/whl/cu128\ntorch==2.9.1\n"
        )

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["torch==2.9.1"]
        assert opts == ["--extra-index-url https://download.pytorch.org/whl/cu128"]
        assert pkg_opts == {}

    def test_collects_index_url(self, req_tree):
        """Collects --index-url as a pip option"""
        req_file = req_tree / "req.txt"
        req_file.write_text("--index-url https://internal.example.com/simple\nnumpy==1.26.4\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["numpy==1.26.4"]
        assert opts == ["--index-url https://internal.example.com/simple"]
        assert pkg_opts == {}

    def test_collects_multiple_options(self, req_tree):
        """Collects multiple different pip options"""
        req_file = req_tree / "req.txt"
        req_file.write_text(
            "--extra-index-url https://example.com/whl\n"
            "--trusted-host example.com\n"
            "--pre\n"
            "torch==2.9.1\n"
        )

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["torch==2.9.1"]
        assert opts == [
            "--extra-index-url https://example.com/whl",
            "--trusted-host example.com",
            "--pre",
        ]
        assert pkg_opts == {}

    def test_collects_short_options(self, req_tree):
        """Collects short-form pip options like -i, -f"""
        req_file = req_tree / "req.txt"
        req_file.write_text(
            "-i https://internal.example.com/simple\n-f /local/wheels\nnumpy==1.26.4\n"
        )

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["numpy==1.26.4"]
        assert opts == [
            "-i https://internal.example.com/simple",
            "-f /local/wheels",
        ]
        assert pkg_opts == {}

    def test_resolves_r_includes(self, req_tree):
        """Recursively resolves -r includes, collecting deps and options"""
        common = req_tree / "common.txt"
        common.write_text("typer>=0.9.0\npyyaml==6.0.2\n")

        base = req_tree / "base.txt"
        base.write_text("--extra-index-url https://example.com/whl\n-r common.txt\ntorch==2.9.1\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "base.txt"))

        assert deps == ["typer>=0.9.0", "pyyaml==6.0.2", "torch==2.9.1"]
        assert opts == ["--extra-index-url https://example.com/whl"]
        assert pkg_opts == {}

    def test_resolves_nested_includes_with_options(self, req_tree):
        """Resolves nested -r includes, collecting options from all levels"""
        common = req_tree / "common.txt"
        common.write_text("numpy==1.26.4\n")

        sub = req_tree / "cuda"
        sub.mkdir()

        base = sub / "base.txt"
        base.write_text(
            "--extra-index-url https://download.pytorch.org/whl/cu128\n"
            "-r ../common.txt\ntorch==2.9.1\n"
        )

        train = sub / "train.txt"
        train.write_text("-r ./base.txt\nmegatron-core\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "cuda" / "train.txt"))

        assert deps == ["numpy==1.26.4", "torch==2.9.1", "megatron-core"]
        assert opts == ["--extra-index-url https://download.pytorch.org/whl/cu128"]
        assert pkg_opts == {}

    def test_annotation_applies_to_next_package_only(self, req_tree):
        """# [--option] annotation applies only to the next package line"""
        req_file = req_tree / "req.txt"
        req_file.write_text(
            "numpy==1.26.4\n"
            "# [--no-build-isolation]\n"
            "megatron-core @ git+https://github.com/flagos-ai/Megatron-LM-FL.git\n"
            "scipy==1.14.1\n"
        )

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == [
            "numpy==1.26.4",
            "megatron-core @ git+https://github.com/flagos-ai/Megatron-LM-FL.git",
            "scipy==1.14.1",
        ]
        assert opts == []
        assert pkg_opts == {
            "megatron-core @ git+https://github.com/flagos-ai/Megatron-LM-FL.git": [
                "--no-build-isolation"
            ]
        }

    def test_annotation_does_not_affect_subsequent_packages(self, req_tree):
        """Packages after the annotated one are normal deps"""
        req_file = req_tree / "req.txt"
        req_file.write_text("# [--no-build-isolation]\npkg-a\npkg-b\npkg-c\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["pkg-a", "pkg-b", "pkg-c"]
        assert pkg_opts == {"pkg-a": ["--no-build-isolation"]}

    def test_multiple_annotations_stack(self, req_tree):
        """Multiple # [...] comments before one package merge their options"""
        req_file = req_tree / "req.txt"
        req_file.write_text("# [--no-build-isolation]\n# [--verbose]\npkg-a\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["pkg-a"]
        assert pkg_opts == {"pkg-a": ["--no-build-isolation", "--verbose"]}

    def test_multiple_options_in_one_bracket(self, req_tree):
        """Multiple options in a single # [...] comment"""
        req_file = req_tree / "req.txt"
        req_file.write_text("# [--no-build-isolation --verbose]\npkg-a\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["pkg-a"]
        assert pkg_opts == {"pkg-a": ["--no-build-isolation", "--verbose"]}

    def test_annotation_propagates_through_includes(self, req_tree):
        """pkg_options from -r included files are collected in the parent"""
        sub = req_tree / "cuda"
        sub.mkdir()

        child = sub / "child.txt"
        child.write_text("# [--no-build-isolation]\nchild-nbi-pkg\n")

        parent = sub / "parent.txt"
        parent.write_text("normal-pkg\n-r ./child.txt\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "cuda" / "parent.txt"))

        assert deps == ["normal-pkg", "child-nbi-pkg"]
        assert pkg_opts == {"child-nbi-pkg": ["--no-build-isolation"]}

    def test_annotation_with_includes_does_not_consume(self, req_tree):
        """Pending options are NOT consumed by -r includes"""
        base = req_tree / "base.txt"
        base.write_text("torch==2.9.1\n")

        train = req_tree / "train.txt"
        train.write_text("# [--no-build-isolation]\n-r ./base.txt\nmegatron-core\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "train.txt"))

        assert deps == ["torch==2.9.1", "megatron-core"]
        assert pkg_opts == {"megatron-core": ["--no-build-isolation"]}

    def test_regular_comments_not_treated_as_annotation(self, req_tree):
        """Regular comments are not confused with annotations"""
        req_file = req_tree / "req.txt"
        req_file.write_text(
            "# This is a regular comment about no-build-isolation\n"
            "numpy==1.26.4\n"
            "# another comment\n"
            "scipy==1.14.1\n"
        )

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["numpy==1.26.4", "scipy==1.14.1"]
        assert pkg_opts == {}

    def test_bracket_without_dashes_ignored(self, req_tree):
        """# [word] without -- prefix is not treated as annotation"""
        req_file = req_tree / "req.txt"
        req_file.write_text("# [no-build-isolation]\nnumpy==1.26.4\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["numpy==1.26.4"]
        assert pkg_opts == {}

    def test_pep508_git_url_passes_through(self, req_tree):
        """PEP 508 git URL specifiers pass through as deps"""
        req_file = req_tree / "req.txt"
        req_file.write_text("megatron-core @ git+https://github.com/flagos-ai/Megatron-LM-FL.git\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["megatron-core @ git+https://github.com/flagos-ai/Megatron-LM-FL.git"]
        assert opts == []
        assert pkg_opts == {}

    def test_pep508_direct_url_passes_through(self, req_tree):
        """PEP 508 direct URL specifiers (wheel URLs) pass through as deps"""
        req_file = req_tree / "req.txt"
        req_file.write_text("some-pkg @ https://internal.example.com/wheels/some-pkg-1.0.whl\n")

        deps, opts, pkg_opts = parse_requirements(str(req_tree / "req.txt"))

        assert deps == ["some-pkg @ https://internal.example.com/wheels/some-pkg-1.0.whl"]
        assert opts == []
        assert pkg_opts == {}

    @pytest.mark.parametrize("platform", DISCOVERED_PLATFORMS)
    def test_real_platform_base(self, platform):
        """Parse real requirements/<platform>/base.txt successfully"""
        deps, opts, pkg_opts = parse_requirements(
            os.path.join(PROJECT_ROOT, "requirements", platform, "base.txt")
        )
        assert len(deps) > 0, f"requirements/{platform}/base.txt produced no deps"

    def test_real_common_txt(self):
        """Parse the real requirements/common.txt"""
        deps, opts, pkg_opts = parse_requirements(
            os.path.join(PROJECT_ROOT, "requirements", "common.txt")
        )

        assert len(deps) > 0
        assert any("typer" in dep for dep in deps)


# --- TestBuildExtras: integration tests with real requirements ---


class TestBuildExtras:
    """Tests for build_extras() function"""

    def test_returns_correct_types(self):
        """build_extras() returns (extras_dict, platforms_set, tasks_set)"""
        extras, platforms, tasks = build_extras()
        assert isinstance(extras, dict)
        assert isinstance(platforms, set)
        assert isinstance(tasks, set)

    def test_has_dev_extra(self):
        """Has 'dev' extra from requirements/dev.txt with actual deps"""
        extras, _, _ = build_extras()
        assert "dev" in extras
        assert any("pytest" in dep for dep in extras["dev"])

    @pytest.mark.parametrize("platform", DISCOVERED_PLATFORMS)
    def test_has_platform_extra_with_empty_deps(self, platform):
        """Each platform directory produces a platform extra with empty deps"""
        extras, platforms, _ = build_extras()
        assert platform in extras, f"Missing '{platform}' extra"
        assert extras[platform] == [], f"Platform extra '{platform}' should have empty deps"
        assert platform in platforms

    def test_has_task_extras_with_empty_deps(self):
        """Task names from requirements files are extras with empty deps"""
        extras, _, tasks = build_extras()
        for task in tasks:
            assert task in extras, f"Missing task extra '{task}'"
            assert extras[task] == [], f"Task extra '{task}' should have empty deps"

    def test_has_all_extra(self):
        """Has 'all' extra with empty deps"""
        extras, _, _ = build_extras()
        assert "all" in extras
        assert extras["all"] == []

    def test_has_flagcx_extra(self):
        """Has 'flagcx' extra with empty deps"""
        extras, _, _ = build_extras()
        assert "flagcx" in extras
        assert extras["flagcx"] == []

    def test_discovers_known_tasks(self):
        """Discovers expected task names from cuda directory"""
        _, _, tasks = build_extras()
        expected_tasks = {"train", "serve", "inference", "rl", "hetero_train"}
        assert expected_tasks.issubset(tasks), f"Missing tasks: {expected_tasks - tasks}"
        assert "all" not in tasks, "'all' should not be in tasks (it's a special marker)"

    @pytest.mark.parametrize("extra_name", sorted(EXTRAS.keys()))
    def test_extra_is_list_of_strings(self, extra_name):
        """All extras values are lists of strings"""
        deps = EXTRAS[extra_name]
        assert isinstance(deps, list), f"Extra '{extra_name}' is not a list"
        for dep in deps:
            assert isinstance(dep, str), f"Dep '{dep}' in extra '{extra_name}' is not a string"


# --- TestBuildFlagcx: unit tests for FlagCX build integration ---


class TestBuildFlagcx:
    """Tests for _build_flagcx() function"""

    def test_invalid_adaptor_raises(self):
        """_build_flagcx() raises ValueError for unknown adaptor"""
        with pytest.raises(ValueError, match="Unknown FlagCX adaptor 'unknown_hw'"):
            _build_flagcx("unknown_hw")

    @unittest.mock.patch("setup.subprocess.check_call")
    def test_make_command_nvidia(self, mock_call, monkeypatch):
        """Verify make is called with USE_NVIDIA=1 for nvidia adaptor"""
        monkeypatch.setattr("setup.os.path.isdir", lambda p: True)
        monkeypatch.setattr("setup.os.listdir", lambda p: ["Makefile"])

        _build_flagcx("nvidia")

        # First call: nested submodule init, second: make, third: pip install
        assert mock_call.call_count == 3
        make_cmd = mock_call.call_args_list[1][0][0]
        assert make_cmd[0] == "make"
        assert "USE_NVIDIA=1" in make_cmd
        assert any(arg.startswith("-j") for arg in make_cmd)

    @unittest.mock.patch("setup.subprocess.check_call")
    def test_make_command_ascend(self, mock_call, monkeypatch):
        """Verify make is called with USE_ASCEND=1 for ascend adaptor"""
        monkeypatch.setattr("setup.os.path.isdir", lambda p: True)
        monkeypatch.setattr("setup.os.listdir", lambda p: ["Makefile"])

        _build_flagcx("ascend")

        make_cmd = mock_call.call_args_list[1][0][0]
        assert "USE_ASCEND=1" in make_cmd

    @unittest.mock.patch("setup.subprocess.check_call")
    def test_torch_plugin_install(self, mock_call, monkeypatch):
        """Verify torch plugin pip install is called after make"""
        monkeypatch.setattr("setup.os.path.isdir", lambda p: True)
        monkeypatch.setattr("setup.os.listdir", lambda p: ["Makefile"])

        _build_flagcx("nvidia")

        # Calls: nested submodule init, make, pip install
        assert mock_call.call_count == 3
        pip_cmd = mock_call.call_args_list[2][0][0]
        assert pip_cmd[0] == sys.executable
        assert "-m" in pip_cmd
        assert "pip" in pip_cmd
        assert "--no-build-isolation" in pip_cmd
        assert "-e" not in pip_cmd
        # The plugin path should end with plugin/torch
        plugin_path = pip_cmd[-1]
        assert plugin_path.endswith(os.path.join("plugin", "torch"))
        # FLAGCX_ADAPTOR should be in the env
        call_env = mock_call.call_args_list[2][1].get("env", {})
        assert call_env.get("FLAGCX_ADAPTOR") == "nvidia"

    @unittest.mock.patch("setup.subprocess.check_call")
    def test_submodule_init_when_missing(self, mock_call, monkeypatch):
        """Verify git submodule init is called when source dir is empty"""
        # First call to isdir returns False (submodule missing), subsequent calls return True.
        call_count = {"n": 0}
        original_isdir = os.path.isdir

        def fake_isdir(p):
            if "FlagCX" in p and call_count["n"] == 0:
                call_count["n"] += 1
                return False
            return original_isdir(p)

        monkeypatch.setattr("setup.os.path.isdir", fake_isdir)
        monkeypatch.setattr("setup.os.listdir", lambda p: ["Makefile"])

        _build_flagcx("nvidia")

        # First call should be git submodule update
        assert mock_call.call_count == 3
        git_cmd = mock_call.call_args_list[0][0][0]
        assert git_cmd[0] == "git"
        assert "submodule" in git_cmd
        assert "--init" in git_cmd
        assert "--recursive" in git_cmd


# --- TestGetFlagcxAdaptor: unit tests for extras-based adaptor detection ---


class TestGetFlagcxAdaptor:
    """Tests for _get_flagcx_adaptor() function"""

    def test_returns_none_when_no_flagcx(self):
        """Returns None when flagcx is not in requested extras"""
        assert _get_flagcx_adaptor(["cuda", "train"], {"cuda"}) is None

    def test_returns_adaptor_for_cuda_platform(self):
        """Returns 'nvidia' adaptor when flagcx + cuda are requested"""
        assert _get_flagcx_adaptor(["cuda", "flagcx"], {"cuda"}) == "nvidia"

    def test_raises_without_platform(self):
        """Raises ValueError when flagcx is requested without a platform"""
        with pytest.raises(ValueError, match="requires a platform extra"):
            _get_flagcx_adaptor(["flagcx"], set())

    def test_raises_for_unknown_platform(self):
        """Raises ValueError when platform has no adaptor mapping"""
        with pytest.raises(ValueError, match="No FlagCX adaptor mapping"):
            _get_flagcx_adaptor(["flagcx", "unknown_plat"], {"unknown_plat"})

    def test_handles_pep685_normalization(self):
        """Handles PEP 685 normalized extra names"""
        assert _get_flagcx_adaptor(["flag_cx"], {"cuda"}) is None  # not "flagcx"
        assert _get_flagcx_adaptor(["flagcx"], {"cuda"}) == "nvidia"


# --- TestFlagcxExtras: unit tests for flagcx extra in EXTRAS ---


class TestFlagcxExtras:
    """Tests for flagcx extra in EXTRAS"""

    def test_single_flagcx_extra_exists(self):
        """A single 'flagcx' extra exists in EXTRAS"""
        assert "flagcx" in EXTRAS

    def test_flagcx_extra_has_empty_deps(self):
        """The 'flagcx' extra has empty dependency list"""
        assert EXTRAS["flagcx"] == []

    def test_no_per_adaptor_flagcx_extras(self):
        """No per-adaptor flagcx extras (e.g. flagcx-nvidia) exist"""
        for name in EXTRAS:
            assert not name.startswith("flagcx-"), (
                f"Found per-adaptor extra '{name}' — should be just 'flagcx'"
            )


# --- TestValidation: tests for extras validation logic ---


class TestValidation:
    """Tests for extras validation in _install_platform_task_deps()"""

    @unittest.mock.patch("setup.subprocess.call", return_value=0)
    def test_task_without_platform_raises(self, mock_call, monkeypatch):
        """Task extras without a platform raise ValueError"""
        monkeypatch.setattr("setup._get_requested_extras", lambda: ["train"])
        with pytest.raises(ValueError, match="require a platform extra"):
            _install_platform_task_deps()

    @unittest.mock.patch("setup.subprocess.call", return_value=0)
    def test_all_without_platform_raises(self, mock_call, monkeypatch):
        """'all' extra without a platform raises ValueError"""
        monkeypatch.setattr("setup._get_requested_extras", lambda: ["all"])
        with pytest.raises(ValueError, match="require a platform extra"):
            _install_platform_task_deps()

    def test_dev_without_platform_ok(self, monkeypatch):
        """'dev' extra without a platform does not raise"""
        monkeypatch.setattr("setup._get_requested_extras", lambda: ["dev"])
        # Should not raise — dev is platform-independent
        _install_platform_task_deps()

    @unittest.mock.patch("setup.subprocess.call", return_value=0)
    def test_cuda_train_installs_deps(self, mock_call, monkeypatch):
        """'cuda,train' combo installs deps via subprocess"""
        monkeypatch.setattr("setup._get_requested_extras", lambda: ["cuda", "train"])
        _install_platform_task_deps()
        # Should have called subprocess.call at least once for deps
        assert mock_call.call_count >= 1

    @unittest.mock.patch("setup._build_flagcx")
    @unittest.mock.patch("setup.subprocess.call", return_value=0)
    def test_cuda_train_flagcx_triggers_build(self, mock_call, mock_build, monkeypatch):
        """'cuda,train,flagcx' combo triggers FlagCX build"""
        monkeypatch.setattr("setup._get_requested_extras", lambda: ["cuda", "train", "flagcx"])
        _install_platform_task_deps()
        mock_build.assert_called_once_with("nvidia")

    @unittest.mock.patch("setup.subprocess.call", return_value=0)
    def test_flagcx_without_platform_raises(self, mock_call, monkeypatch):
        """'flagcx' without platform raises ValueError"""
        monkeypatch.setattr("setup._get_requested_extras", lambda: ["flagcx"])
        with pytest.raises(ValueError, match="requires a platform extra"):
            _install_platform_task_deps()

    def test_no_extras_is_noop(self, monkeypatch):
        """No extras requested does nothing"""
        monkeypatch.setattr("setup._get_requested_extras", lambda: None)
        # Should not raise
        _install_platform_task_deps()

    @unittest.mock.patch("setup.subprocess.call", return_value=0)
    def test_cuda_all_installs_all_task_files(self, mock_call, monkeypatch):
        """'cuda,all' installs all task files for the platform"""
        monkeypatch.setattr("setup._get_requested_extras", lambda: ["cuda", "all"])
        _install_platform_task_deps()
        # Should have called subprocess for deps
        assert mock_call.call_count >= 1

    @unittest.mock.patch("setup.subprocess.call", return_value=0)
    def test_platform_only_installs_base(self, mock_call, monkeypatch):
        """'cuda' alone installs only base.txt"""
        monkeypatch.setattr("setup._get_requested_extras", lambda: ["cuda"])
        _install_platform_task_deps()
        # Should install base deps
        assert mock_call.call_count >= 1

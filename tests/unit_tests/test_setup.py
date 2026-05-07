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
        PIP_OPTIONS,
        PKG_OPTIONS,
        build_extras,
        parse_requirements,
    )


# --- Dynamic discovery ---


def _discover_platforms():
    """Discover platform directories under requirements/."""
    req_dir = os.path.join(PROJECT_ROOT, "requirements")
    return sorted(e for e in os.listdir(req_dir) if os.path.isdir(os.path.join(req_dir, e)))


def _discover_extras():
    """Discover all extra names from the pre-computed EXTRAS dict."""
    return sorted(EXTRAS.keys())


PLATFORMS = _discover_platforms()
ALL_EXTRAS = _discover_extras()


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

    @pytest.mark.parametrize("platform", PLATFORMS)
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

    def test_returns_dicts(self):
        """build_extras() returns (extras_dict, pip_options_dict, pkg_options_dict)"""
        extras, pip_options, pkg_options = build_extras()
        assert isinstance(extras, dict)
        assert isinstance(pip_options, dict)
        assert isinstance(pkg_options, dict)

    def test_has_dev_extra(self):
        """Has 'dev' extra from requirements/dev.txt"""
        extras, _, _ = build_extras()
        assert "dev" in extras
        assert any("pytest" in dep for dep in extras["dev"])

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_has_platform_base_extra(self, platform):
        """Each platform directory produces a base extra"""
        extras, _, _ = build_extras()
        assert platform in extras, f"Missing '{platform}' extra from {platform}/base.txt"
        assert len(extras[platform]) > 0

    def test_pkg_options_keys_subset_of_extras(self):
        """PKG_OPTIONS keys are a subset of EXTRAS keys"""
        for name in PKG_OPTIONS:
            assert name in EXTRAS, f"PKG_OPTIONS has key '{name}' not found in EXTRAS"

    def test_annotated_packages_excluded_from_extras(self):
        """Packages with per-package options are NOT in extras_require"""
        for name, pkg_opts in PKG_OPTIONS.items():
            for pkg in pkg_opts:
                assert pkg not in EXTRAS.get(name, []), (
                    f"Annotated pkg '{pkg}' should not be in EXTRAS['{name}']"
                )

    def test_pkg_options_values_are_dicts(self):
        """PKG_OPTIONS values are dicts mapping pkg -> list of options"""
        for name, pkg_opts in PKG_OPTIONS.items():
            assert isinstance(pkg_opts, dict), f"PKG_OPTIONS['{name}'] is not a dict"
            for pkg, opts in pkg_opts.items():
                assert isinstance(pkg, str)
                assert isinstance(opts, list)
                assert opts[0].startswith("-"), (
                    f"Option '{opts}' for '{pkg}' in PKG_OPTIONS['{name}'] doesn't start with -"
                )

    @pytest.mark.parametrize("extra_name", ALL_EXTRAS)
    def test_extra_is_list_of_strings(self, extra_name):
        """All extras values are lists of strings"""
        deps = EXTRAS[extra_name]
        assert isinstance(deps, list), f"Extra '{extra_name}' is not a list"
        for dep in deps:
            assert isinstance(dep, str), f"Dep '{dep}' in extra '{extra_name}' is not a string"

    @pytest.mark.parametrize("extra_name", ALL_EXTRAS)
    def test_extra_has_deps_or_pkg_options(self, extra_name):
        """All extras have at least one dependency or per-package option"""
        has_deps = len(EXTRAS[extra_name]) > 0
        has_pkg_opts = extra_name in PKG_OPTIONS and len(PKG_OPTIONS[extra_name]) > 0
        assert has_deps or has_pkg_opts, f"Extra '{extra_name}' has no deps and no pkg_options"

    def test_pip_options_values_are_string_lists(self):
        """PIP_OPTIONS values are lists of option strings"""
        for name, opts in PIP_OPTIONS.items():
            assert isinstance(opts, list), f"PIP_OPTIONS['{name}'] is not a list"
            for opt in opts:
                assert isinstance(opt, str), (
                    f"Option '{opt}' in PIP_OPTIONS['{name}'] is not a string"
                )
                assert opt.startswith("-"), (
                    f"Option '{opt}' in PIP_OPTIONS['{name}'] doesn't start with -"
                )

    def test_pip_options_keys_subset_of_extras(self):
        """PIP_OPTIONS keys are a subset of EXTRAS keys"""
        for name in PIP_OPTIONS:
            assert name in EXTRAS, f"PIP_OPTIONS has key '{name}' not found in EXTRAS"

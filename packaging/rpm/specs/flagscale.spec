%global debug_package %{nil}

# Filter the auto-generated Requires for: hydra-core.
# Reason: hydra-core not in any distro; users install via pip.
# See packaging/INSTALL.md (or future flagos-packaging install docs) for the
# user-side pip install incantation.
%global __requires_exclude ^python3(\.[0-9]+)?dist\((hydra-core)\)$
Name:           python3-flagscale
Version:        1.0.0
Release:        1%{?dist}
Summary:        FlagScale large model training toolkit

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagScale
Source0:        %{url}/archive/refs/tags/v%{version}.tar.gz#/flagscale-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 68.0
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
BuildRequires:  pyproject-rpm-macros

%description
FlagScale is a comprehensive toolkit designed to support the entire
lifecycle of large models, including training, serving, inference,
and reinforcement learning. Developed by BAAI.

This package provides the core library and the flagscale CLI.
Heavy dependencies (PyTorch, Megatron, vLLM, etc.) should be
installed via pip extras: pip install "flagscale[cuda-train]"

%prep
%autosetup -n flagscale-%{version}

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files flagscale

%check
# Smoke find_spec test (no actual import) — verifies the built module
# lands at the expected sitelib path. Doesn't import flagscale because
# module-level deps (torch, megatron, vllm, ...) are install-time
# concerns, not packaging concerns, and aren't in the build container.
# PYTHONSAFEPATH=1 keeps the cwd (the unpacked source tree, which also
# contains flagscale/) off sys.path, so find_spec resolves against the
# installed copy under PYTHONPATH, not the source tree.
PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flagscale'); assert s and s.origin, 'flagscale not findable'; print('OK: flagscale at', s.origin)"

%files -f %{pyproject_files}
%license LICENSE
# Verified empirically: %%pyproject_save_files does NOT include the
# %%{_bindir}/flagscale console script entry, despite the wheel
# declaring [project.scripts]. List it explicitly to avoid the
# "Installed (but unpackaged) file(s) found" error.
%{_bindir}/flagscale

%changelog
* Mon Apr 13 2026 FlagOS Contributors <contact@flagos.io> - 1.0.0-1
- Initial packaging of FlagScale 1.0.0
- Core library and CLI tool
- Support for training, serving, inference, and RL workflows

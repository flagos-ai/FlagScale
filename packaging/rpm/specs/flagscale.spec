%global debug_package %{nil}

Name:           python3-flagscale
Version:        1.0.0
Release:        1%{?dist}
Summary:        FlagScale large model training toolkit

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagScale
Source0:        flagscale-%{version}.tar.gz

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

%files -f %{pyproject_files}
%license LICENSE
%{_bindir}/flagscale

%changelog
* Mon Apr 13 2026 FlagOS Contributors <contact@flagos.io> - 1.0.0-1
- Initial packaging of FlagScale 1.0.0
- Core library and CLI tool
- Support for training, serving, inference, and RL workflows

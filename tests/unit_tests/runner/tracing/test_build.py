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

"""Build selection must preserve CUDA while supporting installed RCCL headers."""

from unittest.mock import Mock

import pytest

from flagscale.runner.tracing import build


@pytest.fixture
def compiler(monkeypatch):
    for key in (
        "NCCL_HOME",
        "CUDA_HOME",
        "CUDA_PATH",
        "RCCL_HOME",
        "ROCM_HOME",
        "ROCM_PATH",
        "HIP_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(build.shutil, "which", lambda _: "/usr/bin/g++")
    invoked = Mock()
    monkeypatch.setattr(build.subprocess, "run", invoked)
    return invoked


def test_default_retains_nccl(tmp_path, monkeypatch, compiler):
    monkeypatch.setattr(build, "_rccl_include_dirs", lambda *_: [])
    result = build.build_probe(output=tmp_path / "probe.so")
    command = compiler.call_args.args[0]
    assert result == tmp_path / "probe.so"
    assert "-DFLAGSCALE_TRACE_USE_RCCL=1" not in command
    compiler.assert_called_once_with(command, check=True)


def test_auto_rccl_from_headers(tmp_path, monkeypatch, compiler):
    include = tmp_path / "include"
    include.mkdir()
    (include / "rccl.h").touch()
    monkeypatch.setattr(build, "_rccl_include_dirs", lambda *_: [include])
    build.build_probe(output=tmp_path / "probe.so")
    command = compiler.call_args.args[0]
    assert "-DFLAGSCALE_TRACE_USE_RCCL=1" in command
    assert "-D__HIP_PLATFORM_AMD__=1" in command
    assert command[-2:] == ["-I", str(include)]


@pytest.mark.parametrize("explicit", [True, False])
def test_cuda_root_wins_discovery(tmp_path, monkeypatch, compiler, explicit):
    (tmp_path / "rccl.h").touch()
    monkeypatch.setattr(build, "_rccl_include_dirs", lambda *_: [tmp_path])
    kwargs = {"cuda_home": "/cuda"} if explicit else {}
    if not explicit:
        monkeypatch.setenv("CUDA_HOME", "/cuda")
    build.build_probe(output=tmp_path / "probe.so", **kwargs)
    command = compiler.call_args.args[0]
    assert "-DFLAGSCALE_TRACE_USE_RCCL=1" not in command
    assert command[-2:] == ["-I", "/cuda/include"]


def test_explicit_rocm_wins_cuda_environment(tmp_path, monkeypatch, compiler):
    (tmp_path / "rccl.h").touch()
    monkeypatch.setattr(build, "_rccl_include_dirs", lambda *_: [tmp_path])
    monkeypatch.setenv("CUDA_HOME", "/cuda")
    build.build_probe(output=tmp_path / "probe.so", rocm_home="/rocm")
    assert "-DFLAGSCALE_TRACE_USE_RCCL=1" in compiler.call_args.args[0]


def test_explicit_rccl_requires_headers(tmp_path, monkeypatch, compiler):
    monkeypatch.setattr(build, "_rccl_include_dirs", lambda *_: [])
    with pytest.raises(RuntimeError, match="RCCL headers not found"):
        build.build_probe(output=tmp_path / "probe.so", backend="rccl")
    compiler.assert_not_called()


def test_discovers_dtk_and_hip_layout(tmp_path, monkeypatch, compiler):
    include = tmp_path / "include/rccl"
    hip_include = tmp_path / "hip/include"
    include.mkdir(parents=True)
    hip_include.mkdir(parents=True)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    includes = build._rccl_include_dirs(None, None)
    assert include in includes
    assert hip_include in includes
    assert len(includes) == len(set(includes))

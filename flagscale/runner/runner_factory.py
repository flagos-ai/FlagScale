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

from typing import TypeVar

from flagscale.runner.backend import (
    BackendBase,
    LlamaCppBackend,
    MegatronBackend,
    NativeCompressBackend,
    NativeServeBackend,
    NativeTrainBackend,
    SglangBackend,
    VerlBackend,
    VllmBackend,
)
from flagscale.runner.launcher import CloudLauncher, LauncherBase, SshLauncher

BackendType = TypeVar("BackendType", bound=BackendBase)
LauncherType = TypeVar("LauncherType", bound=LauncherBase)


class RunnerFactory:
    """Manage registration and retrieval of tasks, backends, and launchers."""

    _backend_registry: dict[str, type[BackendBase]] = {}
    _launcher_registry: dict[str, type[LauncherBase]] = {}

    @classmethod
    def register_backend(cls, name: str, backend_cls: type[BackendType]) -> None:
        if name in cls._backend_registry:
            raise ValueError(f"Backend '{name}' is already registered")
        cls._backend_registry[name] = backend_cls

    @classmethod
    def get_backend(cls, name: str) -> type[BackendType]:
        try:
            return cls._backend_registry[name]  # type: ignore[return-value]
        except KeyError:
            raise ValueError(f"Unknown backend type: {name!r}")

    @classmethod
    def register_launcher(cls, name: str, launcher_cls: type[LauncherType]) -> None:
        if name in cls._launcher_registry:
            raise ValueError(f"Launcher '{name}' is already registered")
        cls._launcher_registry[name] = launcher_cls

    @classmethod
    def get_launcher(cls, name: str) -> type[LauncherType]:
        try:
            return cls._launcher_registry[name]  # type: ignore[return-value]
        except KeyError:
            raise ValueError(f"Unknown launcher type: {name!r}")


# backends
RunnerFactory.register_backend("megatron", MegatronBackend)
RunnerFactory.register_backend("vllm", VllmBackend)
RunnerFactory.register_backend("sglang", SglangBackend)
RunnerFactory.register_backend("llama_cpp", LlamaCppBackend)
RunnerFactory.register_backend("verl", VerlBackend)
RunnerFactory.register_backend("native_compress", NativeCompressBackend)
RunnerFactory.register_backend("native_serve", NativeServeBackend)
RunnerFactory.register_backend("native_train", NativeTrainBackend)

# launchers
RunnerFactory.register_launcher("ssh", SshLauncher)
RunnerFactory.register_launcher("cloud", CloudLauncher)

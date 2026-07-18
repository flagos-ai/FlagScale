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

from flagscale.runner.auto_tuner import AutoTunerBase, ServeAutoTuner, TrainAutoTuner

AutoTunerType = TypeVar("AutoTunerType", bound=AutoTunerBase)


class AutotunerFactory:
    """Manage registration and retrieval of tasks, backends, and launchers."""

    _autotuner_registry: dict[str, type[AutoTunerBase]] = {}

    @classmethod
    def register_autotuner(cls, name: str, autotuner_cls: type[AutoTunerType]) -> None:
        if name in cls._autotuner_registry:
            raise ValueError(f"AutoTuner '{name}' is already registered")
        cls._autotuner_registry[name] = autotuner_cls

    @classmethod
    def get_autotuner(cls, name: str) -> type[AutoTunerType]:
        try:
            return cls._autotuner_registry[name]  # type: ignore[return-value]
        except KeyError:
            raise ValueError(f"Unknown autotuner type: {name!r}")


# tuners
AutotunerFactory.register_autotuner("train", TrainAutoTuner)
AutotunerFactory.register_autotuner("serve", ServeAutoTuner)

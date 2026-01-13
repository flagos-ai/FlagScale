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

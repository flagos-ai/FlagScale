from abc import ABC, abstractmethod
from enum import Enum

from omegaconf import DictConfig

from flagscale.runner.factory import RunnerFactory
from flagscale.runner.utils import parse_hostfile


class JobStatus(Enum):
    RUNNING = "Running"
    TRANSITIONAL = "Transitional (Stopping or Starting)"
    COMPLETED_OR_IDLE = "Completed or Not Started"


class Runner(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        hostfile = self.config.experiment.runner.get("hostfile", None)
        self.resources = parse_hostfile(hostfile) if hostfile else None
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type in [
            "train",
            "inference",
            "compress",
            "serve",
            "rl",
        ], f"Unsupported task type: {self.task_type}"
        self.backend_type = getattr(self.config.experiment.task, "backend", "custom")
        # TODO(cz): trans engine type into backend type
        assert self.backend_type in [
            "megatron",
            "torchrun",
            "vllm",
            "sglang",
            "llama_cpp",
            "custom",
        ], f"Unsupported backend type: {self.backend_type}"
        self.launcher_type = getattr(self.config.experiment.task, "backend_type", "ssh")
        assert self.launcher_type in [
            "ssh",
            "cloud",
        ], f"Unsupported launcher type: {self.launcher_type}"

        self.backend = RunnerFactory.get_backend(self.backend_type)(self.config)
        self.launcher = RunnerFactory.get_launcher(self.launcher_type)(self.config, self.backend)

    def run(self, *args, **kwargs):
        return self.launcher.run(*args, **kwargs)

    def stop(self, *args, **kwargs):
        """Optional method to override."""
        return self.launcher.stop(*args, **kwargs)


class RunnerBase(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def stop(self, *args, **kwargs):
        """Optional method to override."""
        pass

from abc import ABC, abstractmethod
from enum import Enum

from omegaconf import DictConfig

from flagscale.runner.runner_factory import RunnerFactory
from flagscale.runner.utils import parse_hostfile

TASK_TO_BACKEND_MAP = {
    "train": ["megatron", "torchrun"],
    "inference": ["vllm"],
    "compress": ["compress_native"],
    "serve": ["vllm", "sglang", "llama_cpp", "serve_native"],
    "rl": ["verl"],
}


class Runner(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        hostfile = self.config.experiment.runner.get("hostfile", None)
        self.resources = parse_hostfile(hostfile) if hostfile else None
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type in TASK_TO_BACKEND_MAP, f"Unsupported task type: {self.task_type}"

        backend_attr = getattr(self.config.experiment.task, "backend", None)
        if self.task_type == "serve":
            if backend_attr is None:
                backend_attr = self.config.serve[0]["engine"]

        # backend is required for train and inference
        if self.task_type in ("train", "inference", "rl"):
            assert backend_attr is not None, (
                f"backend_type is required for task_type='{self.task_type}'. "
                f"Allowed backends: {TASK_TO_BACKEND_MAP[self.task_type]}"
            )
            self.backend_type = backend_attr
        else:
            # backend is optional for compress / serve
            self.backend_type = backend_attr or f"{self.task_type}_native"

        # validate task_type and backend_type compatibility
        allowed_backends = TASK_TO_BACKEND_MAP[self.task_type]
        assert self.backend_type in allowed_backends, (
            f"Unsupported backend type '{self.backend_type}' for task_type='{self.task_type}'. "
            f"Allowed backends: {allowed_backends}"
        )

        self.backend = RunnerFactory.get_backend(self.backend_type)(self.config)
        self.launcher = RunnerFactory.get_launcher("ssh")(
            self.config, self.backend
        )  # TODO add cloud launcher_type

    def run(self, *args, **kwargs):
        return self.launcher.run(*args, **kwargs)

    def stop(self, *args, **kwargs):
        """Optional method to override."""
        return self.launcher.stop(*args, **kwargs)

from abc import ABC

from omegaconf import DictConfig

from flagscale.runner.runner_factory import RunnerFactory
from flagscale.runner.utils import parse_hostfile

# None --> native
# native --> native_{task_type} in inner Factory registry
TASK_TO_BACKEND_MAP = {
    "train": ["megatron"],  # TODO: add "pi0", "robotics"
    "inference": ["vllm"],
    "compress": ["native", None],
    "serve": ["vllm", "sglang", "llama_cpp", "native", None],
    "rl": ["verl"],
}


class Runner(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        hostfile = self.config.experiment.runner.get("hostfile", None)
        self.resources = parse_hostfile(hostfile) if hostfile else None
        self.task_type = getattr(self.config.experiment.task, "type", None)
        self.launcher_type = self.config.experiment.runner.get("type", "ssh")
        assert self.task_type in TASK_TO_BACKEND_MAP, f"Unsupported task type: {self.task_type}"

        backend_attr = getattr(self.config.experiment.task, "backend", None)
        if self.task_type == "serve":
            if self.launcher_type == "cloud":
                backend_attr = "vllm"  # do not support other backend
            elif backend_attr is None and not self.config.experiment.task.get("entrypoint", None):
                backend_attr = self.config.serve[0]["engine"]

        # backend is required for train / inference / rl
        if self.task_type in ("train", "inference", "rl"):
            assert backend_attr is not None, (
                f"backend_type is required for task_type='{self.task_type}'. "
                f"Allowed backends: {TASK_TO_BACKEND_MAP[self.task_type]}"
            )
            backend_type = backend_attr
        else:
            # compress / serve: backend optional
            backend_type = backend_attr or "native"

        # normalize native → native_{task_type}
        if backend_type == "native":
            backend_type = f"native_{self.task_type}"

        self.backend_type = backend_type

        # validate task_type and backend_type compatibility
        allowed_backends = TASK_TO_BACKEND_MAP[self.task_type]
        assert backend_attr in allowed_backends, (
            f"Unsupported backend type '{backend_attr}' for task_type='{self.task_type}'. "
            f"Allowed backends: {allowed_backends}"
        )

        self.backend = RunnerFactory.get_backend(self.backend_type)(self.config)
        self.launcher = RunnerFactory.get_launcher(self.launcher_type)(self.config, self.backend)

    def run(self, *args, **kwargs):
        return self.launcher.run(*args, **kwargs)

    def stop(self, *args, **kwargs):
        """Optional method to override."""
        return self.launcher.stop(*args, **kwargs)

    def query(self, *args, **kwargs):
        """Optional method to override."""
        return self.launcher.query(*args, **kwargs)

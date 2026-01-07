from abc import ABC, abstractmethod

from omegaconf import DictConfig

from flagscale.runner.utils import validate_serve_config


class BackendBase(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        # Validate serve configuration if task type is serve
        task_type = getattr(self.config.experiment.task, "type", None)
        if task_type == "serve":
            validate_serve_config(self.config)

    @abstractmethod
    def generate_run_script(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate_stop_script(self, *args, **kwargs):
        raise NotImplementedError

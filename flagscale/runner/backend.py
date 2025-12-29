from abc import ABC, abstractmethod

from omegaconf import DictConfig


class BackendBase(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def generate_run_script(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate_stop_script(self, *args, **kwargs):
        raise NotImplementedError

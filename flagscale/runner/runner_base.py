from abc import ABC, abstractmethod
from enum import Enum

from omegaconf import DictConfig


class JobStatus(Enum):
    RUNNING = "Running"
    TRANSITIONAL = "Transitional (Stopping or Starting)"
    COMPLETED_OR_IDLE = "Completed or Not Started"


class RunnerBase(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def stop(self, *args, **kwargs):
        """Optional method to override."""
        pass

from abc import ABC, abstractmethod


class LauncherBase(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def stop(self, *args, **kwargs):
        raise NotImplementedError

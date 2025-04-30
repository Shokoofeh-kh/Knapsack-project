from abc import ABC, abstractmethod


class Agent(ABC):
    """
    A Common Interface for any Agent of the Multi-Agent System.
    """
    @abstractmethod
    def act(self, state: str | list[dict] | dict) -> str:
        raise NotImplementedError('Act method must be implemented.')

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

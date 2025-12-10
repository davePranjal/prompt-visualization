from abc import ABC, abstractmethod
from typing import List

class Model:
    """A simple, standardized data structure for a model."""
    def __init__(self, name: str, provider: str):
        self.name = name
        self.provider = provider

    def __repr__(self) -> str:
        return f"Model(name='{self.name}', provider='{self.provider}')"

class LLMProvider(ABC):
    """Abstract base class for a generic LLM provider."""

    @abstractmethod
    def list_models(self) -> List[Model]:
        """Lists all generative models available from the provider."""
        pass
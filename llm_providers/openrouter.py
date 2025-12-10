import os
from openai import OpenAI
from typing import List
from .base import LLMProvider, Model

class OpenRouterProvider(LLMProvider):
    """Concrete implementation for the OpenRouter provider."""

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("'OPENROUTER_API_KEY' environment variable not set.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def list_models(self) -> List[Model]:
        """Lists all models available from OpenRouter."""
        models = []
        response = self.client.models.list()
        for m in response.data:
            models.append(Model(name=m.id, provider="OpenRouter"))
        return models
import os
from huggingface_hub import HfApi
from typing import List
from .base import LLMProvider, Model

class HuggingFaceProvider(LLMProvider):
    """Concrete implementation for the Hugging Face Hub provider."""

    def __init__(self):
        # The token is optional for listing public models but good practice
        token = os.getenv("HUGGINGFACE_API_KEY")
        self.client = HfApi(token=token)

    def list_models(self) -> List[Model]:
        """Lists text-generation models from Hugging Face Hub."""
        models = []
        # Filter for popular text generation models
        for m in self.client.list_models(filter="text-generation", sort="likes", direction=-1, limit=50):
            models.append(Model(name=m.id, provider="HuggingFace"))
        return models
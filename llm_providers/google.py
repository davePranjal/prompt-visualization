import os
import google.generativeai as genai
from typing import List
from .base import LLMProvider, Model

class GoogleProvider(LLMProvider):
    """Concrete implementation for the Google Generative AI provider."""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("'GOOGLE_API_KEY' environment variable not set.")

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            raise ConnectionError(f"Failed to configure Google AI client: {e}") from e

    def list_models(self) -> List[Model]:
        """Lists all generative models from Google."""
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(Model(name=m.name, provider="Google"))
        return models
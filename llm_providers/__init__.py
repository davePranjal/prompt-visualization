from .base import LLMProvider
from .google import GoogleProvider
from .openrouter import OpenRouterProvider
from .huggingface import HuggingFaceProvider

def get_provider(provider_name: str) -> LLMProvider:
    """
    Factory function to get an instance of a model provider.

    Args:
        provider_name: The name of the provider (e.g., 'google', 'openrouter').

    Returns:
        An instance of the corresponding LLMProvider class.
    """
    provider_name = provider_name.lower()
    if provider_name == "google":
        return GoogleProvider()
    elif provider_name == "openrouter":
        return OpenRouterProvider()
    elif provider_name == "huggingface":
        return HuggingFaceProvider()
    else:
        raise ValueError(f"Unknown provider: '{provider_name}'. Supported providers are 'google', 'openrouter', 'huggingface'.")
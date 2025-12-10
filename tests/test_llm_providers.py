import pytest
from unittest.mock import patch, MagicMock
import os
from llm_providers import get_provider
from llm_providers.google import GoogleProvider
from llm_providers.openrouter import OpenRouterProvider
from llm_providers.huggingface import HuggingFaceProvider

# --- Factory Tests ---
def test_get_provider_factory():
    # Mock environment to prevent init errors during factory creation
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "key", "OPENROUTER_API_KEY": "key", "HUGGINGFACE_API_KEY": "key"}):
        with patch("llm_providers.google.genai"), \
             patch("llm_providers.openrouter.OpenAI"), \
             patch("llm_providers.huggingface.HfApi"):
            
            assert isinstance(get_provider("google"), GoogleProvider)
            assert isinstance(get_provider("openrouter"), OpenRouterProvider)
            assert isinstance(get_provider("huggingface"), HuggingFaceProvider)

def test_get_provider_invalid():
    with pytest.raises(ValueError):
        get_provider("invalid_provider")

# --- Google Provider Tests ---
@patch("llm_providers.google.genai")
def test_google_provider_init_and_list(mock_genai):
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}):
        # Mock list_models response
        mock_model = MagicMock()
        mock_model.name = "models/gemini-pro"
        mock_model.supported_generation_methods = ["generateContent"]
        mock_genai.list_models.return_value = [mock_model]
        
        provider = GoogleProvider()
        mock_genai.configure.assert_called_with(api_key="fake_key")
        
        models = provider.list_models()
        assert len(models) == 1
        assert models[0].name == "models/gemini-pro"
        assert models[0].provider == "Google"

def test_google_provider_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleProvider()

# --- OpenRouter Provider Tests ---
@patch("llm_providers.openrouter.OpenAI")
def test_openrouter_provider_list(mock_openai):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"}):
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "openai/gpt-4"
        mock_client.models.list.return_value.data = [mock_model]
        mock_openai.return_value = mock_client
        
        provider = OpenRouterProvider()
        models = provider.list_models()
        
        assert len(models) == 1
        assert models[0].name == "openai/gpt-4"
        assert models[0].provider == "OpenRouter"

# --- HuggingFace Provider Tests ---
@patch("llm_providers.huggingface.HfApi")
def test_huggingface_provider_list(mock_hf_api):
    with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "fake_key"}):
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "meta-llama/Llama-2-7b"
        mock_client.list_models.return_value = [mock_model]
        mock_hf_api.return_value = mock_client
        
        provider = HuggingFaceProvider()
        models = provider.list_models()
        
        assert len(models) == 1
        assert models[0].name == "meta-llama/Llama-2-7b"
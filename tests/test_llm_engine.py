import pytest
from unittest.mock import MagicMock, patch
from prompt_visualization.llm_engine import run_prompt_experiment, configure_genai

@patch("prompt_visualization.llm_engine.genai")
def test_configure_genai(mock_genai):
    configure_genai("test_key")
    mock_genai.configure.assert_called_once_with(api_key="test_key")

@patch("prompt_visualization.llm_engine.genai")
@patch("prompt_visualization.llm_engine.mlflow")
def test_run_prompt_experiment_success(mock_mlflow, mock_genai):
    # --- Setup Mocks ---
    # Mock MLflow run context
    mock_run = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_run.info.run_id = "test_run_id_123"
    
    # Mock GenAI Model and Response
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    mock_response = MagicMock()
    mock_response.text = '{"result": "success"}'
    # Mock usage metadata
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_response.usage_metadata.total_token_count = 30
    # Mock candidates
    mock_candidate = MagicMock()
    mock_candidate.finish_reason.name = "STOP"
    mock_response.candidates = [mock_candidate]
    # Mock safety ratings
    mock_response.prompt_feedback.safety_ratings = []
    
    mock_model.generate_content.return_value = mock_response
    
    # --- Execute ---
    result = run_prompt_experiment(
        raw_json_input='{"input": "test"}',
        system_prompt="System Prompt",
        run_name="test_run",
        model_name="gemini-pro"
    )
    
    # --- Assertions ---
    assert result["status"] == "Pass"
    assert result["output_text"] == '{"result": "success"}'
    assert result["run_id"] == "test_run_id_123"
    assert result["latency"] > 0
    
    mock_mlflow.log_param.assert_any_call("model_name", "gemini-pro")
    mock_mlflow.log_metric.assert_any_call("total_token_count", 30)
    mock_mlflow.log_dict.assert_called() # Should log the parsed JSON

@patch("prompt_visualization.llm_engine.genai")
@patch("prompt_visualization.llm_engine.mlflow")
def test_run_prompt_experiment_failure(mock_mlflow, mock_genai):
    # Mock GenAI to raise an exception
    mock_genai.GenerativeModel.side_effect = Exception("API Error")
    
    result = run_prompt_experiment("{}", "Prompt", "run", "model")
    
    assert result["status"] == "Fail"
    assert "API Error" in result["output_text"]
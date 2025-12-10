import mlflow
import google.generativeai as genai
import os
import time
import json

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5010")

def configure_genai(api_key):
    """Configures the generative AI model."""
    genai.configure(api_key=api_key)

def run_prompt_experiment(raw_json_input, system_prompt, run_name, model_name):
    """
    Runs a prompt experiment using a generative AI model and logs the results to MLflow.

    Args:
        raw_json_input (str): The raw JSON input for the prompt.
        system_prompt (str): The system prompt to guide the model's response.
        run_name (str): The name for the MLflow run.
        model_name (str): The name of the generative model to use.

    Returns:
        dict: A dictionary containing the status, output text, latency, and run_id.
    """
    message = f"{system_prompt}\n\n{raw_json_input}"
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        try:
            # Log inputs
            mlflow.log_param("model_name", model_name)
            mlflow.log_text(system_prompt, "system_prompt.txt")

            # Execute the model
            start_time = time.time()
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(message)
            end_time = time.time()
            latency = end_time - start_time

            # Log outputs
            mlflow.log_metric("latency", latency)
            
            # Attempt to log the response as both a text file and a JSON artifact
            try:
                # Clean the response text to make it valid JSON
                # This is a common requirement when dealing with LLM text outputs
                cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
                output_data = json.loads(cleaned_text)
                mlflow.log_dict(output_data, "output.json")
            except json.JSONDecodeError:
                # If it's not valid JSON, just log as text and move on
                mlflow.log_text(response.text, "output.txt")


            return {
                "status": "Pass",
                "output_text": response.text,
                "latency": latency,
                "run_id": run_id
            }
        except Exception as e:
            mlflow.log_param("error", str(e))
            return {
                "status": "Fail",
                "output_text": str(e),
                "latency": 0,
                "run_id": run_id
            }

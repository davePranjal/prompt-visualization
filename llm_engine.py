import mlflow
import google.generativeai as genai
import os
import time
import streamlit as st

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5010")

# Configure the generative AI model using Streamlit's secrets
# This is now the recommended way to handle secrets securely.
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except (KeyError, FileNotFoundError):
    # This error will be caught and displayed in the Streamlit app UI
    # if the secret is not set.
    pass

def run_prompt_experiment(raw_json_input, system_prompt, run_name):
    """
    Runs a prompt experiment using a generative AI model and logs the results to MLflow.

    Args:
        raw_json_input (str): The raw JSON input for the prompt.
        system_prompt (str): The system prompt to guide the model's response.
        run_name (str): The name for the MLflow run.

    Returns:
        dict: A dictionary containing the status, output text, and latency of the experiment.
    """
    message = f"{system_prompt}\n\n{raw_json_input}"
    
    with mlflow.start_run(run_name=run_name) as run:
        try:
            # Check if the API key was configured successfully
            if not genai.get_key():
                raise ValueError("Google API Key not found. Please set it in .streamlit/secrets.toml")

            # Log inputs
            mlflow.log_param("model_name", "gemini-1.0-pro")
            mlflow.log_text(system_prompt, "system_prompt.txt")

            # Execute the model
            start_time = time.time()
            model = genai.GenerativeModel('gemini-1.0-pro')
            response = model.generate_content(message)
            end_time = time.time()
            latency = end_time - start_time

            # Log outputs
            mlflow.log_metric("latency", latency)
            mlflow.log_text(response.text, "response.txt")
            
            return {
                "status": "Pass",
                "output_text": response.text,
                "latency": latency
            }
        except Exception as e:
            mlflow.log_param("error", str(e))
            return {
                "status": "Fail",
                "output_text": str(e),
                "latency": 0
            }

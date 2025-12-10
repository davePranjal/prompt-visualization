# LLM Prompt Consistency & Evaluation Tool

This project provides a local development environment to test, iterate, and log Large Language Model (LLM) prompts using Google Gemini, Streamlit, and MLflow.

The primary goal of this application is to run a given prompt against a raw JSON input multiple times. This allows developers to check for output consistency, measure performance metrics, and visualize the results in a simple dashboard, with all experiment data logged for review in MLflow.

## Features

- **Interactive UI**: A Streamlit-based web interface to control experiments.
- **MLflow Prompt Registry**: Fetch registered prompts from MLflow and register new ones directly from the UI.
- **Consistency Testing**: Run the same prompt multiple times to check for variations in the LLM's output.
- **Comprehensive Logging**: Automatically logs a wide range of metrics and parameters to MLflow, including:
  - **Performance**: Latency per run.
  - **Token Usage**: `prompt_token_count`, `candidates_token_count`, and `total_token_count`.
  - **Model Behavior**: The `finish_reason` (e.g., `STOP`, `MAX_TOKENS`).
  - **Safety**: Safety ratings for categories like Harassment and Hate Speech.
- **Visual Diffing**: A side-by-side comparison of outputs from any two runs in an experiment.
- **Secure Secret Management**: Uses Streamlit's built-in secrets management for API keys.
- **Reproducible Environments**: Leverages `uv` for fast and reliable dependency management.

## Prerequisites

- Python 3.9+
- `uv` (a fast Python package installer and resolver). You can install it with `pip install uv`.
- An active Google API Key with the Gemini API enabled.
- An MLflow server instance.

## Setup and Installation

This project uses `uv` for fast and reliable dependency management.

1.  **Clone the Repository**
    ```sh
    git clone <your-repository-url>
    cd prompt-visualization
    ```

2.  **Create and Activate Virtual Environment**
    `uv` will create a virtual environment in a `.venv` directory and activate it.
    ```sh
    uv venv
    source .venv/bin/activate
    # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    First, compile the complete dependency list from `pyproject.toml` into a `requirements.txt` lock file. This ensures your environment is reproducible.
    ```sh
    uv pip compile pyproject.toml -o requirements.txt
    ```
    Then, sync your virtual environment with the lock file.
    ```sh
    uv pip sync requirements.txt
    ```

## Configuration

This project uses Streamlit's secrets management to handle your API key and model configuration securely.

1.  **Create the Secrets Directory and File**
    In your project's root directory, create a `.streamlit` directory and a `secrets.toml` file within it.
    ```sh
    mkdir .streamlit
    touch .streamlit/secrets.toml
    ```

2.  **Add Your Configuration**
    Open `.streamlit/secrets.toml` and add your Google API key. You can also optionally specify a model name.
    ```toml
    # Required: Your Google API Key
    GOOGLE_API_KEY = "your_google_api_key"

    # Optional: The Gemini model to use. Defaults to "gemini-1.0-pro" if not set.
    # Example: MODEL_NAME = "gemini-1.5-flash"
    MODEL_NAME = "gemini-1.0-pro"
    ```

## Execution

The application requires two separate terminal sessions: one for the MLflow UI and one for the Streamlit app.

**Terminal 1: Launch the MLflow UI**

If you don't have a central MLflow server, you can run one locally. Make sure your virtual environment is activated. The UI will store all experiment data in a local `mlruns` directory.
```sh
mlflow ui --port 5010
```
The application is configured to connect to MLflow on port `5010`. You can view the UI at `http://localhost:5010`.

**Terminal 2: Launch the Streamlit App**

In a new terminal, ensure the virtual environment is activated.
```sh
streamlit run app.py
```
This will open the Streamlit application in your web browser (usually at `http://localhost:8501`).

## How to Use

1.  Open the Streamlit app in your browser.
2.  **Configure the Experiment**:
    - Use the sidebar to set the **Number of Runs** for the consistency check (max 100).
3.  **Manage Prompts**:
    - **Load a Prompt**: Select a registered prompt from the "Load Registered Prompt" dropdown and click "Load Prompt".
    - **Create a Prompt**: Write or paste a new system prompt in the "System Prompt" text area.
    - **Register a Prompt**: To save the current system prompt to MLflow, give it a name in the "New Prompt Name" field and click "Register Current Prompt".
4.  **Provide Input**: Paste your **Raw JSON Input** in the right-hand text area.
5.  **Run**: Click the **"Run Experiment"** button.
6.  **Review Results**:
    - A table will appear showing the status, latency, and an output preview for each run.
    - If you ran more than one test, a **Visual Diffing** section will allow you to compare the full text output of any two runs.
    - To review all logged metrics, parameters, and artifacts, navigate to the MLflow UI.

## Project Structure

```
.
├── .streamlit/
│   └── secrets.toml      # Local secrets for Streamlit (ignored by Git)
├── .github/              # GitHub-specific files (e.g., workflows)
├── llm_providers/        # Support for different LLM providers
├── prompt_visualization/ # Core application logic
│   ├── __init__.py
│   ├── consistency_evaluator.py # Calculates consistency metrics
│   └── llm_engine.py     # Backend logic for Gemini interaction & MLflow logging
├── .gitignore            # Files and directories ignored by Git
├── app.py                # Frontend UI (Streamlit)
├── pyproject.toml        # Project metadata and dependencies
├── README.md             # This file
└── requirements.txt      # Locked dependency file for reproducibility
```

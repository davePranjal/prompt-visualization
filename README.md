# LLM Prompt Consistency & Visualization Tool

This project provides a local development environment to test, iterate, and log Large Language Model (LLM) prompts using Google Gemini, Streamlit, and MLflow.

The primary goal of this application is to run a given prompt against a raw JSON input multiple times. This allows developers to check for output consistency, measure latency, and visualize the results in a simple dashboard, with all experiment data logged for review in MLflow.

## Features

- **Interactive UI**: A Streamlit-based web interface to control experiments.
- **Secure Secret Management**: Uses Streamlit's built-in secrets management for API keys.
- **Consistency Testing**: Run the same prompt multiple times to check for variations in the LLM's output.
- **Performance Logging**: Automatically logs execution time (latency) for each run.
- **Experiment Tracking**: Integrates with MLflow to log all parameters, artifacts (prompts and responses), and metrics.
- **Side-by-Side Comparison**: Displays results in a clear table for easy comparison of status, latency, and output previews.

## Prerequisites

- Python 3.9+
- `uv` (a fast Python package installer and resolver)
- An active Google API Key with the Gemini API enabled.
- An MLflow server instance (optional, can be run locally).

## Setup and Installation

This project uses `uv` for fast and reliable dependency management.

1.  **Clone the Repository**
    ```sh
    git clone <your-repository-url>
    cd prompt-visualization
    ```

2.  **Create Virtual Environment**
    `uv` will automatically create a virtual environment in a `.venv` directory.
    ```sh
    uv venv
    ```

3.  **Activate the Environment**
    - On macOS/Linux:
      ```sh
      source .venv/bin/activate
      ```
    - On Windows:
      ```sh
      .venv\Scripts\activate
      ```

4.  **Install Dependencies**
    First, compile the complete dependency list from `pyproject.toml` into a `requirements.txt` lock file. This ensures your environment is reproducible.
    ```sh
    uv pip compile pyproject.toml -o requirements.txt
    ```
    Then, sync your virtual environment with the lock file.
    ```sh
    uv pip sync requirements.txt
    ```

## Configuration

This project uses Streamlit's secrets management to handle your API key securely.

1.  **Create the Secrets Directory**
    In your project's root directory, create a new directory named `.streamlit`.
    ```sh
    mkdir .streamlit
    ```

2.  **Create the Secrets File**
    Create a file named `secrets.toml` inside the `.streamlit` directory.
    ```sh
    touch .streamlit/secrets.toml
    ```

3.  **Add Your API Key**
    Open `.streamlit/secrets.toml` and add your Google API key in the following format:
    ```toml
    GOOGLE_API_KEY = "your_google_api_key"
    ```
    This file is included in `.gitignore` and will not be committed to your repository.

## Execution

The application requires two separate terminal sessions: one for the MLflow UI and one for the Streamlit app.

**Terminal 1: Launch the MLflow UI**

If you don't have a central MLflow server, you can run one locally. Make sure your virtual environment is activated.
```sh
mlflow ui --port 5010
```
The application is configured to connect to MLflow on port `5010`. You can view the UI at `http://localhost:5010`.

**Terminal 2: Launch the Streamlit App**

In a new terminal, activate the virtual environment.
```sh
streamlit run app.py
```
This will open the Streamlit application in your web browser (usually at `http://localhost:8501`).

## How to Use

1.  Open the Streamlit app in your browser.
2.  Use the sidebar to select the **Number of Runs** for the experiment.
3.  Enter your **System Prompt** in the left text area, or import one from a past MLflow run using the sidebar.
4.  Paste your **Raw JSON Input** in the right text area.
5.  Click the **"Run Experiment"** button.
6.  The app will show a progress bar and then display a table with the results of each run.
7.  To review the full details, navigate to the MLflow UI.

## Project Structure

```
.
├── .streamlit/
│   └── secrets.toml      # Local secrets for Streamlit (ignored by Git)
├── .gitignore            # Files and directories ignored by Git
├── app.py                # Frontend UI (Streamlit)
├── llm_engine.py         # Backend logic (Gemini interaction & MLflow logging)
├── pyproject.toml        # Project metadata and dependencies
├── README.md             # This file
└── requirements.txt      # Locked dependency file for reproducibility
```

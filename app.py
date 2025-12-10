import streamlit as st
import pandas as pd
import mlflow
import time
import json
from llm_engine import configure_genai, run_prompt_experiment
from consistency_evaluator import calculate_consistency_metric

# --- Page Configuration and Initialization ---
st.set_page_config(layout="wide", page_title="LLM Prompt Evaluator")
st.title("LLM Prompt Consistency & Evaluation Tool")

# --- Cached Functions for Resource Management ---
@st.cache_resource
def get_mlflow_client():
    """Initializes and caches the MLflow client."""
    try:
        mlflow.set_tracking_uri("http://localhost:5010")
        mlflow_client = mlflow.tracking.MlflowClient()
        mlflow_client.search_experiments()
        return mlflow_client
    except Exception as e:
        st.warning(f"Could not connect to MLflow. Logging and evaluation are disabled. Error: {e}")
        return None

# --- Configuration ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    model_name = st.secrets.get("MODEL_NAME")
    configure_genai(api_key)
except (KeyError, FileNotFoundError):
    st.error("`GOOGLE_API_KEY` not found. Please create a `.streamlit/secrets.toml` file.")
    st.stop()

client = get_mlflow_client()

# Initialize session state
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = "You are an expert system that extracts structured data from recipes. Respond with only valid JSON."
if 'raw_json_input' not in st.session_state:
    st.session_state.raw_json_input = '{\n  "recipe_text": "A simple cake recipe with 2 cups of flour, 1 cup of sugar, and 3 eggs."\n}'


# --- Sidebar ---
with st.sidebar:
    st.header("Experiment Configuration")
    num_runs = st.slider("Number of Runs for Consistency Check", 1, 10, 3)
    st.info(f"Using model: `{model_name}`")
    
    if client:
        st.markdown("[View MLflow UI](http://localhost:5010)")
        st.divider()
        # The prompt import functionality can be added back here if desired

# --- Main Input Area ---
col1, col2 = st.columns(2)

with col1:
    st.header("System Prompt")
    st.text_area("System Prompt", height=300, key="system_prompt")

with col2:
    st.header("Raw JSON Input")
    st.text_area("Raw JSON Input", height=300, key="raw_json_input")

# --- Execution and Evaluation ---
if st.button("Run Experiment", type="primary"):
    if not st.session_state.system_prompt or not st.session_state.raw_json_input:
        st.error("Please provide both a system prompt and raw JSON input.")
    else:
        results = []
        run_ids = []
        
        # Get the current experiment ID to associate runs correctly
        # We'll assume a default experiment or create one if it doesn't exist
        experiment_name = "LLM_Consistency_Tests"
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = client.create_experiment(experiment_name)
        else:
            exp_id = exp.experiment_id

        progress_bar = st.progress(0)
        
        for i in range(num_runs):
            run_name = f"batch_{int(time.time())}_run_{i+1}"
            result = run_prompt_experiment(
                raw_json_input=st.session_state.raw_json_input,
                system_prompt=st.session_state.system_prompt,
                run_name=run_name,
                model_name=model_name
            )
            results.append(result)
            if result.get("run_id"):
                run_ids.append(result["run_id"])
            
            progress_bar.progress((i + 1) / num_runs)

        # --- Display Results Table ---
        st.header("Experiment Run Results")
        display_data = [{
            "Run": i + 1,
            "Status": r["status"],
            "Latency (s)": f"{r['latency']:.2f}",
            "Output Preview": r["output_text"][:200] + "..."
        } for i, r in enumerate(results)]
        st.table(pd.DataFrame(display_data))

        # --- Consistency Calculation and Display ---
        if client and len(run_ids) > 1:
            st.header("Consistency Evaluation")
            with st.spinner("Calculating consistency score..."):
                consistency_score = calculate_consistency_metric(exp_id, run_ids, artifact_path="output.json")
                
                # Log the score to the first run of the batch
                try:
                    client.log_metric(run_ids[0], "batch_consistency_score", consistency_score)
                    st.metric("Batch Consistency Score", f"{consistency_score:.4f}")
                    st.caption(f"Score logged to parent run: `{run_ids[0]}`")
                except Exception as e:
                    st.error(f"Failed to log consistency score to MLflow: {e}")
        elif len(run_ids) <= 1:
            st.info("At least 2 runs are required to calculate a consistency score.")

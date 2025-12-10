import streamlit as st
import pandas as pd
import mlflow
from llm_engine import run_prompt_experiment

# Page configuration
st.set_page_config(layout="wide")

# Initialize MLflow client
mlflow.set_tracking_uri("http://localhost:5010")
client = mlflow.tracking.MlflowClient()

# Initialize session state for the prompt. This runs only once.
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = ""

st.title("LLM Prompt Consistency Tester")

# --- Sidebar ---
with st.sidebar:
    st.header("Experiment Configuration")
    num_runs = st.slider("Number of Runs", 1, 10, 5)
    st.markdown("[View MLflow UI](http://localhost:5010)")

    st.divider()

    st.header("Import Prompt from MLflow")
    try:
        experiments = client.search_experiments()
        exp_names = [exp.name for exp in experiments]
        selected_exp_name = st.selectbox("Select Experiment", exp_names)

        selected_exp = next((exp for exp in experiments if exp.name == selected_exp_name), None)

        if selected_exp:
            # Sort runs by start time to show the most recent first
            runs = client.search_runs(experiment_ids=[selected_exp.experiment_id], order_by=["start_time DESC"])
            run_options = {f"{run.info.run_name} ({run.info.run_id[:8]})": run.info.run_id for run in runs}
            
            if run_options:
                selected_run_display = st.selectbox("Select Run", options=list(run_options.keys()))
                
                if st.button("Import Prompt"):
                    run_id = run_options[selected_run_display]
                    local_path = client.download_artifacts(run_id, "system_prompt.txt")
                    with open(local_path, "r") as f:
                        # This now correctly updates the session state, and Streamlit will
                        # automatically reflect this change in the text_area below.
                        st.session_state.system_prompt = f.read()
                    st.success("Prompt imported successfully!")
            else:
                st.info("No runs found in this experiment.")
    except Exception as e:
        st.error(f"Could not connect to MLflow: {e}")


# --- Main Input Area ---
col1, col2 = st.columns(2)

with col1:
    st.header("System Prompt")
    # By using a 'key', this text_area is directly linked to st.session_state.system_prompt.
    # Streamlit handles the two-way data binding automatically.
    st.text_area("System Prompt", height=300, key="system_prompt")

with col2:
    st.header("Raw JSON Input")
    raw_json_input = st.text_area("Enter the raw JSON input here", height=300)

# Execution
if st.button("Run Experiment"):
    # We now read the value directly from the session state.
    if not st.session_state.system_prompt or not raw_json_input:
        st.error("Please provide both a system prompt and raw JSON input.")
    else:
        results = []
        progress_bar = st.progress(0)
        
        for i in range(num_runs):
            run_name = f"run_{i+1}"
            result = run_prompt_experiment(raw_json_input, st.session_state.system_prompt, run_name)
            results.append({
                "Run": i + 1,
                "Status": result["status"],
                "Latency (s)": f"{result['latency']:.2f}",
                "Output Preview": result["output_text"][:200] + "..." if len(result["output_text"]) > 200 else result["output_text"]
            })
            progress_bar.progress((i + 1) / num_runs)

        # Display results
        st.header("Experiment Results")
        df = pd.DataFrame(results)
        st.table(df)

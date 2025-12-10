import streamlit as st
import pandas as pd
import mlflow
import time
import json
import difflib
from prompt_visualization.llm_engine import configure_genai, run_prompt_experiment
from prompt_visualization.consistency_evaluator import calculate_consistency_metric
import streamlit.components.v1 as components

# --- Page Configuration and Initialization ---
st.set_page_config(layout="wide", page_title="LLM Prompt Evaluator")
st.title("LLM Prompt Consistency & Evaluation Tool")

# --- Cached Functions for Resource Management ---
@st.cache_resource
def get_mlflow_client():
    """Initializes and caches the MLflow client."""
    try:
        mlflow.set_tracking_uri("http://localhost:5010")
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        return client
    except Exception as e:
        st.warning(f"Could not connect to MLflow. Logging and evaluation are disabled. Error: {e}")
        return None


@st.cache_data
def get_registered_prompts():
    """Fetches registered prompt names from MLflow."""
    try:
        if hasattr(mlflow, 'search_prompts'):
            prompts = mlflow.search_prompts()
            return [p.name for p in prompts]
        return []
    except Exception as e:
        st.error(f"Failed to fetch prompts from MLflow: {e}")
        return []


# --- Helper Functions ---
def get_clean_json(raw_text):
    """Cleans raw model output and attempts to parse it as JSON."""
    if not isinstance(raw_text, str):
        return None
    clean_text = raw_text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return None

def load_system_prompt_file():
    uploaded_file = st.session_state.get("prompt_uploader")
    if uploaded_file is not None:
        st.session_state.system_prompt = uploaded_file.getvalue().decode("utf-8")

def load_json_input_file():
    uploaded_file = st.session_state.get("json_uploader")
    if uploaded_file is not None:
        try:
            json_data = json.load(uploaded_file)
            st.session_state.raw_json_input = json.dumps(json_data, indent=2)
        except Exception as e:
            st.error(f"Invalid JSON file: {e}")


# --- Configuration ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    model_name = st.secrets.get("MODEL_NAME", "gemini-1.0-pro")
    configure_genai(api_key)
except (KeyError, FileNotFoundError):
    st.error("`GOOGLE_API_KEY` not found. Please create a `.streamlit/secrets.toml` file.")
    st.stop()

client = get_mlflow_client()

# --- Session State Initialization ---
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = "You are an expert system that extracts structured data from recipes. Respond with only valid JSON."
if 'raw_json_input' not in st.session_state:
    st.session_state.raw_json_input = '{\n  "recipe_text": "A simple cake recipe with 2 cups of flour, 1 cup of sugar, and 3 eggs."\n}'
if 'results' not in st.session_state:
    st.session_state.results = []


# --- Sidebar ---
with st.sidebar:
    st.header("Experiment Configuration")
    num_runs = st.number_input(
        "Number of Runs for Consistency Check",
        min_value=1,
        max_value=100,
        value=3,
        help="Max limit is 100",
        step=1,
    )
    st.info(f"Using model: `{model_name}`")

    has_prompt_registry = hasattr(mlflow, 'search_prompts')

    if client and has_prompt_registry:
        st.markdown("[View MLflow UI](http://localhost:5010)")
        st.divider()

        st.header("Prompt Management")
        registered_prompts = get_registered_prompts()

        if registered_prompts:
            selected_prompt = st.selectbox("Load Registered Prompt", options=registered_prompts,
                                           key="selected_mlflow_prompt")
            if st.button("Load Prompt"):
                try:
                    loaded_prompt = mlflow.load_prompt(selected_prompt)
                    if hasattr(loaded_prompt, 'template'):
                        st.session_state.system_prompt = loaded_prompt.template
                    else:
                        st.session_state.system_prompt = str(loaded_prompt)
                    st.success(f"Successfully loaded prompt: `{selected_prompt}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load prompt: {e}")
        else:
            st.info("No registered prompts found in MLflow.")

        st.divider()
        new_prompt_name = st.text_input("New Prompt Name", placeholder="e.g., recipe_parser_v1",
                                        key="new_prompt_name_input")
        if st.button("Register Current Prompt"):
            if new_prompt_name and st.session_state.system_prompt:
                try:
                    mlflow.register_prompt(name=new_prompt_name, template=st.session_state.system_prompt)
                    st.success(f"Prompt `{new_prompt_name}` registered successfully!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to register prompt: {e}")
            else:
                st.warning("Please provide a name and a system prompt to register.")
    elif client:
        st.warning("MLflow Prompt Registry features are not available in the installed MLflow version.")

# --- Main Input Area ---
col1, col2 = st.columns(2)
with col1:
    st.header("System Prompt")
    st.file_uploader("Upload Prompt (txt/md)", type=["txt", "md"], key="prompt_uploader",
                     on_change=load_system_prompt_file)
    st.text_area("System Prompt", height=400, key="system_prompt")

with col2:
    st.header("Raw JSON Input")
    st.file_uploader("Upload JSON Input", type=["json"], key="json_uploader", on_change=load_json_input_file)
    st.text_area("Raw JSON Input", height=400, key="raw_json_input")

# --- Execution and Evaluation ---
if st.button("Run Experiment", type="primary"):
    if not st.session_state.system_prompt or not st.session_state.raw_json_input:
        st.error("Please provide both a system prompt and raw JSON input.")
    else:
        results = []
        run_ids = []
        experiment_name = "LLM_Consistency_Tests"
        exp = client.get_experiment_by_name(experiment_name)
        exp_id = exp.experiment_id if exp else client.create_experiment(experiment_name)

        progress_bar = st.progress(0)
        for i in range(num_runs):
            run_name = f"batch_{int(time.time())}_run_{i + 1}"
            result = run_prompt_experiment(st.session_state.raw_json_input, st.session_state.system_prompt, run_name,
                                           model_name)
            results.append(result)
            if result.get("run_id"):
                run_ids.append(result["run_id"])
            progress_bar.progress((i + 1) / num_runs)
        st.session_state.results = results

# --- Display Results ---
if st.session_state.results:
    results = st.session_state.results
    st.header("Experiment Run Results")
    display_data = [{"Run": i + 1, "Status": r["status"], "Latency (s)": f"{r['latency']:.2f}",
                      "Output Preview": r["output_text"][:200] + "..."} for i, r in enumerate(results)]
    st.table(pd.DataFrame(display_data))

    run_ids = [r["run_id"] for r in results if "run_id" in r]
    if client and len(run_ids) > 1:
        st.header("Consistency Evaluation")
        exp_id = client.get_experiment_by_name("LLM_Consistency_Tests").experiment_id
        consistency_score = calculate_consistency_metric(exp_id, run_ids, artifact_path="output.json")
        client.log_metric(run_ids[0], "batch_consistency_score", consistency_score)
        st.metric("Batch Consistency Score", f"{consistency_score:.4f}")
        st.caption(f"Score logged to parent run: `{run_ids[0]}`")

    if len(results) > 1:
        st.header("Visual Diffing")
        st.write("Select two runs to compare their outputs.")
        run_options = {f"Run {i + 1} ({r['status']})": i for i, r in enumerate(results)}
        diff_col1, diff_col2 = st.columns(2)
        with diff_col1:
            run_a_idx = st.selectbox("Compare Run:", options=list(run_options.keys()), index=0)
        with diff_col2:
            run_b_idx = st.selectbox("With Run:", options=list(run_options.keys()), index=1)

        run_a = results[run_options[run_a_idx]]
        run_b = results[run_options[run_b_idx]]

        differ = difflib.HtmlDiff(wrapcolumn=80)
        diff_html = differ.make_table(run_a['output_text'].splitlines(), run_b['output_text'].splitlines(),
                                      fromdesc=f"Output of {run_a_idx}", todesc=f"Output of {run_b_idx}")
        components.html(diff_html, height=600, scrolling=True)

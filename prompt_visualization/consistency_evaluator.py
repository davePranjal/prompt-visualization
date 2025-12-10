import mlflow
import json
import difflib
from mlflow.tracking import MlflowClient

def normalize_ingredients(data):
    """
    Recursively extracts ingredient names from the ingredient_composition structure.
    Returns a sorted list of strings.
    """
    ingredients = []
    
    def extract(item):
        if isinstance(item, dict):
            # Check for 'name' key which usually holds the ingredient name
            if "name" in item:
                ingredients.append(str(item["name"]))
            # Recursively check all values
            for value in item.values():
                extract(value)
        elif isinstance(item, list):
            for sub_item in item:
                extract(sub_item)
        elif isinstance(item, str):
            # If the list is just strings, add them
            ingredients.append(item)

    extract(data)
    # Return unique, sorted, lowercase strings for consistent comparison
    return sorted(list(set([i.lower().strip() for i in ingredients])))

def calculate_consistency_metric(experiment_id, run_ids, artifact_path="output.json"):
    """
    Calculates the consistency metric for a batch of runs.
    
    Args:
        experiment_id (str): The ID of the experiment.
        run_ids (list): List of run IDs to evaluate.
        artifact_path (str): The path to the JSON output artifact.
        
    Returns:
        float: The average consistency score (0.0 to 1.0).
    """
    client = MlflowClient()
    normalized_lists = []

    for run_id in run_ids:
        try:
            local_path = client.download_artifacts(run_id, artifact_path)
            with open(local_path, "r") as f:
                content = json.load(f)
            
            # Extract ingredient_composition
            # We assume the root is a dict containing this key, or the list itself
            if isinstance(content, dict):
                ingredients_data = content.get("ingredient_composition", [])
            else:
                ingredients_data = content # Fallback if the root is the list
            
            norm_list = normalize_ingredients(ingredients_data)
            normalized_lists.append(norm_list)
        except Exception as e:
            print(f"Error processing run {run_id}: {e}")
            # We append an empty list to represent a failure to parse, 
            # which will naturally lower the consistency score against valid runs.
            normalized_lists.append([])

    if len(normalized_lists) < 2:
        return 0.0

    scores = []
    # Compare every pair of runs
    for i in range(len(normalized_lists)):
        for j in range(i + 1, len(normalized_lists)):
            # Convert lists to string representation for SequenceMatcher
            str1 = "\n".join(normalized_lists[i])
            str2 = "\n".join(normalized_lists[j])
            
            matcher = difflib.SequenceMatcher(None, str1, str2)
            scores.append(matcher.ratio())

    if not scores:
        return 0.0
        
    return sum(scores) / len(scores)

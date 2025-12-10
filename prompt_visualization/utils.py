import json

def get_clean_json(raw_text):
    """Cleans raw model output and attempts to parse it as JSON."""
    if not isinstance(raw_text, str):
        return None
    clean_text = raw_text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return None
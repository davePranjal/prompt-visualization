from prompt_visualization.utils import get_clean_json

def test_get_clean_json_valid():
    raw = '{"key": "value"}'
    assert get_clean_json(raw) == {"key": "value"}

def test_get_clean_json_markdown_formatting():
    raw = '```json\n{"key": "value"}\n```'
    assert get_clean_json(raw) == {"key": "value"}

def test_get_clean_json_invalid_json():
    raw = '{"key": "value"' # Missing closing brace
    assert get_clean_json(raw) is None

def test_get_clean_json_non_string():
    assert get_clean_json(None) is None
    assert get_clean_json(123) is None
    assert get_clean_json({}) is None
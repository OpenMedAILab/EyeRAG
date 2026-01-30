"""Utility functions for EyeRAG."""

import hashlib
import json
import os
import re

import pandas as pd

from config import LLM_RESPONSE_TEMPERATURE


def wrap_context(context):
    """Wrap context into a formatted string."""
    if isinstance(context, str):
        return context
    elif isinstance(context, list):
        if len(context) == 0:
            return ""
        if all(isinstance(item, str) for item in context):
            return "\n\n".join(context)
        if all(isinstance(item, dict) for item in context):
            formatted_context = "\n".join(
                f"---------\n"
                f"Context {k}: \nSource: {item.get('source', 'Unknown Source')}\nContent: {item['content']}"
                for k, item in enumerate(context) if isinstance(item, dict) and 'content' in item
            )
            return formatted_context
        else:
            raise ValueError("All items in the context list must be either strings or dictionaries with a 'content' key.")
    else:
        raise ValueError("Context must be a string or a list of strings.")


def get_catch_file_path(cache_dir, question, param_dict, question_id=''):
    """Generate cache file path based on question and parameters."""
    param_str = json.dumps(param_dict, sort_keys=True)
    query = question.replace('\n', ' ').replace('\r', ' ').replace(' ', '_').replace(
        '?', '').replace('#', '').replace(',', '_').replace('.', '_').replace(':', '_')
    if question_id:
        query = f"{question_id}_{query}"
    filename = query[:100] + '|' + generate_key(param_str) + ".json"
    cache_file_path = os.path.join(cache_dir, filename)
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
    assert cache_file_path
    return cache_file_path


def load_cache_file(use_cache_file, cache_file_path, key):
    """Load data from cache file if it exists."""
    if use_cache_file and os.path.isfile(cache_file_path):
        data = load_json_data(cache_file_path)
        assert f'{key}' in data, f"Invalid cache file: {cache_file_path}"
        print(f"Loaded {key} from cache file: {cache_file_path}")
        return data[key]
    else:
        return None


def save_cache_file(cache_file_path, data_dict_to_save: dict, param_dict=None):
    """Save data to cache file."""
    assert cache_file_path
    assert isinstance(data_dict_to_save, dict)
    if param_dict is not None:
        assert isinstance(param_dict, dict)
        data_dict_to_save.update(param_dict)
    save_dict_to_json(dict_to_save=data_dict_to_save, out_file=cache_file_path)
    print(f"Saved cache file: {cache_file_path}")


def generate_key(original_text):
    """Generate MD5 hash key from text."""
    chunk_key = hashlib.md5(original_text.encode('utf-8')).hexdigest()
    return chunk_key


def get_temperature_from_answer_type(answer_type):
    """Extract temperature from answer type string."""
    if answer_type.find("Temperature") > -1:
        temperature = float(answer_type.split("Temperature")[-1])
    else:
        temperature = LLM_RESPONSE_TEMPERATURE
    return temperature


def save_dict_to_json(dict_to_save, out_file):
    """Save dictionary to JSON file."""
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(dict_to_save, f, ensure_ascii=False, indent=4)


def load_json_data(filepath):
    """Load data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        data = {}
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")
        data = {}
    return data


def safe_json_parse(text):
    """Safely parse JSON, handling various format issues."""
    cleaned = text.strip()

    # Remove markdown markers
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Try direct parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract JSON object
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Fix common JSON format issues
    try:
        # Fix unquoted keys
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
        # Fix single quotes
        fixed = fixed.replace("'", '"')
        # Fix trailing comma
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    print(f"JSON parsing failed, original response: {text}")
    return {"choice": "A", "reason": "JSON parsing error, using default value"}


def load_text_from_file(file_path: str) -> str:
    """
    Load text from a specified file path.

    Args:
        file_path: The path to the text file.

    Returns:
        The content of the file, or an empty string if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""


def save_text_to_file(text, file_path):
    """Save text to file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def load_csv(file_path, delimiter=','):
    """Load data from CSV or Excel file."""
    if file_path.find(".csv") > -1:
        df = pd.read_csv(
            file_path, delimiter=delimiter,
            dtype={0: str}
        )
    else:
        df = pd.read_excel(
            file_path, engine='openpyxl',
            decimal=delimiter, dtype={0: str}
        )
    return df


def load_excel_with_sheet_name(file_path, delimiter=',', sheet_name=None):
    """Load data from Excel file with specified sheet name."""
    df = pd.read_excel(
        file_path, engine='openpyxl', sheet_name=sheet_name,
        decimal=delimiter, dtype={0: str}
    )
    return df


def get_json_file_name(question_id, llm_name, answer_type):
    """Generate JSON filename for question response."""
    return f"{question_id}_{llm_name}_{answer_type}.json"

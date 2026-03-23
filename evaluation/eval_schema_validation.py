
import pandas as pd
import re
import jsonref
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import copy
from typing import Dict, Tuple, Set, Any
import json
import os
from jsonschema import validators, exceptions
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, Any
from deepdiff import DeepDiff


EVAL_DIR = Path("/content/drive/MyDrive/json_generation/evaluation")

def remove_markdown_fencing(text: str) -> str:
    """
    Removes markdown code fences like ```json ... ``` or ``` ... ```
    from a string and returns raw JSON text.
    """

    # remove opening ```json or ```JSON or ```
    text = re.sub(r"^\s*```[a-zA-Z]*\s*", "", text)

    # remove closing ```
    text = re.sub(r"\s*```\s*$", "", text)

    return text.strip()

def load_json_safe(path):
    try:
        with open(path, "r") as f:
          raw_text = f.read()

          # NEW: remove markdown fencing
          raw_text = remove_markdown_fencing(raw_text)
          data = json.loads(raw_text)
        return data, True

    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path}: {e}")
        return None, False

def is_valid_json_schema(schema: dict) -> Tuple[bool, str]:
    """
    Validate whether a JSON object is a valid JSON Schema.
    Returns (is_valid, error_message).
    """

    try:
        # Detect correct validator from $schema if present
        ValidatorClass = validators.validator_for(schema)
        ValidatorClass.check_schema(schema)
        return True, ""

    except exceptions.SchemaError as e:
        return False, str(e)

    except Exception as e:
        return False, f"Unknown schema validation error: {e}"


def evaluate_difficulty(difficulty):
    invalid_json_count = 0
    invalid_schema_count = 0

    invalid_schema_files = []
    invalid_json_files = []

    for i in range(1, 11):
        print(f"==================================={i}===========================================")

        generated_path = f"{EVAL_DIR}/qwen-0.5b-coder-roleprompting/generated_baseline/{difficulty}/prompt_{i}_generated.json"

        generated_schema, valid = load_json_safe(
            generated_path
        )

        if not valid:
            invalid_json_count+=1
            invalid_schema_count+=1
            invalid_json_files.append(i)

        schema_valid, schema_error = is_valid_json_schema(generated_schema)

        if not schema_valid and i not in invalid_json_files:
            print(f"Invalid JSON Schema: {schema_error}")
            invalid_schema_count += 1
            invalid_schema_files.append(i)

    return (
    invalid_json_count,
    invalid_schema_count,
    invalid_json_files,
    invalid_schema_files,
)

if __name__ == "__main__":

    difficulties = ["ambiguous"]

    rows = []
    keyword_count_all_df = pd.DataFrame()

    for difficulty in difficulties:

        print(f"\nRunning evaluation for: {difficulty}")

        invalid_json_count, invalid_schema_count, invalid_json_files, invalid_schema_files = evaluate_difficulty(difficulty)

        rows.append({
            "difficulty": difficulty,
            "#json_invalid": invalid_json_count,
            "files_invalid_json": invalid_json_files,
            "#schema_invalid": invalid_schema_count,
            "files_invalid_schema": invalid_schema_files
        })

        # keyword_count_df = pd.DataFrame(keyword_count_df)
        # keyword_count_all_df = pd.concat([keyword_count_all_df, keyword_count_df], ignore_index=True)

    df_results = pd.DataFrame(rows)

    print("\nFinal Results:")
    print(df_results)

    # print("\nKeyword Count:")
    # keyword_count_all_df.to_excel(EVAL_DIR / "keyword_count_all.xlsx", index=False)
    # print(keyword_count_all_df)
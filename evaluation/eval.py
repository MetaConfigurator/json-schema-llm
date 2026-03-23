
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
from sentence_transformers import SentenceTransformer, util


embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
INVALID_SCHEMA_POLICY = "IN" #options : IG, IN
COMPARISON_TYPE = 'name' #options : name, all
ENABLE_JEDI_METRICS = True #options : True, False
EVAL_DIR = Path("/content/drive/MyDrive/json_generation/evaluation")
JSON_SCHEMA_KEYWORDS = {
    "$id", "$schema", "$ref", "$defs", "definitions",
    "type", "enum", "const",
    "properties", "patternProperties", "additionalProperties",
    "required", "items", "additionalItems", "contains",
    "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
    "dependentRequired", "dependentSchemas", "dependencies",
    "title", "description", "default", "examples",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf",
    "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties",
    "format",
    "contentEncoding", "contentMediaType",
    "readOnly", "writeOnly"
}

PROPERTY_CONTAINERS = {
    "properties",
    "patternProperties",
    "$defs",
    "definitions"
}

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

def compute_json_edit_distance(reference, predicted):
    """
    Compute JSON Edit Distance using structural diff.
    """

    diff = DeepDiff(
        reference,
        predicted,
        ignore_order=True,
        report_repetition=True
    )

    insert_ops = len(diff.get("dictionary_item_added", []))
    delete_ops = len(diff.get("dictionary_item_removed", []))
    replace_ops = len(diff.get("values_changed", []))
    type_ops = len(diff.get("type_changes", []))

    total_edits = insert_ops + delete_ops + replace_ops + type_ops

    return {
        "jed_insertions": insert_ops,
        "jed_deletions": delete_ops,
        "jed_replacements": replace_ops,
        "jed_type_changes": type_ops,
        "jed_total_edits": total_edits
    }

def compute_jed_score(reference, predicted):

    edits = compute_json_edit_distance(reference, predicted)

    ref_size = len(flatten_schema(reference, True))
    pred_size = len(flatten_schema(predicted, True))

    denom = ref_size + pred_size

    if denom == 0:
        similarity = 1.0
    else:
        similarity = 1 - (edits["jed_total_edits"] / denom)

    edits["jed_similarity"] = max(similarity, 0)

    return edits

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


#Resolve $ref
def resolve_refs(schema):
    """
    Resolve $ref in a schema eagerly.
    Raises ValueError if any $ref cannot be resolved.
    """
    # try:
    #     resolved = jsonref.JsonRef.replace_refs(schema, load_on_repr=False)

    #     # Force traversal so lazy refs are resolved now
    #     json.dumps(resolved)

    #     return resolved
    try:
        resolved = jsonref.JsonRef.replace_refs(schema)

        # Convert JsonRef proxies → pure dict/list
        resolved = json.loads(json.dumps(resolved, default=dict))
        return resolved

    except Exception as e:
        raise ValueError(f"Unresolvable $ref: {e}")

def count_schema_keywords(schema):

    keyword_counts = defaultdict(int)

    def traverse(obj, parent_key=None):

        if isinstance(obj, dict):

            for key, value in obj.items():

                # If parent is properties/definitions etc,
                # the key is a property name → NOT a schema keyword
                if parent_key not in PROPERTY_CONTAINERS:
                    if key in JSON_SCHEMA_KEYWORDS:
                        keyword_counts[key] += 1

                traverse(value, key)

        elif isinstance(obj, list):

            for item in obj:
                traverse(item, parent_key)

    traverse(schema)

    return dict(keyword_counts)

def apply_property_mapping(schema: Any, mapping: Dict[str, str]) -> Any:
    """
    Recursively apply property renaming across the entire schema.
    """

    if isinstance(schema, dict):
        new_schema = {}

        for key, value in schema.items():

            # 🔹 Case 1: properties → rename keys
            if key == "properties" and isinstance(value, dict):
                new_props = {}
                for prop_name, prop_schema in value.items():
                    new_name = mapping.get(prop_name, prop_name)
                    new_props[new_name] = apply_property_mapping(prop_schema, mapping)
                new_schema[key] = new_props

            # 🔹 Case 2: required → update list values
            elif key == "required" and isinstance(value, list):
                new_schema[key] = [mapping.get(v, v) for v in value]

            # 🔹 Case 3: dependentRequired
            elif key == "dependentRequired" and isinstance(value, dict):
                new_dep = {}
                for dep_key, dep_list in value.items():
                    new_key = mapping.get(dep_key, dep_key)
                    new_dep[new_key] = [mapping.get(v, v) for v in dep_list]
                new_schema[key] = new_dep

            # 🔹 Case 4: dependencies (older spec)
            elif key == "dependencies" and isinstance(value, dict):
                new_dep = {}
                for dep_key, dep_val in value.items():
                    new_key = mapping.get(dep_key, dep_key)

                    if isinstance(dep_val, list):
                        new_dep[new_key] = [mapping.get(v, v) for v in dep_val]
                    else:
                        new_dep[new_key] = apply_property_mapping(dep_val, mapping)

                new_schema[key] = new_dep

            # 🔹 Default: recurse
            else:
                new_schema[key] = apply_property_mapping(value, mapping)

        return new_schema

    elif isinstance(schema, list):
        return [apply_property_mapping(item, mapping) for item in schema]

    return schema

def semantic_normalize_schema(gen_schema, gt_schema, threshold=0.6):
    """Recursively align property names in generated schema to ground truth."""
    # Load embedding model (you can reuse the one from before)
    semantic_mapped_df = []

    gen_schema = copy.deepcopy(gen_schema)  # to avoid modifying original

    # Only compare if both are objects with properties
    if (
            isinstance(gen_schema, dict)
            and "properties" in gen_schema
            and isinstance(gt_schema, dict)
            and "properties" in gt_schema
    ):
        gen_props = list(gen_schema["properties"].keys())
        gt_props = list(gt_schema["properties"].keys())

        if gen_props and gt_props:
            # Encode all property names once
            gen_emb = embedding_model.encode(gen_props, convert_to_tensor=True)
            gt_emb = embedding_model.encode(gt_props, convert_to_tensor=True)
            sim_matrix = util.cos_sim(gen_emb, gt_emb)

            for i, g_key in enumerate(gen_props):
                j = sim_matrix[i].argmax().item()
                best_gt = gt_props[j]
                score = sim_matrix[i][j].item()
                # print(f"gen :'{g_key}' best_match_gt : '{best_gt}' (sim={score:.2f})")

                # Record the result
                semantic_mapped_df.append({
                    "Generated Property": g_key,
                    "Best Match in GT": best_gt,
                    "Similarity": round(score, 2),
                    "Renamed": "Yes" if score >= threshold and g_key != best_gt else "No"
                })

                mapping = {}
                used_targets = set()

                for i, g_key in enumerate(gen_props):
                    j = sim_matrix[i].argmax().item()
                    best_gt = gt_props[j]
                    score = sim_matrix[i][j].item()

                    semantic_mapped_df.append({
                        "Generated Property": g_key,
                        "Best Match in GT": best_gt,
                        "Similarity": round(score, 2),
                        "Renamed": "Yes" if score >= threshold and g_key != best_gt else "No"
                    })

                    # prevent collisions
                    if score >= threshold and g_key != best_gt and best_gt not in used_targets:
                        mapping[g_key] = best_gt
                        used_targets.add(best_gt)
                if mapping:
                    gen_schema = apply_property_mapping(gen_schema, mapping)

        # Recurse into each sub-property
        for prop_key, sub_schema in list(gen_schema["properties"].items()):
            if (
                    prop_key in gt_schema.get("properties", {})
                    and isinstance(sub_schema, dict)
            ):
                gen_schema["properties"][prop_key] = semantic_normalize_schema(
                    sub_schema, gt_schema["properties"][prop_key], threshold
                )

    return gen_schema


def load_json_schema(schema_path):
    """Load a JSON schema from a file."""
    with open(schema_path, 'r') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON schema from {schema_path}: {e}")
            return None


def compare_schemas(reference, predicted, results: list, comparison_type : str, compare_metadata_and_pattern: bool):

    if comparison_type == 'all' :
      flat1 = flatten_schema(reference, not compare_metadata_and_pattern)
      flat2 = flatten_schema(predicted, not compare_metadata_and_pattern)
      precision, recall, f1, tp, fp, fn, mismatches = compare_flattened_schemas(flat1, flat2, True)
    else :
      precision, recall, f1, tp, fp, fn, mismatches = compare_flattened_property_names(reference, predicted)


    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1)
    # print("True Positives:", tp)
    # print("False Positives:", fp)
    # print("False Negatives:", fn)
    # print("Value Mismatches:", mismatches)
    result = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "mismatches": mismatches,
    "tp": len(tp),
    "fp": len(fp),
    "fn": len(fn)
}

    # Optional JSON Edit Distance
    if ENABLE_JEDI_METRICS:
        jed_metrics = compute_jed_score(reference, predicted)
        result.update(jed_metrics)

    results.append(result)

def flatten_schema(schema: Any, ignore_metadata: bool, path: str = "") -> Dict[str, Any]:
    result = {}

    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in {"title", "description", "examples", "$id", "$schema"} and ignore_metadata:
                continue  # ignore metadata

            new_path = f"{path}.{key}" if path else key
            result.update(flatten_schema(value, ignore_metadata, new_path))

    elif isinstance(schema, list):
        normalized = [flatten_schema(item, ignore_metadata) if isinstance(item, (dict, list)) else item for item in schema]
        normalized.sort(key=lambda x: json.dumps(x, sort_keys=True))  # sort normalized list
        result[path] = normalized
    else:
        result[path] = schema

    #print(result)

    return result

def extract_property_names(schema: Dict[str, Any], parent_path: str = "") -> Set[str]:

    paths = set()

    if not isinstance(schema, dict):
        return paths

    properties = schema.get("properties", {})

    if isinstance(properties, dict):
        for prop, sub_schema in properties.items():

            current_path = f"{parent_path}.{prop}" if parent_path else prop
            paths.add(current_path)

            paths |= extract_property_names(sub_schema, current_path)

    return paths



def compare_flattened_property_names(
    reference: dict,
    predicted: dict,
) -> Tuple[float, float, float, Set[str], Set[str], Set[str], Set[str]]:
    """
    Compare property names only (ignores values, types, patterns, metadata).
    """
    ref_props = extract_property_names(reference)
    pred_props = extract_property_names(predicted)


    true_positives = ref_props & pred_props
    false_positives = pred_props - ref_props
    false_negatives = ref_props - pred_props
    value_mismatches = set()  # irrelevant for property-name-only comparison

    # print(true_positives)
    # print(false_positives)
    # print(false_negatives)

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1, true_positives, false_positives, false_negatives, value_mismatches


def compare_flattened_schemas(
    reference: Dict[str, Any],
    predicted: Dict[str, Any],
        compare_pattern: bool
) -> Tuple[float, float, float, Set[str], Set[str], Set[str], Set[str]]:
    ref_keys = set(reference.keys())
    pred_keys = set(predicted.keys())

    true_positives = set()
    value_mismatches = set()
    false_positives = pred_keys - ref_keys
    false_negatives = ref_keys - pred_keys

    for key in ref_keys & pred_keys:
        val1 = reference[key]
        val2 = predicted[key]
        # do not compare values for the '*.pattern' key
        if key.endswith('.pattern') and not compare_pattern:
            true_positives.add(key)
            continue
        if _equal_values(val1, val2):
            true_positives.add(key)
        else:
            value_mismatches.add(key)
    #print(true_positives)
    #print(false_positives)
    #print(false_negatives)

    # Count value mismatches as FP and FN
    tp = len(true_positives)
    fp = len(false_positives) + len(value_mismatches)
    fn = len(false_negatives) + len(value_mismatches)

    # print(f'TP : {true_positives}')
    # print(f'FP : {false_positives}')
    # print(f'FN : {false_negatives}')
    # print(f'Value Mismatch : {value_mismatches}')

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1, true_positives, false_positives, false_negatives, value_mismatches


def _equal_values(v1: Any, v2: Any) -> bool:
    if isinstance(v1, list) and isinstance(v2, list):
        norm1 = [normalize_value(i) for i in v1]
        norm2 = [normalize_value(i) for i in v2]
        norm1_sorted = sorted(norm1, key=lambda x: json.dumps(x, sort_keys=True))
        norm2_sorted = sorted(norm2, key=lambda x: json.dumps(x, sort_keys=True))
        return norm1_sorted == norm2_sorted
    return normalize_value(v1) == normalize_value(v2)

def normalize_value(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, set):
        return sorted([normalize_value(i) for i in v])
    if isinstance(v, list):
        return [normalize_value(i) for i in v]
    if isinstance(v, dict):
        return {k: normalize_value(v[k]) for k in sorted(v)}
    return v

def evaluate_difficulty(difficulty):

    TP_total = 0
    FP_total = 0
    FN_total = 0

    invalid_ref_count = 0
    invalid_json_count = 0
    invalid_schema_count = 0

    invalid_schema_files = []
    invalid_ref_files = []
    invalid_json_files = []

    keyword_count_df = []
    f1_scores = []
    jed_scores = []

    for i in range(1, 11):
        print(f"==================================={i}===========================================")

        with open(f"{EVAL_DIR}/ground_truth/{difficulty}/prompt_{i}_expected.json", "r") as f:
            gt_schema = json.load(f)

        keyword_count = count_schema_keywords(gt_schema)

        keyword_count_df.append({
        "difficulty": difficulty,
        "prompt": f"prompt_{i}",
        "keyword_count": keyword_count
        })

        generated_path = f"{EVAL_DIR}/generated/qcoder/generated_trained/{difficulty}/prompt_{i}_generated.json"

        generated_schema, valid = load_json_safe(
            generated_path
        )

        if not valid:
            invalid_json_count+=1
            invalid_schema_count+=1
            invalid_json_files.append(i)
            if INVALID_SCHEMA_POLICY == "IG":
                continue
            else:
                generated_schema = {}
            
        schema_valid, schema_error = is_valid_json_schema(generated_schema)

        if not schema_valid:
            print(f"Invalid JSON Schema: {schema_error}")
            invalid_schema_count += 1
            invalid_schema_files.append(i)

            # if INVALID_SCHEMA_POLICY == "IG":
            #     continue
            # else:
            #     generated_schema = {}

        gt_schema_ref_resolved = resolve_refs(gt_schema)

        try:
            #gt_schema_ref_resolved = resolve_refs(gt_schema)
            generated_schema_ref_resolved = resolve_refs(generated_schema)
        except ValueError as e:
            print(f"$ref resolution failed for sample {i} due to $ref error: {e}")
            invalid_ref_count += 1
            invalid_ref_files.append(i)
            if i not in invalid_schema_files:
                invalid_schema_count+=1
                invalid_schema_files.append(i)
            if INVALID_SCHEMA_POLICY == "IG":
                continue
            else:
                generated_schema_ref_resolved = {}

        semantic_normalized_gen_schema = semantic_normalize_schema(
            generated_schema_ref_resolved,
            gt_schema_ref_resolved,
            threshold=0.8
        )

        results = []

        if difficulty == 'partial' :
          compare_schemas(
            gt_schema_ref_resolved,
            semantic_normalized_gen_schema,
            results,
            'name',
            False
        )

        else :
          compare_schemas(
            gt_schema_ref_resolved,
            semantic_normalized_gen_schema,
            results,
            'all',
            False
        )


        result = results[0]
        f1_scores.append(result["f1_score"])
        TP_total += result["tp"]
        FP_total += result["fp"]
        FN_total += result["fn"]
        if ENABLE_JEDI_METRICS:
          jed_scores.append(result["jed_similarity"])

        print(f'{difficulty}-{i} : Precision = {result['precision']}, Recall = {result['recall']}, F1 = {result['f1_score']}')

    accuracy = TP_total / (TP_total + FP_total + FN_total) if TP_total + FP_total + FN_total else 0
    precision = TP_total / (TP_total + FP_total) if TP_total + FP_total else 0
    recall = TP_total / (TP_total + FN_total) if TP_total + FN_total else 0
    f1_micro = 2 * precision * recall / (precision + recall) if precision + recall else 0
    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    jed_macro = sum(jed_scores) / len(jed_scores) if jed_scores else None

    return (
    accuracy,
    precision,
    recall,
    f1_micro,
    f1_macro,
    jed_macro,
    invalid_json_count,
    invalid_schema_count,
    invalid_ref_count,
    invalid_json_files,
    invalid_schema_files,
    invalid_ref_files,
    keyword_count_df
)

if __name__ == "__main__":

    difficulties = ["simple", "nested","modification", "partial"]

    rows = []
    keyword_count_all_df = pd.DataFrame()

    for difficulty in difficulties:

        print(f"\nRunning evaluation for: {difficulty}")

        accuracy, precision, recall, f1_micro, f1_macro, jed_macro, invalid_json_count, invalid_schema_count, invalid_ref_count, invalid_json_files, invalid_schema_files, invalid_ref_files, keyword_count_df = evaluate_difficulty(difficulty)

        rows.append({
            "difficulty": difficulty,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "#json_invalid": invalid_json_count,
            "files_invalid_json": invalid_json_files,
            "#schema_invalid": invalid_schema_count,
            "files_invalid_schema": invalid_schema_files,
            "#ref_invalid": invalid_ref_count,
            "files_invalid_ref": invalid_ref_files,
            "jed_similarity": jed_macro if ENABLE_JEDI_METRICS else None
        })

        keyword_count_df = pd.DataFrame(keyword_count_df)
        keyword_count_all_df = pd.concat([keyword_count_all_df, keyword_count_df], ignore_index=True)

    df_results = pd.DataFrame(rows)

    print("\nFinal Results:")
    print(df_results)

    print("\nKeyword Count:")
    # keyword_count_all_df.to_excel(EVAL_DIR / "keyword_count_all.xlsx", index=False)
    print(keyword_count_all_df)
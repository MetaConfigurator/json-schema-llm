import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from natsort import natsorted

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCHEMA_DIR = "dataset/simple"

# ---------------------------------------------------------
# LOADER: robust JSON loader for ANY encoding
# ---------------------------------------------------------
def load_json_any_encoding(path):
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            raise RuntimeError(f"File is not valid JSON: {path}")
        except Exception:
            continue

    raise RuntimeError(f"Failed to read JSON file with any encoding: {path}")

# ---------------------------------------------------------
#  SIMPLE NATURAL LANGUAGE DESCRIPTION 
# ---------------------------------------------------------
def generate_simple_description(schema_content):
    prompt = f"""
You must respond ONLY with valid JSON.

Given the following JSON Schema:

{schema_content}

Generate:
1. A simple natural-language instruction describing what this JSON Schema is about.
   Keep it short, high-level, non-technical, and NOT detailed.

Return EXACTLY:
{{
  "description": "string"
}}
"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    text = response.output_text.strip()
    return json.loads(text)["description"]

# ---------------------------------------------------------
# DETAILED DESCRIPTION 
# ---------------------------------------------------------
def generate_detailed_description(schema_content):
    prompt = f"""
You must respond ONLY with valid JSON.

You are an expert at converting JSON Schemas into precise, faithful natural-language descriptions.
Accuracy is more important than verbosity. Do NOT invent structure that does not exist.

RULES (MANDATORY):

1. Start the description with:
   "The schema describes [main purpose]. It is an [type] type."

2. Definitions section (CONDITIONAL):
   - Include the sentence:
     "It defines several reusable components under 'definitions':"
     ONLY IF the schema contains "definitions" or "$defs".
   - If present, describe EACH definition:
     - "[DefinitionName]" is an [type], [description].
       Use the correct indefinite article ("a" or "an") based on the type.
       Mention required fields, enums, patterns, array rules,
       additionalProperties behavior, and validation constraints.
   - If no definitions or "$defs" exist, OMIT this section entirely.
   - Do NOT describe inline object properties as reusable components.
   
Root $ref handling (MANDATORY):
- If the root schema contains a "$ref" and does NOT define "properties":
  - Do NOT generate a "main properties" section.
  - Do NOT describe "$ref" as a property.
  - Explicitly state that the root schema resolves to the referenced definition.
  - Describe the referenced definition as the effective root schema.

3. Main properties section:
   - Add the line:
     "The main properties of the schema are:"
   - ONLY IF the root schema directly defines "properties".
   - For EACH root-level property:
     - "[propertyName]" is a [type] [description].
     - Mention:
       • required or optional status  
       • default values  
       • enums, patterns, min/max, oneOf/anyOf/allOf  
       • array item rules  
       • references to definitions (ONLY if "$ref" exists)
       
Enum handling (MANDATORY):
- If a property defines an "enum", list ALL allowed values explicitly.
- Do NOT summarize, abstract, or refer to enum values indirectly.
- Preserve the exact literal values as they appear in the schema.

4. Compositional constraints:
   - Explicitly describe schema-level constraints such as:
     oneOf, anyOf, allOf, not, dependencies, conditional schemas.
   - Explain what is required for each valid configuration.

5. additionalProperties:
   - End with:
     "No additional properties are allowed at the root level."
     ONLY IF additionalProperties is explicitly set to false at the root.

STRICT PROHIBITIONS:
- Do NOT mention "definitions" unless they exist.
- Do NOT hallucinate "$ref" usage.
- Do NOT duplicate information across sections.
- Do NOT describe optional fields as required.
- Do NOT infer intent beyond what is structurally present.

Return EXACTLY the following JSON structure and nothing else:
{{
  "description": "string"
}}

JSON Schema:
{schema_content}
"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    text = response.output_text.strip()
    return json.loads(text)["description"]


# ---------------------------------------------------------
#LIMITED DETAIL DESCRIPTION
# ---------------------------------------------------------
def generate_moderate_description(schema_content):
    prompt = f"""
You must respond ONLY with valid JSON.

Your task:
Generate a moderately detailed natural-language description of the following JSON Schema.

RULES FOR THIS LEVEL:
1. Provide a clear overview of what the schema represents (2–4 sentences).
3. If the schema contains "definitions":
   - List all definition names (ONLY the names; no details).
4. List all root-level properties:
   - ONLY the property names.
   - DO NOT describe constraints, types, enums, patterns, or required rules.
5. Keep the description structured, concise, and mid-level in detail.
6. Do NOT use any of the highly detailed format rules from the detailed generator.
7. Do NOT be too short like the simple generator.

Return EXACTLY:
{{
  "description": "string"
}}

JSON Schema:
{schema_content}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    text = response.output_text.strip()
    return json.loads(text)["description"]


# -------- MAIN SCRIPT --------
if __name__ == "__main__":

    # schema_files = ['o46406.json']
    schema_files = natsorted([
        f for f in os.listdir(SCHEMA_DIR)
        if f.endswith(".json")
    ])

    print(schema_files)

    if len(schema_files) < 0:
        raise RuntimeError("Expected at least 100 schemas in /easy folder.")

    ##Moderate
    for file in schema_files[0:]:
        print(file)
        path = os.path.join(SCHEMA_DIR, file)

        try:
            schema = load_json_any_encoding(path)
        except Exception as e:
            print(f"[ERROR] Could not load {file}: {e}")
            continue

        schema_str = json.dumps(schema, indent=2)

        try:
            description = generate_detailed_description(schema_str)
        except Exception as e:
            print(f"[ERROR] GPT failed for {file}: {e}")
            continue

        base = os.path.splitext(file)[0]  # removes .json safely
        txt_path = os.path.join(SCHEMA_DIR, f"{base}_detailed.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(description)

        print(f"[OK] Description saved → {txt_path}")

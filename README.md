# JSON-Schema-LLM

## Overview
This repository investigates fine-tuning small-scale LLMs for JSON Schema generation and modification tasks and evaluating their performance.

Fine-tuned models:
- `Qwen2.5-Coder-0.5B-Instruct`
- `Qwen2.5-0.5B-Instruct`
- `Qwen2-0.5B`
- `Llama-3.2-1B-Instruct`

## Dataset
Training Dataset :
   - Instruction–response pairs: natural language schema descriptions to JSON Schema outputs
   - Source schemas: JSONSchemaBench / JSONSchemaStore (easy & medium subsets)
   - Prompt generation: primarily GPT-4o-mini
Training dataset split into train and validation sets

- Evaluation Dataset:
  - simple schema descriptions (Ground Truth Schema : Jsonschemastore, Prompt : GPT-4o-mini )
  - nested schema descriptions (Ground Truth Schema : Jsonschemastore, Prompt : GPT-4o-mini )
  - schema modification tasks (Ground Truth Schema : WashingtonPost, Prompt : Manual)
  - listing property names (Ground Truth Schema : GPT-4o, Prompt : GPT-4o-mini )
  - ambiguous or underspecified descriptions (Prompt : Manual)

Dataset Size:
- Total: 3897
- Train: 3508
- Validation: 389
- Test: 50

## How to Run
###Training & Inference
- Go to https://colab.research.google.com
- Select File → Open notebook → GitHub and paste the repository URL, or upload the .ipynb file directly.
- Runtime → Change runtime type → Hardware accelerator → T4 GPU
- Run all the cells

###Evaluation
- cd evaluation
- python eval.py

###Deployment
1. Model deployment 
  - cd deployment
  - model_deployment.py -> Replace the folder_path and repo_id with your paths
  - python model_deployment.py


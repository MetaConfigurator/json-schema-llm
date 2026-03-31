

# FastAPI version (commented out since we're using Gradio for simplicity)
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Enable CORS (important if frontend calls directly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "durga-lakshmi-2000/qwen-2.0"  # or local path if inside Space

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
)
model.eval()

class Request(BaseModel):
    description: str  # Only the user's description

def build_inputs(description: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON Schema generator.\n"
                "Return ONLY valid JSON.\n"
                "No markdown. No explanations.\n"
                "The response must start with { and end with }."
            ),
        },
        {
            "role": "user",
            "content": description,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

@app.post("/generate")
def generate(req: Request):
    # Build the full training prompt automatically
    inputs = build_inputs(req.description)

    with torch.no_grad():
        outputs = model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

    # Remove the prompt from the generated output
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    # Validate JSON to ensure the output is proper
    try:
        response_text = json.loads(generated_text)
        # response_text = json.dumps(json_obj, indent=2)
    except json.JSONDecodeError:
        # Fallback: return raw text if JSON is invalid
        return {"error": "Generated text is not valid JSON", "raw_response": generated_text}

    return {"response": response_text}
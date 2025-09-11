# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from .model import ModelWrapper
import os

app = FastAPI(title="Finetuned Unsloth Demo")

# Load your model
model = ModelWrapper(model_dir=os.environ.get("MODEL_DIR", "finetuned_unsloth"))

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h3>Unsloth finetuned model demo running. POST to /generate</h3>"

@app.post("/generate")
def generate(req: GenerateRequest):
    text = model.generate(req.prompt, max_new_tokens=req.max_new_tokens)
    return {"generated_text": text}

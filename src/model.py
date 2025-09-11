import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel


class ModelWrapper:
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.environ.get("MODEL_DIR", "finetuned_unsloth")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ModelWrapper] Trying to load model from {self.model_dir} on {self.device}")

        try:
            if self.device == "cuda":
                # 🚀 Preferred: Unsloth on GPU
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_dir,
                    max_seq_length=2048,
                    dtype=torch.float16,
                    load_in_4bit=True,
                )
                print("[ModelWrapper] Loaded with Unsloth on GPU")
            else:
                raise RuntimeError("CUDA not available")

        except Exception as e:
            print(f"[ModelWrapper] GPU load failed: {e}")
            print("[ModelWrapper] Falling back to Hugging Face AutoModel on CPU…")

            # 🐢 Fallback: Hugging Face on CPU
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            self.device = "cpu"

    def generate(self, prompt: str, max_new_tokens: int = 150, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import inferless
from pydantic import BaseModel, Field
from typing import Optional

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Give me a short introduction to large language model.")

@inferless.response
class ResponseObjects(BaseModel):
    generated_result: str = Field(default="Test output")
    thinking_hidden: str = Field(default="Test output")

class InferlessPythonModel:
    def initialize(self, context=None):
        self.model_id = "Qwen/Qwen3-8B"
        snapshot_download(repo_id=self.model_id,allow_patterns=["*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id,torch_dtype="auto",device_map="cuda")

    def _build_chat_prompt(self, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        
    def infer(self, request: RequestObjects) -> ResponseObjects:
        chat_prompt = self._build_chat_prompt(request.prompt)
        model_inputs = self.tokenizer([chat_prompt],return_tensors="pt").to(self.model.device)
        gen_ids = self.model.generate(**model_inputs,max_new_tokens=2048,temperature=0.7,do_sample=True)
        
        try:
            cut = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            cut = 0
        
        thinking_text = self.tokenizer.decode(output_ids[:cut],skip_special_tokens=True).strip()
        content_text = self.tokenizer.decode(output_ids[cut:],skip_special_tokens=True).strip()

        generateObject = ResponseObjects(generated_result=content_text,thinking_hidden=thinking_text)
        return generateObject

    def finalize(self):
        self.model = None

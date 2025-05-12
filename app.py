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
    temperature: Optional[float] = 0.7
    repetition_penalty: Optional[float] = 1.18
    max_new_tokens: Optional[int] = 2048
    

@inferless.response
class ResponseObjects(BaseModel):
    generated_result: str = Field(default="Test output")
    thinking_hidden: str = Field(default="Test output")

class InferlessPythonModel:
    def initialize(self, context=None):
        model_id = "Qwen/Qwen3-8B"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto",device_map="cuda")
        
    def infer(self, request: RequestObjects) -> ResponseObjects:
        messages = [
            {"role": "user", "content": request.prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs,temperature=request.temperature, max_new_tokens=request.max_new_tokens, repetition_penalty=request.repetition_penalty)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        generateObject = ResponseObjects(generated_result=content,thinking_hidden=thinking_content)
        return generateObject

    def finalize(self):
        self.model = None

import os

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InferlessPythonModel:
    def initialize(self, context=None):
        self.model_id = "Qwen/Qwen3-8B"

        # Pull only the files we really need into the Inferless volume
        snapshot_download(
            repo_id=self.model_id,
            allow_patterns=[
                "*.safetensors", "*.bin",               # weights
                "config.json", "generation_config.json", # config
                "tokenizer.*", "added_tokens.json"       # tokenizer
            ],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True
        )

        # “device_map='auto'” puts the model on the available GPU(s)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="cuda"
        )

    def _build_chat_prompt(self, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # keep the “thinking” channel
        )
    def infer(self, inputs):
        prompt = inputs.get("prompt", "")
        chat_prompt = self._build_chat_prompt(prompt)

        model_inputs = self.tokenizer(
            [chat_prompt],
            return_tensors="pt"
        ).to(self.model.device)

        gen_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True
        )
        try:
            cut = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            cut = 0  # no thinking section found

        thinking_text = self.tokenizer.decode(
            output_ids[:cut],
            skip_special_tokens=True
        ).strip()

        content_text = self.tokenizer.decode(
            output_ids[cut:],
            skip_special_tokens=True
        ).strip()

        return {
            "generated_result": content_text,
            # comment out the next line if you don’t want to expose the hidden reasoning
            "thinking_hidden": thinking_text
        }

    def finalize(self):
        self.model = None

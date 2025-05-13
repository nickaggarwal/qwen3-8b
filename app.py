import time

class InferlessPythonModel:
    def initialize(self):
        time.sleep(240)
        from vllm import LLM, SamplingParams
        try:
            self.model = LLM(model="Qwen/Qwen3-8B", trust_remote_code=True)
        except Exception as e:
            print(f"Model initialization error: {e}")
            raise

    def infer(self, inputs):
        prompt = inputs["prompt"]
        min_tokens = int(inputs.get("min_tokens", 0))
        max_tokens = int(inputs.get("max_tokens", 128))
        temperature = float(inputs.get("temperature", 1.0))
        top_p = float(inputs.get("top_p", 1.0))
        top_k = int(inputs.get("top_k", 50))
        repetition_penalty = float(inputs.get("repetition_penalty", 1.0))
        params = SamplingParams(
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        outputs = self.model.generate(prompt, params)

        result = prompt
        for output in outputs:
            result += output.outputs[0].text

        return {"generated_text": result}

    def finalize(self):
        self.model = None


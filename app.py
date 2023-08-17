import torch
from transformers import GPTJForCausalLM, GPT2Tokenizer


class InferlessPythonModel:
    def initialize(self):
        self.model = GPTJForCausalLM.from_pretrained(
            "PygmalionAI/pygmalion-6b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).cuda()
        self.tokenizer =  GPT2Tokenizer.from_pretrained(
            "PygmalionAI/pygmalion-6b"
        )

    def infer(self, inputs):
        prompt = inputs.get('prompt', None)

        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
        output = self.model.generate(input_tokens, max_new_tokens=1000, pad_token_id=self.tokenizer.eos_token_id)

        output_text = self.tokenizer.batch_decode(output, skip_special_tokens = True)[0]
        result = {"output": output_text}

        return result

    def finalize(self):
        self.model = None
        self.tokenizer = None


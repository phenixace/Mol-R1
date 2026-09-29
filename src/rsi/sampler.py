"""vLLM sampling backend for the harvest step."""

from src.common import chat_prompt


class VLLMSampler:
    """Draws n responses per description from a local checkpoint with vLLM.

    Prompts are built with the checkpoint's chat template (src.common.chat_prompt) and
    passed as token ids, so vLLM does not add a second BOS token.
    """

    def __init__(self, model_dir, temperature=0.6, top_p=0.9, max_tokens=4096,
                 seed=0, tensor_parallel_size=1, gpu_memory_utilization=0.9,
                 max_model_len=None):
        from vllm import LLM

        kwargs = dict(model=model_dir, tensor_parallel_size=tensor_parallel_size,
                      seed=seed, gpu_memory_utilization=gpu_memory_utilization)
        if max_model_len:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.params = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                           stop=["</answer>"], include_stop_str_in_output=True)

    def __call__(self, questions, n):
        from vllm import SamplingParams

        prompts = [{"prompt_token_ids": self.tokenizer(chat_prompt(self.tokenizer, q),
                                                       add_special_tokens=False)["input_ids"]}
                   for q in questions]
        outputs = self.llm.generate(prompts, SamplingParams(n=n, **self.params))
        return [[c.text for c in out.outputs] for out in outputs]

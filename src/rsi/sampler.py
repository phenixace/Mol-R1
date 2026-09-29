"""vLLM sampling backend for the harvest step."""


class VLLMSampler:
    """Draws n completions per prompt from a local checkpoint with vLLM."""

    def __init__(self, model_dir, temperature=1.0, top_p=1.0, max_tokens=4096,
                 seed=0, tensor_parallel_size=1, gpu_memory_utilization=0.9,
                 max_model_len=None):
        from vllm import LLM

        kwargs = dict(model=model_dir, tensor_parallel_size=tensor_parallel_size,
                      seed=seed, gpu_memory_utilization=gpu_memory_utilization)
        if max_model_len:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)
        self.params = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                           stop=["</answer>"], include_stop_str_in_output=True)

    def __call__(self, prompts, n):
        from vllm import SamplingParams

        outputs = self.llm.generate(prompts, SamplingParams(n=n, **self.params))
        return [[c.text for c in out.outputs] for out in outputs]

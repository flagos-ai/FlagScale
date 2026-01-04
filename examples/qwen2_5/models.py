from vllm import LLM, SamplingParams


class Model1Logic:
    def __init__(self):
        self.llm = LLM(
            model="/models/Qwen2.5-0.5B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )

    def forward(self, prompt: str) -> str:
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)
        result = self.llm.generate([prompt], sampling_params=sampling_params)
        return result[0].outputs[0].text + "__model1__"


class Model2Logic:
    def __init__(self):
        self.llm = LLM(
            model="/models/Qwen2.5-7B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )

    def forward(self, prompt: str) -> str:
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)
        result = self.llm.generate([prompt], sampling_params=sampling_params)
        return result[0].outputs[0].text + "__model2__"


class Model3Logic:
    def forward(self, arg1: str, arg2: str) -> str:
        res = arg1 + arg2 + "__model3__"
        return res

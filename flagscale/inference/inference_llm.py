# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config


def inference(cfg):
    """Initialize the LLMEngine"""
    # step 1: parse inference config
    prompts = cfg.generate.get("prompts", [])
    assert prompts, "Please set the prompts in the config yaml."

    # step 2: initialize the LLM engine
    llm_cfg = cfg.get("llm", {})
    llm = LLM(**llm_cfg)

    # tokenizer_cfg = llm_cfg.get("tokenizer", None)
    # if tokenizer_cfg:
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg, trust_remote_code=True)
    #     llm.set_tokenizer(tokenizer)

    # step 3: initialize the sampling parameters
    # TODO(zhaoyinglia): support config logits processor
    sampling_cfg = cfg.generate.get("sampling", {})
    assert not sampling_cfg.get("logits_processors", None), (
        "logits_processors is not supported yet."
    )
    sampling_params = SamplingParams(**sampling_cfg)
    print(f"=> {sampling_params=}")

    # step 4: build inputs
    inputs = [{"prompt": prompt} for prompt in prompts]
    print(f"=> {inputs=}")

    # step 5: generate outputs
    outputs = llm.generate(inputs, sampling_params)
    for output in outputs:
        print("*" * 50)
        print(f"{output.prompt=}")
        print(f"{output.outputs[0].text=}")
    print("#" * 50)


if __name__ == "__main__":
    cfg = parse_config()
    inference(cfg)

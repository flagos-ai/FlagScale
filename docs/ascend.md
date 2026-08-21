<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Ascend Guide

## Quick Start
### Serve with vllm
#### 1. Launch Container
```bash
ASCEND_IMAGE=harbor.baai.ac.cn/flagos-dev/flagscale:manual-20260812-ascend-dev-inference
docker pull "$ASCEND_IMAGE"
```

```bash
docker run \
    --name test-ascend \
    --network host \
    --ipc=host \
    --privileged \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    "$ASCEND_IMAGE" \
    bash
```

#### 2. Clone and Install

```bash
git clone https://github.com/flagos-ai/FlagScale.git
cd FlagScale
```

Install FlagScale without replacing the validated vLLM, vllm-plugin-FL, and
FlagGems versions provided by the image:

```bash
pip install . --no-build-isolation
```

#### 3. Start Serving

```bash
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export TRITON_ALL_BLOCKS_PARALLEL=1

vllm serve --model /models/Qwen3-4B --served-model-name qwen --enforce-eager
```

#### 4. Test the Service

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    max_tokens=1024,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    stream=True,
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Expected output:

```
Large language models (LLMs) are advanced artificial intelligence systems designed to
understand and generate human-like text. They are trained on vast amounts of textual data,
allowing them to perform a wide range of tasks such as answering questions, writing articles,
coding, and creative writing. These models are built using deep learning techniques and have
become a cornerstone of modern AI, enabling more natural and efficient interactions between
humans and machines.
```

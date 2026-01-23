## FlagOS-Robo Overview

🤖 FlagOS-Robo is built upon the unified and open-source AI system software stack, [FlagOS](https://flagos.io), which supports various AI chips.
It serves as an integrated training and inference framework for AI models used in robots🤖 , so-called Embodied Intelligence.
It can be deployed across diverse scenarios, ranging from edge to cloud.
Being portable across various chip models, it enables efficient training, inference, and deployment
for both Vision Language Models (VLMs) and Vision Language Action (VLA) models.
Here, VLMs usually act as the brain🧠 for task planning, while VLA models act as the cerebellum to output actions for robot control🦾.

FlagOS-Robo provides a powerful computational foundation and systematic support for cutting-edge researches
and industrial applications in embodied intelligence, accelerating innovations and real-world deployments
of intelligent agents.

## Feature Highlights

- [FlagScale](https://github.com/flagos-ai/FlagScale/tree/main) as users' entrypoint supports robot related AI model training and inference, including Pi-0, Pi-0.5, RoboBrain2, , RoboBrainX0. RoboBrain2.5 and RoboBrainX0.5 will be released soon.
- FlagOS-Robo supports [RoboOS](https://github.com/FlagOpen/RoboOS)-based cross-embodiment collaboration,
  ensuring compatibility with different data formats, efficient edge-cloud coordination,
  and real-machine evaluation.

## Quick Start🚀

| Models | Type | Checkpoint | Train | Inference | Serve | Evaluate |
|--------------|--------|--------|--------|-------------------|----------------------|---------------------------|
| PI0 | VLA | [Huggingface](https://huggingface.co/lerobot/pi0_base) | ✅︎  [Guide](../examples/pi0/README.md#training) | ✅︎  [Guide](../examples/pi0/README.md#inference) | ✅ [Guide](../examples/pi0/README.md#serving) | ❌ |
| PI0.5 | VLA | [Huggingface](https://huggingface.co/lerobot/pi05_libero_base) | ✅︎  [Guide](../examples/pi0_5/README.md#training) | ✅ [Guide](../examples/pi0_5/README.md#inference) | ✅   [Guide](../examples/pi0_5/README.md#serving)|  ❌ |
| RoboBrain-2.0 | VLM | [Huggingface](https://huggingface.co/BAAI/RoboBrain2.0-7B) | ✅︎  [Guide](../examples/qwen2_5_vl/README.md) | ✅[Guide](../examples/robobrain2/README.md#inference) | ✅[Guide](../examples/robobrain2/README.md#serving) | ✅   [Guide](../examples/qwen2_5_vl/README.md#evaluation) |
| RoboBrain-2.5 | VLM | ❌ | ✅︎  [Guide](../examples/qwen3_vl/README.md) | ✅[Guide](../examples/robobrain2_5/README.md#inference) | ✅[Guide](../examples/robobrain2_5/README.md#serving) | ✅   [Guide](../examples/qwen2_5_vl/README.md#evaluation) |
| RoboBrain-X0 | VLA | [Huggingface](https://huggingface.co/BAAI/RoboBrain-X0-Preview) | ✅︎  [Guide](../examples/robobrain_x0/README.md#training) | ❌ | ✅   [Guide](../examples/robobrain_x0/README.md#serving)| ❌ |

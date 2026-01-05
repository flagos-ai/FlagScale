## Change History

- **[2026/01]** Released [v1.0.0-alpha.0](https://github.com/flagos-ai/FlagScale/releases/tag/v1.0.0-alpha.0):

  - Major updates: Refactored the code base by moving hardware-specific (multi-chip) support into plugin repositories
    such as [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL) and
    [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL).
    These plugins build on top of [FlagOS](https://flagos.io/), a unified open-source AI system software stack.
    We also re-initialized the Git commit history to reduce repository size.

  - Compatibility caveats: If you are using or upgrading from a version earlier than `v1.0.0-alpha.0`,
    please use the [`main-legacy`](https://github.com/flagos-ai/FlagScale/tree/main-legacy) branch.
    It will continue to receive critical bug fixes and minor updates for a period of time.

- **[2025/09]** Released [v0.9.0](https://github.com/flagos-ai/FlagScale/tree/release/v0.9.0):

  - Training & fine-tuning: Added LoRA for efficient fine-tuning, improved the autotuner for cross-chip
    heterogeneous training, and enabled distributed RWKV training.

  - Inference & serving: Introduced DiffusionEngine for FLUX.1-dev, Qwen-Image, and Wan2.1-T2V,
    supporting multi-model automatic orchestration and dynamic scaling.

  - Embodied AI: Full lifecycle support for Robobrain, Robotics, and PI0, plus semantic retrieval
    for MCP-based skills for RoboOS.

  - Elasticity & fault tolerance: Detect task status automatically (errors, hangs, etc.) and
    periodically record them.

  - Hardware & system: Broader chip support, upgraded patch mechanism with file-level diffs,
    and enhanced CICD for different chips.

- **[2025/04]** Released [v0.8.0](https://github.com/FlagOpen/FlagScale/tree/release/v0.8.0):

  - Introduced a new flexible and robust multi-backend mechanism and updated vendor adaptation methods.

  - Enabled heterogeneous prefill-decoding disaggregation across vendor chips within a single instance
    via FlagCX (beta).

  - Upgraded DeepSeek-V3 pre-training with the new Megatron-LM and added heterogeneous pre-training
    across different chips for MoE models like DeepSeek-V3.

- **[2025/02]** Released [v0.6.5](https://github.com/FlagOpen/FlagScale/tree/release/v0.6.5):

  - Added support for DeepSeek-V3 distributed pre-training (beta) and
    [DeepSeek-V3/R1 serving](#deepseek-r1-serving) across multiple chips.

  - Introduced an auto-tuning feature for serving and a new CLI feature for one-click deployment.

  - Enhanced the CI/CD system to support more chips and integrated the workflow of
    [FlagRelease](https://huggingface.co/FlagRelease).

- **[2024/11]** Released [v0.6.0](https://github.com/flagos-ai/FlagScale/tree/release/v0.6.0):

  - Introduced general multi-dimensional heterogeneous parallelism and CPU-based communication
    between different chips.

  - Added the full support for LLaVA-OneVision, achieving SOTA results on the
    [Infinity-MM](https://arxiv.org/abs/2410.18558) dataset.

  - Open-sourced the optimized CFG implementation and accelerated the generation and
    understanding tasks for [Emu3](https://arxiv.org/abs/2409.18869).

  - Implemented the auto-tuning feature and enhanced the CI/CD system.

- **[2024/4]** Released [v0.3](https://github.com/flagos-ai/FlagScale/tree/release/v0.3):

  - Achieved heterogeneous hybrid training of the Aquila2-70B-Expr model on a cluster
    using both NVIDIA and Iluvatar chips. Adapted the Aquila2 series to AI chips
    from six different manufacturers.

- **[2023/11]** Released [v0.2](https://github.com/flagos-ai/FlagScale/tree/v0.2):

  - Introduced training support for Aquila2-70B-Expr, enabling heterogeneous training
    across chips with the same or compatible architectures.

- **[2023/10]** Released [v0.1](https://github.com/flagos-ai/FlagScale/tree/v0.1):

  - Supported Aquila models with optimized training schemes for Aquila2-7B and Aquila2-34B,
    including parallel strategies, optimizations, and hyper-parameter settings.

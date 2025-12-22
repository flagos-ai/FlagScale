## Setup

We recommend using the latest release of [NGC's PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
for setup.

The first step is to clone the repository:

```shell
git clone https://github.com/flagos-ai/FlagScale.git
```

The next step is to install the requirements.
You can install from the source or using python wheels. 

### Install from Source

```shell
PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation --verbose \
  --config-settings=device=<device> \
  --config-settings=backend=<backend>
```

<!--TODO(Qiming): check if the spelling of Megatron-LM can have variants.-->
where the `<device>` should be `gpu` for vLLM or Megatron-LM, and the `<backend>` should be
set to `vllm` for vLLM or `Megatron-LM` for Megatron-LM.
It is valid to specify multiple backends (separate by commas) when setting `<backend>`,
i.e. `--config-settings=backend=vllm,Megatron-LM`.


### Install from Wheel

For vLLM backend, the installation command is:

```shell
PYTHONPATH=./:$PYTHONPATH pip install .[vllm-gpu] --no-build-isolation --verbose
flagscale install --backend=vllm --device=gpu
```

For Megatron-LM backend, the installation command is:

```shell
PYTHONPATH=./:$PYTHONPATH pip install .[megatron-gpu] --no-build-isolation --verbose
flagscale install --backend=megatron --device=gpu
```

> [!IMPORTANT]
> The installation methods vary greatly in different chip environments.
> The installation methods above currently only support GPU.
> More backends and chips will be supported in the future.

<!--TODO(Qiming):
### Next Step
-->

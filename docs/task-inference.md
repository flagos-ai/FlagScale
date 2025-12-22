## Running an Inference Task

This tutorial demonstrates the steps to launch an inference task using vLLM.

1. Model preparation

   ```shell
   modelscope download --model BAAI/Aquila-7B README.md --local_dir ./
   ```

1. Edit the configuration

   The example inference configuration can be found at `examples/aquila/conf/inference/7b.yaml`.

   ```yaml
   llm:
       model: /workspace/models/BAAI/Aquila-7B         # modify path here
       tokenizer: /workspace/models/BAAI/Aquila-7B     # modify path here
       trust_remote_code: true
       tensor_parallel_size: 1
       pipeline_parallel_size: 1
       gpu_memory_utilization: 0.5
       seed: 1234
   ```

1. Start inference:

   ```shell
   python run.py --config-path ./examples/aquila/conf --config-name inference action=run
   ```

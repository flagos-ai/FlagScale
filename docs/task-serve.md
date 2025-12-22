## Running a Serving Task

1. Setup the environment

   ```shell
   PYTHONPATH=./:$PYTHONPATH pip install . --config-settings=domain=robotics --config-settings=device=gpu  --verbose --no-build-isolation
   ```

1. Download the tokenizer

   <!--TODO(Qiming): Confirm that this path should not be absolute ones.-->
   ```shell
   mkdir -p /models/physical-intelligence/
   cd /models/physical-intelligence/
   git lfs install
   git clone https://huggingface.co/physical-intelligence/fast
   ```

1. Edit configuration

   The configuration is located in `examples/robobrain_x0/conf/serve/robobrain_x0.yaml`.
   Modify the configuration as shown below:

   ```yaml
   - serve_id: vllm_model
     engine_args:
       model: /models/BAAI/RoboBrain-X0-Preview
       model_sub_task: /models/BAAI/RoboBrain-X0-Preview    # <--
       subtask_mode: true
       host: 0.0.0.0
       port: 5001                                           # <--
       debug: false
       threaded: true
       tokenizer_path: /models/physical-intelligence/fast   # <--
   ```

1. Start the server:

   ```shell
   python run.py --config-path ./examples/robobrain_x0/conf --config-name serve action=run
   ```

1. Stop the server:

   ```shell
   python run.py --config-path ./examples/robobrain_x0/conf --config-name serve action=stop
   ```

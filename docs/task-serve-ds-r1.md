## Running a DeepSeek-R1 Serving Task

We support the model serving of DeepSeek R1 and have implemented the `flagscale serve` command for one-click deployment.
By configuring just two YAML files, you can easily serve the model using the `flagscale serve` command.

1. Configure the YAML files:

   ```none
   FlagScale/
   ├── examples/
   │   └── deepseek_r1/
   │       └── conf/
   │           └── serve.yaml
   |           └── hostfile.txt # Set hostfile (optional)
   │           └── serve/
   │               └── 671b.yaml # Set model parameters and server port
   ```

   > [!NOTE]
   > When a task spans more than one nodes, a [hostfile.txt](./examples/deepseek/conf/hostfile.txt) is required.
   > The file path should be set in the `serve.yaml` configuration file.


1. Install the FlagScale CLI

   ```shell
   cd FlagScale
   PYTHONPATH=./:$PYTHONPATH pip install . --verbose --no-build-isolation
   ```

1. One-click serve:

   ```shell
   flagscale serve deepseek_r1
   ```

1. Customize the service parameters

   ```shell
   flagscale serve <MODEL_NAME> <MODEL_CONFIG_YAML>
   ```

The configuration files allow you to specify the necessary parameters and settings for your deployment,
ensuring a smooth and efficient serving process.

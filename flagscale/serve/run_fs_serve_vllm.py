import logging

from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

from flagscale import serve as flagscale_serve  # Keep original configuration loading logic

# -------------------------- 1. Load existing configuration file --------------------------
flagscale_serve.load_args()
TASK_CONFIG = flagscale_serve.task_config


# Parse target model configuration from the config file (corresponds to serve_id: serve_model in original YAML)
def get_model_config_from_task(serve_id: str = "serve_model"):
    if not TASK_CONFIG.get("serve"):
        raise ValueError("No 'serve' section found in task config.")
    for item in TASK_CONFIG.serve:
        if item.get("serve_id") == serve_id:
            return item
    raise ValueError(f"Model config with serve_id={serve_id} not found.")


model_config = get_model_config_from_task()
engine_args = model_config.get("engine_args", {})
resources = model_config.get("resources", {})

# Define predefined key list for subsequent filtering
para_list = [
    # filter para
    "model",
    "served_model_name",
    "port",
    # default para
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "enforce_eager",
    "enable_chunked_prefill",
    "enable_auto_tool_choice",
    "tool_call_parser",
    "chat_template",
]

# Initialize engine parameter dictionary with predefined options and default values
_engine_kwargs = {key: value for key, value in engine_args.items() if key in para_list[3:]}

# -------------------------- 2. Define Ray LLM configuration --------------------------
llm_config = LLMConfig(
    # Core configuration for model loading
    model_loading_config={
        "model_id": engine_args.get("served_model_name", engine_args["model"]),
        "model_source": engine_args["model"],
    },
    # Deployment resource configuration (corresponds to original resources field)
    deployment_config={
        "autoscaling_config": {
            "min_replicas": resources.get("num_replicas", 1),
            "max_replicas": resources.get("num_replicas", 1),
            "target_num_ongoing_requests_per_replica": 1000,
        }
    },
    # Engine-specific parameters (vLLM exclusive configuration, passed directly to vLLM engine)
    engine_kwargs=_engine_kwargs,
    # Runtime environment (add environment variables if needed)
    # This part is already handled by flagscale
    runtime_env={},
)

# -------------------------- 3. Start Ray Serve service --------------------------
if __name__ == "__main__":
    # Configure logging
    logger = logging.getLogger("ray.serve")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("serve.log"))

    # Add other key-value pairs from engine_args not included in predefined keys
    for key, value in engine_args.items():
        if key not in para_list:
            _engine_kwargs[key] = value
            logger.info("Adding untested engine arg: %s=%s", key, value)
    if engine_args.get("served_model_name", None):
        logger.info("Serving model name: %s", engine_args["served_model_name"])
    else:
        logger.warning(
            f"No served_model_name specified in engine_args, using {engine_args['model']}."
        )

    # Start Ray Serve and set HTTP port (read from original config, default 8000)
    deploy_port = TASK_CONFIG.experiment.get("runner", {}).get("deploy", {}).get("port", 8000)
    serve.start(http_options={"host": "0.0.0.0", "port": deploy_port})  # Allow external access

    # Build OpenAI-compatible app and run
    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(
        app,
        name="vllm_service",  # Service name (corresponds to original SERVICE_NAME)
        blocking=True,  # Block main thread to keep service running
    )

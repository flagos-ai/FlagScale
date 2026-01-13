import ast
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import threading
from typing import Any, Optional, Union, get_origin

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import ray
from fastapi import FastAPI
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, create_model
from ray import serve
from ray.serve.handle import DeploymentHandle

# from flagscale.logger import logger
logger = logging.getLogger("ray.serve")
logger.setLevel(logging.INFO)


def make_task_manager():
    local_app = FastAPI()

    class TaskUpdate(BaseModel):
        id: str
        status: str

    class TaskQuery(BaseModel):
        id: str

    @serve.deployment(num_replicas=1)
    @serve.ingress(local_app)
    class TaskManager:
        def __init__(self):
            self.task_status = {}
            self._lock = threading.Lock()

        @local_app.post("/set_task_status")
        async def set_task_status(self, req: TaskUpdate):
            with self._lock:
                self.task_status[req.id] = req.status
            logger.info(f"Task {req.id} status set to {req.status}")
            return {"ok": True}

        @local_app.post("/get_task_status")
        async def get_task_status(self, req: TaskQuery):
            with self._lock:
                status = self.task_status.get(req.id, "unknown")
            return {"id": req.id, "status": status}

        @local_app.post("/delete_task_status")
        async def delete_task_status(self, req: TaskQuery):
            with self._lock:
                self.task_status.pop(req.id, None)
            return {"ok": True}

    return TaskManager.bind()


def load_class_from_file(file_path: str, class_name: str):
    file_path = os.path.abspath(file_path)
    module_dir = os.path.dirname(file_path)
    logger.info(f"Loading class {class_name} from file: {file_path}")
    sys.path.insert(0, module_dir)
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Cannot create module spec from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, class_name):
            raise ImportError(f"Class {class_name} not found in {file_path}")
        return getattr(module, class_name)
    finally:
        if sys.path[0] == module_dir:
            sys.path.pop(0)


# Allowed root types for safety
_ALLOWED_BASES = {
    int,
    float,
    str,
    bool,
    list,
    dict,
    tuple,
    set,
    Any,
    list,
    dict,
    tuple,
    set,
    Union,
    Optional,
}


# Build all possible valid names for allowed types
_NAME_MAP = {
    n: t
    for t in _ALLOWED_BASES
    for n in (getattr(t, "__name__", None), getattr(t, "_name", None))
    if n
}


def _parse(node):
    if isinstance(node, ast.Name):
        try:
            return _NAME_MAP[node.id]
        except KeyError:
            raise ValueError(f"Unknown type: {node.id!r}")

    if isinstance(node, ast.Subscript):
        base = _parse(node.value)
        if base not in _ALLOWED_BASES:
            raise ValueError(f"Base type not allowed: {base}")
        s = node.slice
        args = tuple(_parse(e) for e in s.elts) if isinstance(s, ast.Tuple) else _parse(s)
        return base[args]

    if isinstance(node, ast.Tuple):
        return tuple(_parse(e) for e in node.elts)

    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def parse_type(type_str: str):
    if not isinstance(type_str, str) or not type_str.strip():
        raise ValueError(f"Invalid type string: {type_str!r}")

    try:
        tree = ast.parse(type_str, mode="eval").body
        # use ast.parse instead of eval(type_str), to avoid security issues
    except Exception:
        raise ValueError(f"Invalid type expression: {type_str!r}")

    t = _parse(tree)
    if (get_origin(t) or t) not in _ALLOWED_BASES:
        raise ValueError(f"Type not allowed: {type_str!r}")

    return t


def build_request_model(request_config):
    """
    Build a Pydantic model dynamically from config.
    New expected format:

        request:
          - arg: prompt
            type: str
          - arg: system_prompt
            type: Optional[str]
            required: false
            default: "..."

    """
    request_config = OmegaConf.to_container(request_config, resolve=True)
    logger.info(f"Building request model from config: {request_config}")

    if not isinstance(request_config, list):
        raise ValueError(
            f"request_config must be a list of argument definitions, but got {request_config} of type {type(request_config)}."
        )

    fields = {}

    for item in request_config:
        if "arg" not in item:
            raise ValueError(f"Missing 'arg' field in request item: {item}")
        if "type" not in item:
            raise ValueError(f"Missing 'type' field in request item: {item}")
        name = item["arg"]
        type_ = parse_type(item["type"])

        required = item.get("required", True)
        default = item.get("default", ...)

        # optional fields must have explicit default
        if not required and default is ...:
            raise ValueError(f"Default value for optional field '{name}' must be provided")

        fields[name] = (type_, default)

    return create_model("Request", **fields)


def make_deployment(logic_cls, **deploy_kwargs):
    @serve.deployment(**deploy_kwargs)
    class WrappedModel:
        def __init__(self):
            self.logic = logic_cls()

        async def forward(self, *args, **kwargs):
            if inspect.iscoroutinefunction(self.logic.forward):
                return await self.logic.forward(*args, **kwargs)
            return self.logic.forward(*args, **kwargs)

    return WrappedModel


@serve.deployment
class FinalModel:
    def __init__(
        self, graph_config: dict[str, Any], handles: dict[str, DeploymentHandle], config: DictConfig
    ):
        self.graph_config = graph_config
        self.handles = handles

        # determine return nodes
        all_nodes = set(graph_config.keys())
        dep_nodes = {dep for cfg in graph_config.values() for dep in cfg.get("depends", [])}
        self.roots = list(all_nodes - dep_nodes)
        assert len(self.roots) == 1, "Only one return node is allowed"
        request_config = config.experiment.runner.deploy.request
        self.request_base = build_request_model(request_config)

    async def __call__(self, http_request):
        origin_request = await http_request.json()
        request_data = self.request_base(**origin_request).dict()

        results_cache = {}

        async def run_node(node_name, **input_data):
            if node_name in results_cache:
                return results_cache[node_name]

            node_cfg = self.graph_config[node_name]
            handle = self.handles[node_name]

            if node_cfg.get("depends"):
                dep_results = []
                for dep in node_cfg["depends"]:
                    res = await run_node(dep, **input_data)
                    dep_results.append(res)
                if len(dep_results) == 1:
                    result = await handle.forward.remote(dep_results[0])
                else:
                    result = await handle.forward.remote(*dep_results)
            else:
                result = await handle.forward.remote(**input_data)

            results_cache[node_name] = result
            return result

        final_results = {}
        root = self.roots[0]
        final_results[root] = await run_node(root, **request_data)
        return final_results[root]


def build_graph(config):
    connection_list = config.serve
    # Convert list of dicts with 'serve_id' to dict keyed by serve_id
    connection = {
        cfg["serve_id"]: {k: v for k, v in cfg.items() if k != "serve_id"}
        for cfg in connection_list
    }

    handles = {}
    deployments = {}
    scale_config_items = [
        "min_replicas",
        "max_replicas",
        "downscale_delay_s",
        "upscale_delay_s",
        "target_ongoing_requests",
    ]
    for name, cfg in connection.items():
        logic_cls = load_class_from_file(cfg["module"], cfg["name"])
        resources = cfg.get("resources", {})
        ray_actor_options = {}
        if "num_gpus" in resources:
            ray_actor_options["num_gpus"] = resources["num_gpus"]
        deploy_kwargs = {
            "num_replicas": resources.get("num_replicas", 1),
            "ray_actor_options": ray_actor_options,
        }
        if "max_ongoing_requests" in resources:
            deploy_kwargs["max_ongoing_requests"] = resources["max_ongoing_requests"]
        scale_config = {}
        for item in scale_config_items:
            if item in resources:
                scale_config[item] = resources[item]
        if scale_config:
            deploy_kwargs["autoscaling_config"] = scale_config
            deploy_kwargs.pop("num_replicas")
            logger.info(f"autoscaling config {scale_config}")
        deployments[name] = make_deployment(logic_cls, **deploy_kwargs)
        handles[name] = deployments[name].bind()

    root_model = FinalModel.bind(connection, handles, config)
    return root_model


class ServeEngine:
    def __init__(self, config):
        self.config = config
        self.model_config = config.serve
        self.exp_config = config.experiment
        self.check_task(self.exp_config)
        self.init_task()

    def check_task(self, config):
        if not config.get("runner", {}).get("deploy", None):
            raise ValueError("key deploy is missing for deployment configuration.")
        self.check_dag()

    def check_dag(self, visibilization=True):
        # Ensure that all dependencies are valid
        dag = {}
        for model_alias, model_config in ((k, v) for d in self.model_config for k, v in d.items()):
            dependencies = []
            if "depends" in model_config:
                deps = model_config["depends"]
                if not isinstance(deps, (list, omegaconf.listconfig.ListConfig)):
                    deps = [deps]
                dependencies = deps
            dag[model_alias] = dependencies

            for dep in dependencies:
                if dep not in self.model_config["deploy"]["models"]:
                    raise ValueError(
                        f"Dependency {dep} for model {model_alias} not found in config['deploy']['models']"
                    )

        # Helper function to check for cycles using DFS
        def _is_cyclic(node, visited, stack):
            visited.add(node)
            stack.add(node)
            for neighbor in dag.get(node, []):
                if neighbor not in visited:
                    if _is_cyclic(neighbor, visited, stack):
                        return True
                elif neighbor in stack:
                    return True
            stack.remove(node)
            return False

        # Check for cycles
        visited = set()
        for node in dag:
            if node not in visited:
                if _is_cyclic(node, visited, set()):
                    raise ValueError(
                        "The graph contains cycles and is not a Directed Acyclic Graph (DAG)."
                    )

        def _visualize_dag_with_force_directed_layout(
            dag, file_name, iterations=100, k=1.0, t=1.0, cooling_factor=0.9
        ):
            nodes = list(dag.keys())
            n = len(nodes)

            # Initialize node positions
            positions = {node: np.random.rand(2) * 10 for node in nodes}

            for _ in range(iterations):
                # Calculate repulsive forces
                for i in range(n):
                    for j in range(i + 1, n):
                        node1, node2 = nodes[i], nodes[j]
                        delta = positions[node1] - positions[node2]
                        distance = np.linalg.norm(delta)
                        if distance > 1e-10:
                            f = (delta / distance) * (k**2 / distance)
                            positions[node1] += f
                            positions[node2] -= f

                # Calculate attractive forces
                for node, neighbors in dag.items():
                    for neighbor in neighbors:
                        delta = positions[node] - positions[neighbor]
                        distance = np.linalg.norm(delta)
                        if distance > 1e-10:
                            f = (delta / distance) * (distance / k)
                            positions[node] -= f
                            positions[neighbor] += f

                # Cool down
                t *= cooling_factor
                # Limit movement step
                for node in nodes:
                    move = np.random.randn(2) * t
                    positions[node] += move

            # Normalize positions
            all_positions = np.array([positions[node] for node in nodes])
            x_min, y_min = all_positions.min(axis=0)
            x_max, y_max = all_positions.max(axis=0)
            all_positions = (all_positions - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])
            for i, node in enumerate(nodes):
                positions[node] = all_positions[i]

            # Create figure
            plt.figure(figsize=(8, 6))

            # Draw edges
            for node, neighbors in dag.items():
                x, y = positions[node]
                for neighbor in neighbors:
                    nx, ny = positions[neighbor]
                    plt.arrow(
                        x,
                        y,
                        nx - x,
                        ny - y,
                        head_width=0.04,
                        head_length=0.08,
                        fc="gray",
                        ec="gray",
                        length_includes_head=True,
                        alpha=0.8,
                        zorder=5,
                    )

            # Draw nodes
            for node, (x, y) in positions.items():
                plt.scatter(x, y, s=800, color="lightblue", edgecolors="black", zorder=3)
                plt.text(x, y, node, fontsize=12, ha="center", va="center", zorder=4)

            # Add title
            plt.title("Directed Acyclic Graph (DAG)", fontsize=14)

            # Set aspect ratio
            plt.axis("equal")

            # Hide axes
            plt.axis("off")

            # Save figure
            plt.savefig(file_name)
            plt.close()

        # Optionally visualize the DAG
        if visibilization:
            dag_img_path = os.path.join(self.exp_config.exp_dir, "dag.png")
            _visualize_dag_with_force_directed_layout(dag, dag_img_path)

    def init_task(self, pythonpath=""):
        logger.info(f" =========== pythonpath {pythonpath} -----------------------")
        runtime_env = {}
        working_dir = (
            self.exp_config.runner.deploy.get("working_dir", "") or self.exp_config.exp_dir
        )
        if pythonpath:
            runtime_env["env_vars"] = {"PYTHONPATH": pythonpath}

        if working_dir:
            if working_dir not in sys.path:
                sys.path.append(working_dir)
            runtime_env["working_dir"] = working_dir
            runtime_env["excludes"] = [
                "*.log",
                "*.out",
                "*.output",
                "*.ckpt",
                "*.safetensors",
                "*.pth",
                "*.pt",
                "*.bin",
                "*.pyc",
                "**/.git/**",
                "**/__pycache__/**",
            ]
        if runtime_env:
            ray.init(runtime_env=runtime_env)
        else:
            ray.init()

    def run_task(self):
        graph = build_graph(self.config)
        port = self.exp_config.runner.deploy.get("port", 8000)
        task_manager = make_task_manager()
        serve.start(http_options={"host": "0.0.0.0", "port": port})
        manager_prefix_name = "/manager"
        serve_prefix_name = self.exp_config.runner.deploy.get("name", "/")
        assert manager_prefix_name != serve_prefix_name, (
            "router /manager exists, use another router name instead"
        )
        serve.run(
            task_manager, name="task_manager", route_prefix=manager_prefix_name, blocking=False
        )
        serve.run(
            graph, name=self.exp_config.exp_name, route_prefix=serve_prefix_name, blocking=True
        )

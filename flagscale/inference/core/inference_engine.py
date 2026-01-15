import dataclasses
import importlib
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import export_to_video
from omegaconf import DictConfig

from flagscale.inference.runtime_context import RuntimeContext
from flagscale.inference.utils import parse_torch_dtype
from flagscale.transformations import create_transformations_from_config


def _check_required_fields(config_dict: DictConfig, required_fields: list[str]) -> None:
    """Check if the required fields are in the config dict."""
    if not config_dict:
        raise ValueError("config_dict is empty")
    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Required field {field} is not in the config dict.")


@dataclass
class InferenceEngineArgs:
    """Args populated from YAML"""

    model: str = None
    loader: str = None
    device: Any = None
    torch_dtype: Any = None
    pipeline: Any = None
    components: Any = None
    tokenizer: str = None
    stat_path: str = None

    results_path: str = None
    output_format: str = None
    state_scopes: list[str] = None
    transformations: Any = None

    @classmethod
    def from_yaml(cls, config_dict: DictConfig) -> "InferenceEngineArgs":
        """Parse flat engine YAML into args using dataclass field introspection."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {name: config_dict.get(name) for name in field_names if name in config_dict}

        return cls(**kwargs)

    def __post_init__(self):
        missing = []
        if not self.model:
            missing.append("model")
        if not self.loader:
            missing.append("loader")
        if not self.results_path:
            missing.append("results_path")
        if not self.output_format:
            missing.append("output_format")
        if missing:
            raise ValueError(f"Missing required engine args: {', '.join(missing)}")

    def create_engine_config(self) -> "InferenceConfig":
        """Create the finalized engine config from args with validation."""
        model_obj = ModelLoadConfig(
            model=self.model,
            loader=self.loader,
            device=self.device,
            torch_dtype=self.torch_dtype,
            pipeline=self.pipeline,
            components=self.components,
            tokenizer=self.tokenizer,
            stat_path=self.stat_path,
        )
        engine_obj = EngineConfig(
            results_path=self.results_path,
            output_format=self.output_format,
            state_scopes=self.state_scopes,
            transformations=self.transformations if self.transformations is not None else {},
        )
        return InferenceConfig(model=model_obj, engine=engine_obj)


@dataclass
class ModelLoadConfig:
    model: str
    loader: str
    device: Any = None
    torch_dtype: Any = None
    pipeline: Any = None
    components: Any = None
    tokenizer: str = None
    stat_path: str = None

    def __post_init__(self):
        if not self.model:
            raise ValueError("'model' is required")
        if not self.loader:
            raise ValueError("'loader' is required")
        allowed_loaders = {"diffusers", "transformers", "custom", "auto"}
        if self.loader not in allowed_loaders:
            raise ValueError(f"Unsupported loader: {self.loader}. Allowed: {allowed_loaders}")


@dataclass
class EngineConfig:
    results_path: str
    output_format: str
    state_scopes: list[str] = None
    transformations: Any = None

    def __post_init__(self):
        if not self.results_path:
            raise ValueError("'results_path' is required")
        if not self.output_format:
            raise ValueError("'output_format' is required")
        allowed_formats = {"image", "video"}
        if self.output_format not in allowed_formats:
            raise ValueError(
                f"Unsupported output_format: {self.output_format}. Allowed: {allowed_formats}"
            )


@dataclass
class InferenceConfig:
    model: ModelLoadConfig
    engine: EngineConfig


class InferenceEngine:
    """A engine for model inference."""

    def __init__(self, config_dict: DictConfig) -> None:
        """Initialize the inference engine.

        Args:
            config_dict: The flat engine config dict. Required keys:
                - model (str)
                - loader (str): one of {"diffusers","transformers","custom","auto"}
                - results_path (str)
                - output_format (str): one of {"image","video"}
              Optional keys include device/torch_dtype/pipeline/components,
              transformations/transforms_cfg, state_scopes, etc.
        """
        self.args = InferenceEngineArgs.from_yaml(config_dict)
        self.vconfig = self.args.create_engine_config()

        self.model_or_pipeline, self.backbone = self.load()
        self.apply_transformations()

    def load(self) -> DiffusionPipeline | nn.Module:
        """Load the model or pipeline.

        Returns:
            The model or pipeline, depending on the loader.
        """

        loader = self.vconfig.model.loader
        _kwargs = {
            k: v
            for k, v in asdict(self.vconfig.model).items()
            if v is not None and k not in ("model", "loader")
        }

        if loader == "diffusers":
            return self.load_diffusers_pipeline(self.vconfig.model.model, **_kwargs)
        elif loader == "transformers":
            raise NotImplementedError("Transformers loader is not implemented")
        elif loader == "custom":
            raise NotImplementedError("Custom loader is not implemented")
        elif loader == "auto":
            raise NotImplementedError("Auto loader is not implemented")
        else:
            raise ValueError(f"Unsupported loader: {loader}")

    def apply_transformations(self) -> None:
        """Apply the transformations to the model or pipeline

        `Transformation` will be applied in the EXACT order as specified in the config.
        """

        # TODO(yupu): run preflight/supports check for each transformation
        transforms_cfg = self.vconfig.engine.transformations or {}
        transformations = create_transformations_from_config(transforms_cfg)
        for t in transformations:
            for name, mod in t.targets(self.backbone):
                success = t.apply(mod)
                if not success:
                    raise ValueError(f"Failed to apply transformation: {t} on {name}")

    def generate(self, **kwargs) -> Any:
        """Generate the output."""

        # Enforce return_dict=True for easier output saving
        kwargs["return_dict"] = True

        # Build a single torch.Generator from DictConfig: {seed: int, device?: str}
        def _build_generator(gen_cfg: DictConfig, default_device: str) -> torch.Generator:
            device = str(gen_cfg.get("device")) if gen_cfg.get("device") else default_device
            if gen_cfg.get("seed") is None:
                raise ValueError("generator.seed is required in config")
            seed = int(gen_cfg.get("seed"))
            return torch.Generator(device).manual_seed(seed)

        default_device = getattr(self.model_or_pipeline, "device", torch.device("cpu"))
        if isinstance(default_device, torch.device):
            default_device = str(default_device)
        if "generator" in kwargs and kwargs["generator"] is not None:
            kwargs["generator"] = _build_generator(kwargs["generator"], default_device)

        with RuntimeContext(self.vconfig.engine.state_scopes).session():
            outputs = self.model_or_pipeline(**kwargs)
            return outputs

    def save(self, outputs, name_prefix: str | None = None) -> bool:
        """Save the output.

        Args:
            outputs: The output object returned by the pipeline.
            name_prefix: Optional file name prefix to distinguish multiple runs.
        """

        os.makedirs(self.vconfig.engine.results_path, exist_ok=True)

        if self.vconfig.engine.output_format == "image":
            if hasattr(outputs, "images") and len(outputs.images) > 0:
                for i, image in enumerate(outputs.images):
                    fname = f"{name_prefix}_output_{i}.png" if name_prefix else f"output_{i}.png"
                    image.save(os.path.join(self.vconfig.engine.results_path, fname))
            else:
                raise NotImplementedError("Not implemented yet")
        elif self.vconfig.engine.output_format == "video":
            if hasattr(outputs, "frames") and len(outputs.frames) > 0:
                for i, frame in enumerate(outputs.frames):
                    fname = f"{name_prefix}_output_{i}.mp4" if name_prefix else f"output_{i}.mp4"
                    export_to_video(
                        frame, os.path.join(self.vconfig.engine.results_path, fname), fps=15
                    )
            else:
                raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError(f"Unsupported output format: {self.vconfig.engine.output_format}")

        return True

    def load_diffusers_pipeline(
        self, pretrained_model_name_or_path: str, **kwargs
    ) -> tuple[DiffusionPipeline, nn.Module]:
        """Load the DiffusionPipeline and locate the backbone.

        Args:
            pretrained_model_name_or_path (str, **required**): The name or path of the pretrained model.
            **kwargs: The kwargs for the `DiffusionPipeline.from_pretrained` method.
                kwargs could contain methods for enabling/disabling certain features of the pipeline.
                e.g. enable_xformers_memory_efficient_attention, enable_model_cpu_offload, etc.

        Returns:
            A tuple of the DiffusionPipeline and the backbone.
            - pipeline: The DiffusionPipeline.
            - backbone: The backbone of the DiffusionPipeline.
        """

        def _resolve_class(dotted_path: str):
            mod_name, _, cls_name = dotted_path.rpartition(".")
            if not mod_name or not cls_name:
                raise ValueError(f"Invalid class path: '{dotted_path}'")
            return getattr(importlib.import_module(mod_name), cls_name)

        def _normalize_dtype_inplace(d, key: str = "torch_dtype") -> None:
            if key in d:
                parsed = parse_torch_dtype(d.get(key))
                if parsed is not None:
                    d[key] = parsed

        def _kwargs_from_cfg(cfg, default_id: str) -> dict:
            assert isinstance(cfg, DictConfig) or cfg is None
            kwargs_local = {} if cfg is None else {k: cfg.get(k) for k in cfg}
            if "pretrained_model_name_or_path" not in kwargs_local:
                kwargs_local["pretrained_model_name_or_path"] = default_id
            _normalize_dtype_inplace(kwargs_local)
            return kwargs_local

        forbidden_keys = {"model", "loader", "device", "pipeline", "components"}
        sanitized_kwargs = {k: v for k, v in kwargs.items() if k not in forbidden_keys}
        _normalize_dtype_inplace(sanitized_kwargs)

        components_cfg = kwargs.get("components")
        if components_cfg:
            for comp_name, comp_spec in components_cfg.items():
                class_path = comp_spec.get("class")
                if not class_path or not isinstance(class_path, str):
                    raise ValueError(
                        f"Component '{comp_name}' requires a 'class' string (e.g., 'diffusers.AutoencoderKL')."
                    )
                cls = _resolve_class(class_path)
                fp_cfg = comp_spec.get("from_pretrained", {})
                fp_kwargs = _kwargs_from_cfg(fp_cfg, pretrained_model_name_or_path)
                sanitized_kwargs[comp_name] = cls.from_pretrained(**fp_kwargs)

        pipeline_cfg = kwargs.get("pipeline")
        pipeline_fp_args = {}
        if pipeline_cfg and pipeline_cfg.get("class"):
            pipeline_cls = _resolve_class(pipeline_cfg["class"])
            fp_cfg = pipeline_cfg.get("from_pretrained", {})
            pipeline_fp_args = _kwargs_from_cfg(fp_cfg, pretrained_model_name_or_path)
        else:
            pipeline_cls = DiffusionPipeline

        pipeline_kwargs = dict(sanitized_kwargs)
        if pipeline_fp_args:
            pipeline_kwargs.update(pipeline_fp_args)
        if "pretrained_model_name_or_path" in pipeline_kwargs:
            pipeline_kwargs.pop("pretrained_model_name_or_path")
        pipeline = pipeline_cls.from_pretrained(pretrained_model_name_or_path, **pipeline_kwargs)

        device = kwargs.get("device", None)
        if device:
            pipeline.to(device)

        # TODO(yupu): Messy, refactor this
        known_methods_wo_args = [
            "enable_xformers_memory_efficient_attention",
            "enable_model_cpu_offload",
            "enable_sequential_cpu_offload",
            "enable_attention_slicing",
            "enable_vae_slicing",
            "enable_vae_tiling",
            "fuse_qkv_projections",
        ]

        for method in known_methods_wo_args:
            if (
                method in kwargs
                and hasattr(pipeline, method)
                and callable(getattr(pipeline, method))
            ):
                getattr(pipeline, method)()

        # Simple heuristic to find the module to apply the transformations to
        if hasattr(pipeline, "unet"):
            backbone = pipeline.unet
            assert isinstance(backbone, nn.Module), (
                f"unet should be a `nn.Module`, but got {type(backbone)}"
            )
        elif hasattr(pipeline, "transformer"):
            backbone = pipeline.transformer
            assert isinstance(backbone, nn.Module), (
                f"transformer should be a `nn.Module`, but got {type(backbone)}"
            )
        else:
            raise ValueError(f"Failed to find the backbone of the DiffusionPipeline: {pipeline}")

        return pipeline, backbone

# Mainly adopted from
# https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/scripts/lerobot_train.py

import argparse
import os
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any, TypedDict

try:
    from typing import Unpack  # Python 3.11+
except ImportError:
    from typing_extensions import Unpack  # Python < 3.11

from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch
import torch.distributed as dist
from torch.optim import Optimizer
# from torch.nn.parallel import DistributedDataParallel as DDP  # Commented out: using accelerate instead

# Accelerate for distributed training (matching starVLA)
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed as accelerate_set_seed

from flagscale.runner.utils import logger
from flagscale.train.train_config import TrainConfig, DataConfig
from flagscale.train.datasets.transforms import ImageTransforms
from flagscale.train.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from flagscale.train.datasets.utils import dataset_to_policy_features
from flagscale.train.processor import PolicyAction, PolicyProcessorPipeline
from flagscale.train.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from flagscale.models.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from flagscale.models.configs.types import PolicyFeature
from flagscale.models.utils.constants import ACTION, OBS_PREFIX, REWARD
from flagscale.models.configs.types import FeatureType
from flagscale.train.utils.logging_utils import (
    AverageMeter,
    MetricsTracker,
    format_big_number,
)
from flagscale.train.utils.train_utils import (
    save_checkpoint,
    get_step_checkpoint_dir,
    update_last_checkpoint,
)
from flagscale.train.utils.optim_setup import setup_optimizer_and_scheduler
from flagscale.models.vla.qwen_gr00t import QwenGr00t
from flagscale.models.qwen_pi.qwen_pi import Qwen_PI

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

from PIL import Image
from torch.utils.data import Dataset as TorchDataset

def collate_fn_starvla(batch):
    """Simple collate function that returns batch as list of dicts (starVLA style)."""
    return batch


class StarVLAFormatDataset(TorchDataset):
    """
    Wrapper dataset that converts FlagScale tensor images to match starVLA format.

    Conversion to match starVLA exactly:
    1. FlagScale tensor: float32 CHW, [0,1] range
    2. Convert to uint8 HWC: multiply by 255, permute, cast to uint8
    3. PIL.fromarray + resize (same as starVLA)

    starVLA format:
        dict(
            action=np.ndarray [T, action_dim],  # float16
            image=[PIL.Image, ...],             # list of PIL images (224x224)
            lang=str,                           # language instruction
        )
    """

    def __init__(
        self,
        dataset: "LeRobotDataset",
        image_keys: list[str] = None,
        image_size: tuple[int, int] = (224, 224),
    ):
        self.dataset = dataset
        self.image_keys = image_keys or [
            "observation.images.image",
            "observation.images.wrist_image",
        ]
        self.image_size = image_size

        # Get action stats for min_max normalization (matching starVLA's StateActionTransform)
        action_stats = dataset.meta.stats.get("action", {})
        self.action_min = action_stats.get("min", None)
        self.action_max = action_stats.get("max", None)
        # Convert to numpy if needed
        if self.action_min is not None and hasattr(self.action_min, 'numpy'):
            self.action_min = self.action_min.numpy()
        if self.action_max is not None and hasattr(self.action_max, 'numpy'):
            self.action_max = self.action_max.numpy()

        # Debug: print stats
        print(f"[StarVLAFormatDataset] action_min: {self.action_min}")
        print(f"[StarVLAFormatDataset] action_max: {self.action_max}")
        self._debug_count = 0  # Counter for debug prints

    def __len__(self):
        return len(self.dataset)

    @property
    def num_frames(self):
        return self.dataset.num_frames

    @property
    def num_episodes(self):
        return self.dataset.num_episodes

    def _tensor_to_pil_starvla(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL exactly like starVLA:
        1. tensor is float32 CHW [0,1] from torchcodec
        2. Convert to uint8 HWC [0,255]
        3. PIL.fromarray + resize
        """
        # Remove batch dim if present
        if tensor.ndim == 4:
            tensor = tensor[0]

        # CHW -> HWC
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)

        # float32 [0,1] -> uint8 [0,255]
        img_np = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)

        # PIL.fromarray + resize (exactly like starVLA)
        pil_img = Image.fromarray(img_np).resize(self.image_size)
        return pil_img

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # Convert images to PIL format (matching starVLA processing)
        images = []
        for key in self.image_keys:
            if key in item:
                pil_img = self._tensor_to_pil_starvla(item[key])
                images.append(pil_img)

        # Get action (convert to numpy float16 like starVLA)
        action = item["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # Debug: print raw action values (only first few samples)
        if self._debug_count < 16:
            traj_id = item.get("episode_index", -1)
            if isinstance(traj_id, torch.Tensor):
                traj_id = traj_id.item()
            frame_idx = item.get("index", idx)
            if isinstance(frame_idx, torch.Tensor):
                frame_idx = frame_idx.item()
            print(f"[StarVLAFormatDataset] idx={idx} traj={traj_id} frame={frame_idx} RAW action[0,:5]: {action[0,:5].tolist()}")
            print(f"[StarVLAFormatDataset] idx={idx} RAW action sum: {action.sum():.4f}")

        # Apply min_max normalization (matching starVLA's Libero4in1DataConfig exactly)
        # starVLA only normalizes action.x, y, z, roll, pitch, yaw (indices 0-5)
        # action.gripper (index 6) is NOT normalized
        # Formula: 2 * (x - min) / (max - min) - 1
        if self.action_min is not None and self.action_max is not None:
            # Only normalize first 6 dimensions (x, y, z, roll, pitch, yaw)
            # Keep gripper (dim 6) as raw value
            normalize_dims = 6  # Only normalize first 6 dims
            action_range = self.action_max[:normalize_dims] - self.action_min[:normalize_dims]
            mask = action_range > 1e-8

            normalized = action.copy()
            # Normalize dimensions 0-5 where range > 0
            for i in range(normalize_dims):
                if mask[i]:
                    normalized[..., i] = (action[..., i] - self.action_min[i]) / action_range[i]
                    normalized[..., i] = 2.0 * normalized[..., i] - 1.0
                else:
                    normalized[..., i] = 0.0
            # Keep dimension 6 (gripper) as-is (no normalization)
            action = normalized

        # Debug: print normalized action values (only first few samples)
        if self._debug_count < 16:
            print(f"[StarVLAFormatDataset] idx={idx} NORM action[0,:5]: {action[0,:5].tolist()}")
            print(f"[StarVLAFormatDataset] idx={idx} NORM action sum: {action.sum():.4f}")
            self._debug_count += 1

        action = action.astype(np.float16)

        # Get language instruction
        lang = item.get("task", "")
        if isinstance(lang, torch.Tensor):
            lang = lang.item() if lang.numel() == 1 else str(lang.tolist())

        # Get trajectory_id and frame_index for debugging (matching starVLA format)
        trajectory_id = item.get("episode_index", -1)
        if isinstance(trajectory_id, torch.Tensor):
            trajectory_id = trajectory_id.item()
        frame_index = item.get("index", idx)
        if isinstance(frame_index, torch.Tensor):
            frame_index = frame_index.item()

        return dict(
            action=action,
            image=images,
            lang=lang,
            trajectory_id=trajectory_id,
            frame_index=frame_index,
        )

def register_debug_hooks(model_obj):
    """
    给模型挂载带有 Rank 信息的 Forward 和 Backward Hook
    model_obj: 可以是 model (list) 也可以是 model[0] (module)
    """
    # 1. 获取 Rank 的辅助函数
    def get_rank():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0
    # 2. 通用打印函数
    def calc_and_print(tensor, name, tag):
        """
        tensor: 要打印的张量
        name: 模块名称 + 参数位置
        tag: FWD 或 BWD
        """
        if tensor is None:
            return
        # 仅处理 Tensor，忽略 None 或其他类型
        if isinstance(tensor, torch.Tensor):
            # 获取当前 Rank
            rank = get_rank()
            # 计算 sum (转为 float32 防止溢出，item() 会触发同步确保数值准确)
            # 注意：打印日志会显著降低训练速度，仅用于 Debug
            val = torch.sum(tensor.detach().to(torch.float32)).item()
            # 打印格式：[Rank 0][FWD] layers.0.self_attention sum: 1234.56
            print(f"[Rank {rank}][{tag}] {name} sum: {val}", flush=True)
    # 3. 前向 Hook 定义
    def forward_wrapper(name):
        def forward_hook(module, input, output):
            # 打印 Input (元组或张量)
            if isinstance(input, (list, tuple)):
                for i, item in enumerate(input):
                    calc_and_print(item, f"{name}.input[{i}]", "FWD")
            else:
                calc_and_print(input, f"{name}.input", "FWD")
            # 打印 Output
            if isinstance(output, (list, tuple)):
                for i, item in enumerate(output):
                    calc_and_print(item, f"{name}.output[{i}]", "FWD")
            else:
                calc_and_print(output, f"{name}.output", "FWD")
        return forward_hook
    # 4. 反向 Hook 定义 (使用 register_full_backward_hook)
    def backward_wrapper(name):
        def backward_hook(module, grad_input, grad_output):
            # grad_output: 从上一层流回来的梯度 (反向传播的“输入”)
            if isinstance(grad_output, (list, tuple)):
                for i, g in enumerate(grad_output):
                    calc_and_print(g, f"{name}.grad_output[{i}]", "BWD")
            else:
                calc_and_print(grad_output, f"{name}.grad_output", "BWD")
            # grad_input: 当前层计算出的梯度 (准备传给下一层)
            if isinstance(grad_input, (list, tuple)):
                for i, g in enumerate(grad_input):
                    calc_and_print(g, f"{name}.grad_input[{i}]", "BWD")
            else:
                calc_and_print(grad_input, f"{name}.grad_input", "BWD")
        return backward_hook
    # 5. 开始注册
    # 兼容 list 结构
    actual_module = model_obj[0] if isinstance(model_obj, list) else model_obj
    print(f"Rank {get_rank()}: 开始挂载 Debug Hooks (仅叶子层)...", flush=True)
    # 遍历所有子模块
    for name, module in actual_module.named_modules():
        # 【核心修改】跳过容器层，只Hook叶子层（没有子模块的层）
        # 这样可以避免 Hook 顶层模块导致的 View 属性变化，同时也能覆盖所有计算
        if len(list(module.children())) > 0:
            continue
        # 额外的黑名单（可选）：跳过一些不重要的层，比如 Dropout
        if isinstance(module, torch.nn.Dropout):
            continue
        # 注册 FWD Hook
        handle_fwd = module.register_forward_hook(forward_wrapper(name))
        # 注册 BWD Hook
        handle_bwd = module.register_full_backward_hook(backward_wrapper(name))
def remove_debug_hooks_force(model_obj):
    """
    暴力清除模型中所有的 hook，不需要 handle。
    """
    actual_module = model_obj[0] if isinstance(model_obj, list) else model_obj
    print("Force removing all hooks...", flush=True)
    for module in actual_module.modules():
        # 清除前向 hook
        if hasattr(module, "_forward_hooks"):
            module._forward_hooks.clear()
        # 清除反向 hook
        if hasattr(module, "_backward_hooks"):
            module._backward_hooks.clear()
    print("Hooks force removed.", flush=True)




# Commented out: using accelerate's set_seed instead
# def set_seed(seed: int):
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cuda.matmul.allow_tf32 = True

def set_seed(seed: int):
    """Wrapper around accelerate's set_seed with additional cudnn settings."""
    accelerate_set_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False


# Commented out: using accelerate instead of manual DDP
# def init_ddp():
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     torch.distributed.init_process_group(backend="nccl", init_method="env://")
#     return local_rank

# Initialize Accelerator at module level (matching starVLA)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


# TODO: (yupu) Re-enable wandb
# def init_wandb(config, *, resuming: bool, log_code: bool = False, enabled: bool = True):
#     if not enabled:
#         wandb.init(mode="disabled")
#         return

#     ckpt_dir = pathlib.Path(config.checkpoint_dir)
#     if not ckpt_dir.exists():
#         raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
#     if resuming:
#         run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
#         wandb.init(id=run_id, resume="must", project=config.project_name)
#     else:
#         wandb.init(
#             name=config.exp_name, config=vars(config), project=config.project_name
#         )
#         (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

#     if log_code:
#         wandb.run.log_code(epath.Path(__file__).parent.parent)


def make_dataset(cfg: DataConfig):
    # TODO: (yupu) Support image transforms
    enable_image_transform = False
    # TODO: (yupu) Remove hard-coded video backend
    # After not much testing, It feels like that `torchcodec` is more robust than `pyav`
    # `pyav` crashes sometimes
    video_backend = "torchcodec"
    # video_backend = "torchvision_av"
    # video_backend = "pyav"

    # image_transforms = ImageTransforms(cfg.image_transforms) if enable_image_transform else None

    # Match starVLA: resize uint8 via PIL, then normalize to [0,1]
    def _resize_like_starvla(frames: torch.Tensor) -> torch.Tensor:
        if not isinstance(frames, torch.Tensor):
            return frames
        is_single = False
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
            is_single = True
        if frames.dim() != 4:
            return frames
        from PIL import Image
        import numpy as np

        resized_frames = []
        for frame in frames:
            channel_last = frame.shape[-1] in (1, 3, 4)
            if channel_last:
                frame_hwc = frame
            elif frame.shape[0] in (1, 3, 4):
                frame_hwc = frame.permute(1, 2, 0)
            else:
                frame_hwc = frame
                channel_last = True
            frame_uint8 = (frame_hwc * 255).round().clamp(0, 255).to(torch.uint8)
            pil = Image.fromarray(frame_uint8.cpu().numpy()).resize(
                (224, 224), resample=Image.BILINEAR
            )
            out = torch.from_numpy(np.array(pil)).to(frames.device).float() / 255.0
            if not channel_last:
                out = out.permute(2, 0, 1)
            resized_frames.append(out)
        output = torch.stack(resized_frames, dim=0)
        return output[0] if is_single else output

    image_transforms = _resize_like_starvla
    # Leave the revision to None
    ds_meta = LeRobotDatasetMetadata(root=cfg.data_path, revision=None)
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    dataset = LeRobotDataset(
        root=cfg.data_path,
        episodes=None,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=None,
        video_backend=video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def resolve_delta_timestamps(cfg: DataConfig, ds_meta: LeRobotDatasetMetadata) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg: The policy config (PI0Config or PI05Config) to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


# datasets/utils.py
def cycle(iterable: Any) -> Iterator[Any]:
    """Create a dataloader-safe cyclical iterator.

    This is an equivalent of `itertools.cycle` but is safe for use with
    PyTorch DataLoaders with multiple workers.
    See https://github.com/pytorch/pytorch/issues/23900 for details.

    Args:
        iterable: The iterable to cycle over.

    Yields:
        Items from the iterable, restarting from the beginning when exhausted.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def raise_feature_mismatch_error(
    provided_features: set[str],
    expected_features: set[str],
) -> None:
    """
    Raises a standardized ValueError for feature mismatches between dataset/environment and policy config.
    """
    missing = expected_features - provided_features
    extra = provided_features - expected_features
    # TODO (jadechoghari): provide a dynamic rename map suggestion to the user.
    raise ValueError(
        f"Feature mismatch between dataset/environment and policy config.\n"
        f"- Missing features: {sorted(missing) if missing else 'None'}\n"
        f"- Extra features: {sorted(extra) if extra else 'None'}\n\n"
        f"Please ensure your dataset and policy use consistent feature names.\n"
        f"If your dataset uses different observation keys (e.g., cameras named differently), "
        f"use the `--rename_map` argument, for example:\n"
        f'  --rename_map=\'{{"observation.images.left": "observation.images.camera1", '
        f'"observation.images.top": "observation.images.camera2"}}\''
    )


def format_train_tracker_step(train_tracker: MetricsTracker) -> str:
    def _format_meter_val(meter: AverageMeter) -> str:
        fmt = meter.fmt[1:] if meter.fmt.startswith(":") else meter.fmt
        return f"{meter.name}:{format(meter.val, fmt)}"

    display_list = [
        f"step:{format_big_number(train_tracker.steps)}",
        f"smpl:{format_big_number(train_tracker.samples)}",
        f"ep:{format_big_number(train_tracker.episodes)}",
        f"epch:{train_tracker.epochs:.2f}",
        *[_format_meter_val(m) for m in train_tracker.metrics.values()],
    ]
    return " ".join(display_list)


# def validate_visual_features_consistency(
#     cfg: PI0Config,
#     features: dict[str, PolicyFeature],
# ) -> None:
#     """
#     Validates visual feature consistency between a policy config and provided dataset/environment features.

#     Args:
#         cfg (PreTrainedConfig): The model or policy configuration containing input_features and type.
#         features (Dict[str, PolicyFeature]): A mapping of feature names to PolicyFeature objects.
#     """
#     expected_visuals = {k for k, v in cfg.input_features.items() if v.type == FeatureType.VISUAL}
#     provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}
#     if not provided_visuals.issubset(expected_visuals):
#         raise_feature_mismatch_error(provided_visuals, expected_visuals)


def make_policy(
    config: TrainConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
):
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created (PI0Config or PI05Config).
             If `cfg.pretrained_path` is set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).
        model_variant: Model variant to use, either "pi0" or "pi0.5".

    Returns:
        An instantiated and device-placed policy model (PI0Policy or PI05Policy).
    """

    # # Select policy class based on model variant
    # if model_variant == "pi0.5":
    #     policy_cls = PI05Policy
    # else:
    #     policy_cls = PI0Policy

    kwargs = {}
    features = dataset_to_policy_features(ds_meta.features)

    # FIXME
    output_features = {
        # Changed from ft.type is FeatureType.ACTION to ft.type == FeatureType.ACTION
        # for different enum classes: flagscale.FeatureType vs lerobot.FeatureType
        key: ft
        for key, ft in features.items()
        if ft.type == FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }
    # kwargs["config"] = config.model

    # PI0 finetuning, so always load a pretrained policy.
    # Load a pretrained policy and override the config if needed (for example, if there are inference-time
    # hyperparameters that we want to vary).
    # kwargs["pretrained_name_or_path"] = cfg.pretrained_path
    # policy = policy_cls.from_pretrained(cfg.pretrained_path, config=cfg)

    # TODO: (yupu) This is a hack, we should find a better way to handle this. LeRobot does this in the policy config.
    # The order of the images is defined in the dataset config.json
    image_features = {key: ft for key, ft in input_features.items() if ft.type is FeatureType.VISUAL}
    config.data.vla_data.image_features = image_features

    policy = QwenGr00t(config=config)
    # policy = Qwen_PI(config=config)
    print(policy)
    print(f"config: {config}")

    # FIXME
    policy.to("cuda")

    return policy, input_features, output_features


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_preprocessor_from_config(
    config: dict[str, Any] | list[str | dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    """
    Create a preprocessor pipeline from step configurations with optional overrides.

    This function creates a PolicyProcessorPipeline directly from step configurations,
    without requiring a pretrained path. It supports overriding step configurations
    similar to PolicyProcessorPipeline.from_pretrained().

    Args:
        config: Can be either:
            - A dict with "name" and "steps" fields (JSON format):
              {"name": "policy_preprocessor", "steps": [...]}
            - A list of step configurations (concise format):
              ["step_name", {"step_name": {...}}]
        overrides: Optional dictionary to override step configurations. Keys should
            match the step's registry_name. Example:
            {"device_processor": {"device": "cuda"},
             "normalizer_processor": {"stats": dataset.meta.stats}}

    Returns:
        A PolicyProcessorPipeline instance with the configured steps.

    Example (JSON format with overrides):
        ```python
        config = {
            "name": "policy_preprocessor",
            "steps": [
                {"registry_name": "device_processor", "config": {"device": "cpu"}},
                {"registry_name": "normalizer_processor", "config": {"eps": 1e-8}}
            ]
        }
        overrides = {
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {"stats": dataset.meta.stats, "features": {...}}
        }
        preprocessor = make_preprocessor_from_config(config, overrides=overrides)
        # device_processor will use device="cuda" (overridden)
        # normalizer_processor will use eps=1e-8 (from config) and stats from overrides
        ```

    Example (concise list format):
        ```python
        steps = [
            "rename_observations_processor",
            "device_processor",
            {"normalizer_processor": {"eps": 1e-8}}
        ]
        preprocessor = make_preprocessor_from_config(steps)
        ```

    Raises:
        ValueError: If a step configuration is invalid or step cannot be instantiated.
        KeyError: If a registry name is not found.
    """
    from flagscale.train.processor.pipeline import ProcessorStepRegistry

    overrides = overrides or {}

    # Determine format and extract step configs
    if isinstance(config, (dict, DictConfig)) and "steps" in config:
        # JSON format: {"name": "...", "steps": [...]}
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        step_configs = config["steps"]
        pipeline_name = config.get("name", "policy_preprocessor")
    elif isinstance(config, list):
        # Concise list format
        step_configs = config
        pipeline_name = "policy_preprocessor"
    else:
        raise ValueError(
            f"Config must be a dict with 'steps' key or a list, got {type(config)}"
        )

    steps = []
    for step_entry in step_configs:
        # Determine step format and normalize to standard dict
        if isinstance(step_entry, str):
            # Concise format: "step_name"
            step_dict = {"registry_name": step_entry, "config": {}}
        elif isinstance(step_entry, (dict, DictConfig)):
            if "registry_name" in step_entry:
                # JSON format: {"registry_name": "...", "config": {...}}
                if isinstance(step_entry, DictConfig):
                    step_entry = OmegaConf.to_container(step_entry, resolve=True)
                step_dict = step_entry
            elif len(step_entry) == 1:
                # Concise format: {"step_name": {...}}
                step_name = next(iter(step_entry.keys()))
                step_config = step_entry[step_name]
                if isinstance(step_config, DictConfig):
                    step_config = OmegaConf.to_container(step_config, resolve=True)
                step_dict = {"registry_name": step_name, "config": step_config}
            else:
                raise ValueError(
                    f"Step config dict must have either 'registry_name' or exactly one key, "
                    f"got {list(step_entry.keys())}"
                )
        else:
            raise ValueError(
                f"Step config must be str or dict, got {type(step_entry)}: {step_entry}"
            )

        # Get step class
        registry_name = step_dict["registry_name"]
        step_class = ProcessorStepRegistry.get(registry_name)

        # Merge config with overrides (overrides take precedence)
        try:
            base_config = step_dict.get("config", {})
            step_overrides = overrides.get(registry_name, {})
            merged_config = {**base_config, **step_overrides}

            step_instance = step_class(**merged_config)
            steps.append(step_instance)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate processor step '{registry_name}' "
                f"with config {merged_config}. Error: {e!s}"
            ) from e

    return PolicyProcessorPipeline(
        steps=steps,
        name=pipeline_name,
    )


def make_pre_post_processors(
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create or load pre- and post-processor pipelines for a given policy.

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_tdmpc_pre_post_processors`).

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: An optional path to load pretrained processor pipelines from.
            If provided, pipelines are loaded from this path.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        NotImplementedError: If a processor factory is not implemented for the given
            policy configuration type.
    """
    return (
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "preprocessor_config_filename",
                f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            ),
            overrides=kwargs.get("preprocessor_overrides", {}),
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "postprocessor_config_filename",
                f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            ),
            overrides=kwargs.get("postprocessor_overrides", {}),
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def update_policy(
    train_metrics: MetricsTracker,
    policy,
    batch: Any,
    optimizer: Optimizer,
    use_amp: bool,
    grad_clip_norm: float,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Uses accelerate for distributed training (matching starVLA).

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained (wrapped by accelerator).
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()

    # Get the policy model (unwrap accelerator if needed) to access config
    policy_model = accelerator.unwrap_model(policy)

    print(f"use_amp: {use_amp}")

    # Use accelerator.accumulate for gradient accumulation support (matching starVLA)
    with accelerator.accumulate(policy):
        optimizer.zero_grad()

        autocast_context = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        )

        with autocast_context:
            loss = policy.forward(batch)

        # Use accelerator.backward instead of loss.backward() (matching starVLA)
        accelerator.backward(loss)

        # Clip gradients using accelerator (matching starVLA)
        grad_norm = None
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            # Compute grad norm even if not clipping
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), float("inf"))

        with lock if lock is not None else nullcontext():
            optimizer.step()

        # Step through pytorch scheduler at every batch instead of epoch
        if lr_scheduler is not None:
            lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(policy_model, "update"):
        policy_model.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item() if grad_norm is not None else 0.0
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    return train_metrics


# Commented out: old update_policy using manual DDP
# def update_policy_old(
#     train_metrics: MetricsTracker,
#     policy,
#     batch: Any,
#     optimizer: Optimizer,
#     use_amp: bool,
#     grad_clip_norm: float,
#     lr_scheduler=None,
#     lock=None,
# ) -> tuple[MetricsTracker, dict]:
#     start_time = time.perf_counter()
#     policy_model = policy.module if isinstance(policy, DDP) else policy
#     print(f"use_amp: {use_amp}")
#     autocast_context = (
#         torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
#     )
#     with autocast_context:
#         loss = policy.forward(batch)
#     loss.backward()
#     if grad_clip_norm > 0:
#         grad_norm = torch.nn.utils.clip_grad_norm_(
#             policy.module.parameters() if isinstance(policy, DDP) else policy.parameters(),
#             grad_clip_norm,
#         )
#     else:
#         grad_norm = torch.nn.utils.clip_grad_norm_(
#             policy.module.parameters() if isinstance(policy, DDP) else policy.parameters(),
#             float("inf"),
#             error_if_nonfinite=False,
#         )
#     with lock if lock is not None else nullcontext():
#         optimizer.step()
#     optimizer.zero_grad()
#     if lr_scheduler is not None:
#         lr_scheduler.step()
#     if has_method(policy_model, "update"):
#         policy_model.update()
#     train_metrics.loss = loss.item()
#     train_metrics.grad_norm = grad_norm.item()
#     train_metrics.lr = optimizer.param_groups[0]["lr"]
#     train_metrics.update_s = time.perf_counter() - start_time
#     return train_metrics


def main(config: TrainConfig, seed: int):

    # import debugpy
    # debugpy.listen(("0.0.0.0", 9096))
    # debugpy.wait_for_client()
    # debugpy.breakpoint()

    set_seed(seed)
    print(f"[DEBUG RNG main] After set_seed: torch state[:10] = {torch.get_rng_state()[:10].tolist()}")

    # Use accelerator instead of manual DDP (matching starVLA)
    device = accelerator.device
    is_main_process = accelerator.is_main_process
    accelerator.print(accelerator.state)
    print(f"[DEBUG RNG main] After accelerator setup: torch state[:10] = {torch.get_rng_state()[:10].tolist()}")

    # Commented out: old manual DDP initialization
    # local_rank = init_ddp()
    # device = torch.device("cuda", local_rank)
    # rank = dist.get_rank()
    # is_main_process = rank == 0 and local_rank == 0
    # print(f"[DEBUG RNG main] After init_ddp: torch state[:10] = {torch.get_rng_state()[:10].tolist()}")

    dataset = make_dataset(config.data)
    print(f"[DEBUG RNG main] After make_dataset: torch state[:10] = {torch.get_rng_state()[:10].tolist()}")

    accelerator.wait_for_everyone()  # Use accelerator instead of dist.barrier()

    # Reset seed before model creation to match starVLA initialization order
    # (starVLA creates model before dataset, so we reset seed to get same weights)
    # set_seed(seed)
    # print(f"[DEBUG RNG main] After 2nd set_seed: torch state[:10] = {torch.get_rng_state()[:10].tolist()}")

    policy, input_features, output_features = make_policy(config=config, ds_meta=dataset.meta)
    # register_debug_hooks(policy)

    accelerator.wait_for_everyone()  # Use accelerator instead of dist.barrier()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    # Only provide dataset_stats when not resuming from saved processor state
    processor_kwargs["dataset_stats"] = dataset.meta.stats

    # Prepare overrides for preprocessor steps
    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {
                **input_features,
                **output_features,
            }
        },
        # "tokenizer_processor": {"tokenizer_name": config.model.tokenizer_path},
    }

    num_workers = 0 # config.system.num_workers
    shuffle = config.system.shuffle

    # # Wrap dataset with StarVLAFormatDataset for starVLA-compatible output format
    # image_keys = getattr(config.data, "image_keys", None) or [
    #     "observation.images.image",
    #     "observation.images.wrist_image",
    # ]
    # starvla_dataset = StarVLAFormatDataset(
    #     dataset,
    #     image_keys=image_keys,
    #     image_size=(224, 224),
    # )

    # DistributedSampler ensures each rank gets different data
    # Use accelerator's process info (matching starVLA pattern)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        # starvla_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=shuffle,
        drop_last=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        # starvla_dataset,
        num_workers=num_workers,
        batch_size=config.system.batch_size,
        shuffle=False,  # Must be False when using sampler
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        # collate_fn=collate_fn_starvla,  # Return batch as list of dicts (starVLA style)
    )

    # Setup preprocessor
    preprocessor = None
    if config.data.preprocessor is not None:
        preprocessor = make_preprocessor_from_config(
            config.data.preprocessor,
            overrides=preprocessor_overrides
        )

    # Setup optimizer and scheduler (applies freeze config before accelerator.prepare)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(policy, config)

    # Use accelerator.prepare instead of manual DDP wrapping (matching starVLA)
    # This handles DDP wrapping, moving to device, etc.
    accelerator.dataloader_config.dispatch_batches = False  # Match starVLA setting
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

    # Commented out: old manual DDP wrapping
    # policy = DDP(
    #     policy,
    #     device_ids=[local_rank],
    #     find_unused_parameters=True,
    #     output_device=local_rank,
    # )

    accelerator.wait_for_everyone()  # Use accelerator instead of dist.barrier()

    dl_iter = cycle(dataloader)

    # policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use accelerator.num_processes instead of dist.get_world_size()
    effective_batch_size = config.system.batch_size * accelerator.num_processes

    step = 0

    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    # To ensures proper data shuffling across epochs in distributed training
    epoch = 0
    samples_per_epoch = len(dataset) // effective_batch_size
    sampler.set_epoch(epoch)

    action_stats = dataset.meta.stats.get("action", {})
    if is_main_process:
        print(f"[DEBUG GRIPPER] action stats min: {action_stats.get('min', 'N/A')}")
        print(f"[DEBUG GRIPPER] action stats max: {action_stats.get('max', 'N/A')}")
    _debug_dumped = False

    for _ in range(step, config.system.train_steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        if not _debug_dumped and is_main_process and "action" in batch:
            print(f"[DEBUG GRIPPER] BEFORE preproc action[0,0,:]: {batch['action'][0, 0, :].tolist()}")
            print(f"[DEBUG GRIPPER] BEFORE preproc action[:,: ,6] (gripper): {batch['action'][:, :, 6].flatten()[:16].tolist()}")

        if preprocessor is not None:
            batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        if not _debug_dumped and is_main_process and "action" in batch:
            print(f"[DEBUG GRIPPER] AFTER preproc action[0,0,:]: {batch['action'][0, 0, :].tolist()}")
            print(f"[DEBUG GRIPPER] AFTER preproc action[:,:,6] (gripper): {batch['action'][:, :, 6].flatten()[:16].tolist()}")
            _debug_dumped = True

        st = time.perf_counter()
        train_tracker = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            use_amp=config.system.use_amp,
            grad_clip_norm=config.system.grad_clip_norm,
            lr_scheduler=lr_scheduler,
        )
        print(f"update_policy time: {time.perf_counter() - st}")
        print(f"train_tracker at step {step}: {format_train_tracker_step(train_tracker)}")

        step += 1
        train_tracker.step()

        # Update epoch counter for sampler.set_epoch() when we've processed one epoch worth of samples
        # This ensures proper data shuffling across epochs in distributed training
        if samples_per_epoch > 0 and step % samples_per_epoch == 0:
            epoch += 1
            sampler.set_epoch(epoch)

        if step % config.system.log_freq == 0 and is_main_process:
            logger.info(f"step: {step} {format_train_tracker_step(train_tracker)}")
            train_tracker.reset_averages()

        if (
            config.system.checkpoint.save_checkpoint
            and step % config.system.checkpoint.save_freq == 0
        ):
            # Synchronize all processes before checkpoint saving
            accelerator.wait_for_everyone()

            if is_main_process:
                from pathlib import Path
                logger.info(f"Saving checkpoint at step {step}")
                output_dir = Path(config.system.checkpoint.output_directory)
                checkpoint_dir = get_step_checkpoint_dir(
                    output_dir, config.system.train_steps, step
                )
                # Use accelerator.unwrap_model instead of policy.module
                policy_to_save = accelerator.unwrap_model(policy)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    policy=policy_to_save,
                    config=config,
                    preprocessor=preprocessor,
                )
                update_last_checkpoint(checkpoint_dir)

            # Synchronize all processes after checkpoint saving
            accelerator.wait_for_everyone()

    if is_main_process:
        logger.info("Training completed")

    # Properly clean up using accelerator (matching starVLA)
    accelerator.wait_for_everyone()
    # Note: accelerator handles process group cleanup automatically
    # dist.destroy_process_group()  # Commented out: handled by accelerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train QwenGr00t model. This script is typically called by the flagscale runner, not directly."
    )
    parser.add_argument(
        "--config-file", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    config_file_path = args.config_file

    # Load config from YAML file (Hydra-generated config.yaml contains both train and experiment)
    config = OmegaConf.load(config_file_path)

    logger.info(f"full config: {config}")

    # Extract train config and convert to Pydantic TrainConfig (preserves raw configs)
    train_config = TrainConfig.from_hydra_config(config)

    # Extract experiment config (seed, exp_dir, etc.)
    experiment_config = OmegaConf.to_container(config.experiment, resolve=True)
    seed = experiment_config.get("seed", 42)

    logger.info("=" * 100)
    logger.info(f"Experiment: {experiment_config}")
    logger.info(f"Train config: {train_config}")

    main(train_config, seed)

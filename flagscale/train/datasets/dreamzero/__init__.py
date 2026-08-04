# Copyright (c) 2025, FlagScale Authors. All rights reserved.
"""
DreamZero Data Pipeline for FlagScale native training backend.

Multi-anchor temporal sampling with rejection guarantees fixed-shape outputs
for batch_size > 1. Samples exactly max_chunk_size chunks per sample; if the
trajectory is too short, the sample is rejected and a new index is drawn.

Data flow:
  LeRobot parquet + video files
    -> DreamZeroDataset.__getitem__ (multi-anchor sampling, load, transform)
    -> DreamZeroCollator.__call__ (batch, tokenize text)
    -> get_batch (move to device, dtype cast)
    -> model.forward()

Expected batch keys from get_batch:
  - images: (B, 33, 2*H, 2*W, C) uint8 video frames (2-view tiled grid)
  - action: (B, 96, 32) float32 normalized [-1, 1]
  - action_mask: (B, 96, 32) bool
  - state: (B, 4, 64) float32
  - state_mask: (B, 4, 64) bool
  - text: (B, max_text_len) int64 tokenized text
  - text_attention_mask: (B, max_text_len) int64
  - embodiment_id: (B,) int64
  - has_real_action: (B,) bool
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

# Default video frame offsets within each chunk (8 frames per chunk)
DEFAULT_VIDEO_IN_CHUNK_OFFSETS = (0, 3, 6, 9, 12, 15, 18, 21)


class EmptyTemporalSampleError(ValueError):
    """Raised when multi-anchor sampling cannot fill max_chunk_size chunks."""
    pass


class DreamZeroDataset(Dataset):
    """Map-style dataset for DreamZero training with multi-anchor temporal sampling.

    Reads LeRobot v2 format: per-episode parquet files for low-dim data and
    mp4 video files decoded via decord. Each sample produces exactly
    max_chunk_size temporal chunks with fixed output shapes.
    """

    def __init__(
        self,
        data_path: str,
        max_chunk_size: int = 4,
        macro_stride: int = 24,
        action_horizon: int = 24,
        state_horizon: int = 1,
        action_dim: int = 32,
        max_state_dim: int = 64,
        video_in_chunk_offsets: tuple[int, ...] = DEFAULT_VIDEO_IN_CHUNK_OFFSETS,
        embodiment_tag: str = "libero",
        embodiment_tag_mapping: dict[str, int] | None = None,
        image_size: tuple[int, int] = (176, 320),
        num_views: int = 2,
        resample_attempts: int = 8,
        relative_action: bool = True,
        relative_action_keys: list[str] | None = None,
        crop_ratio: float = 0.95,
        color_jitter: bool = True,
        training: bool = True,
    ):
        self.data_path = Path(data_path)
        self.max_chunk_size = max_chunk_size
        self.macro_stride = macro_stride
        self.action_horizon = action_horizon
        self.state_horizon = state_horizon
        self.action_dim = action_dim
        self.max_state_dim = max_state_dim
        self.video_in_chunk_offsets = video_in_chunk_offsets
        self.embodiment_tag = embodiment_tag
        self.embodiment_tag_mapping = embodiment_tag_mapping or {
            "agibot": 26, "oxe_droid": 17, "gr1_unified": 2,
            "mecka_hands": 27, "libero": 17, "dream": 100, "lapa": 101,
        }
        self.image_size = image_size
        self.num_views = num_views
        self.resample_attempts = resample_attempts
        self.relative_action = relative_action
        self.relative_action_keys = relative_action_keys or ["joint_pos"]
        self.crop_ratio = crop_ratio
        self.color_jitter = color_jitter
        self.training = training

        # Derived constants
        self.frames_per_chunk = len(video_in_chunk_offsets)  # 8
        self.total_video_frames = max_chunk_size * self.frames_per_chunk + 1  # 33
        self.total_action_steps = max_chunk_size * action_horizon  # 96
        self.total_state_steps = max_chunk_size * state_horizon  # 4

        # Load dataset metadata
        self._load_metadata()
        self._load_normalization_stats()

        logger.info(
            f"DreamZeroDataset: {len(self)} samples, {len(self._episodes)} episodes, "
            f"max_chunk_size={max_chunk_size}, macro_stride={macro_stride}, "
            f"data_path={data_path}"
        )

    def _load_metadata(self):
        """Load LeRobot v2 dataset metadata and build sample index."""
        import pyarrow.parquet as pq

        meta_dir = self.data_path / "meta"
        with open(meta_dir / "info.json") as f:
            info = json.load(f)

        self._chunks_size = info.get("chunks_size", 1000)
        self._fps = info.get("fps", 10)
        self._features = info.get("features", {})

        # Video keys
        video_keys = [k for k, v in self._features.items() if v.get("dtype") == "video"]
        self._video_keys = video_keys if video_keys else ["observation.images.image"]

        # Load modality.json for state/action slicing
        modality_path = meta_dir / "modality.json"
        if modality_path.exists():
            with open(modality_path) as f:
                self._modality = json.load(f)
        else:
            self._modality = {}

        # Load task texts
        self._tasks = {}
        tasks_path = meta_dir / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    entry = json.loads(line)
                    self._tasks[entry["task_index"]] = entry["task"]

        # Load episodes metadata
        episodes_path = meta_dir / "episodes.jsonl"
        self._episodes = []
        with open(episodes_path) as f:
            for line in f:
                self._episodes.append(json.loads(line))

        # Build sample index: (episode_idx, frame_in_ep) for all valid frames
        # A frame is valid if it can anchor multi-chunk sampling
        self._sample_index = []
        for ep in self._episodes:
            ep_idx = ep["episode_index"]
            ep_len = ep["length"]
            for frame_idx in range(ep_len):
                self._sample_index.append((ep_idx, frame_idx, ep_len))

        # Pre-load parquet tables (they're small for LIBERO)
        self._episode_tables = {}
        data_dir = self.data_path / "data"
        for ep in self._episodes:
            ep_idx = ep["episode_index"]
            chunk_idx = ep_idx // self._chunks_size
            pf = data_dir / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            if pf.exists():
                self._episode_tables[ep_idx] = pq.read_table(pf)

    def _load_normalization_stats(self):
        """Load q99 normalization stats for actions and states."""
        meta_dir = self.data_path / "meta"
        stats_path = meta_dir / "relative_stats_dreamzero.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self._norm_stats = json.load(f)
        else:
            # Fallback: try stats.json
            stats_path = meta_dir / "stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    self._norm_stats = json.load(f)
            else:
                self._norm_stats = {}

    def __len__(self):
        return len(self._sample_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load a sample with multi-anchor rejection loop."""
        last_error = None
        for attempt in range(self.resample_attempts):
            try:
                return self._build_sample(idx)
            except EmptyTemporalSampleError as e:
                last_error = e
                idx = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"Failed to sample valid multi-anchor window after "
            f"{self.resample_attempts} attempts: {last_error}"
        )

    def _sample_anchors(self, frame_in_ep: int, ep_len: int) -> list[int]:
        """Sample exactly max_chunk_size anchors via multi-anchor expansion.

        Expands outward from frame_in_ep in steps of macro_stride.
        Each anchor must allow a full action window (anchor + action_horizon - 1 < ep_len).
        Raises EmptyTemporalSampleError if fewer than max_chunk_size anchors found.
        """
        anchors = []

        def try_add(anchor: int):
            if len(anchors) >= self.max_chunk_size:
                return
            # Anchor must allow full action window and video offsets
            max_offset = max(self.video_in_chunk_offsets)
            if anchor < 0 or anchor + max_offset >= ep_len:
                return False
            if anchor + self.action_horizon - 1 >= ep_len:
                return False
            anchors.append(anchor)
            return True

        # Start with the given frame
        try_add(frame_in_ep)

        step = 1
        back_done = False
        fwd_done = False
        while len(anchors) < self.max_chunk_size and (not back_done or not fwd_done):
            if not back_done:
                back = frame_in_ep - self.macro_stride * step
                if back < 0:
                    back_done = True
                else:
                    try_add(back)

            if len(anchors) >= self.max_chunk_size:
                break

            if not fwd_done:
                fwd = frame_in_ep + self.macro_stride * step
                if fwd >= ep_len:
                    fwd_done = True
                else:
                    try_add(fwd)

            step += 1

        if len(anchors) < self.max_chunk_size:
            raise EmptyTemporalSampleError(
                f"Only found {len(anchors)} anchors (need {self.max_chunk_size}) "
                f"at frame {frame_in_ep}, ep_len={ep_len}"
            )

        return sorted(anchors)

    def _compute_indices(self, anchors: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute video, action, and state frame indices from anchors.

        Returns:
            video_indices: (33,) - 8 per chunk + 1 boundary
            action_indices: (96,) - 24 per chunk
            state_indices: (4,) - 1 per chunk
        """
        video_idx = []
        action_idx = []
        state_idx = []

        for anchor in anchors:
            # Video: 8 frames at offsets within each chunk
            for offset in self.video_in_chunk_offsets:
                video_idx.append(anchor + offset)
            # Action: 24 contiguous frames starting at anchor
            for a in range(self.action_horizon):
                action_idx.append(anchor + a)
            # State: anchor frame
            state_idx.append(anchor)

        # Add boundary frame: last video frame + 3
        last_video = video_idx[-1]
        video_idx.append(last_video + 3)

        return (
            np.array(video_idx, dtype=np.int64),
            np.array(action_idx, dtype=np.int64),
            np.array(state_idx, dtype=np.int64),
        )

    def _build_sample(self, idx: int) -> dict[str, Any]:
        """Build a single training sample."""
        ep_idx, frame_in_ep, ep_len = self._sample_index[idx]

        # 1. Multi-anchor temporal sampling
        anchors = self._sample_anchors(frame_in_ep, ep_len)
        video_indices, action_indices, state_indices = self._compute_indices(anchors)

        # Validate boundary frame
        if video_indices[-1] >= ep_len:
            raise EmptyTemporalSampleError(
                f"Boundary frame {video_indices[-1]} >= ep_len {ep_len}"
            )

        # 2. Load video frames
        images = self._load_video(ep_idx, video_indices)

        # 3. Load actions and states from parquet
        table = self._episode_tables[ep_idx]
        actions, action_mask = self._load_actions(table, action_indices, state_indices)
        states, state_mask = self._load_states(table, state_indices)

        # 4. Load language
        language = self._load_language(table, frame_in_ep, ep_idx)

        # 5. Embodiment ID
        embodiment_id = self.embodiment_tag_mapping.get(self.embodiment_tag, 17)

        return {
            "video": images,  # (33, 2H, 2W, 3) uint8
            "action": actions,  # (96, 32) float32
            "action_mask": action_mask,  # (96, 32) bool
            "state": states,  # (4, 64) float32
            "state_mask": state_mask,  # (4, 64) bool
            "language": language,  # str
            "embodiment_id": embodiment_id,  # int
            "has_real_action": True,
        }

    def _load_video(self, ep_idx: int, frame_indices: np.ndarray) -> np.ndarray:
        """Load video frames, apply transforms, tile multiview.

        Returns: (T, 2*H, 2*W, 3) uint8
        """
        import decord
        decord.bridge.set_bridge("native")

        episode_chunk = ep_idx // self._chunks_size
        H, W = self.image_size  # target per-view size

        video_keys_to_use = self._video_keys[:self.num_views]
        views = []

        for video_key in video_keys_to_use:
            video_path = (
                self.data_path / "videos"
                / f"chunk-{episode_chunk:03d}"
                / video_key
                / f"episode_{ep_idx:06d}.mp4"
            )

            if video_path.exists():
                vr = decord.VideoReader(str(video_path), num_threads=1)
                n_total = len(vr)
                safe_indices = [min(int(i), n_total - 1) for i in frame_indices]
                batch = vr.get_batch(safe_indices).asnumpy()  # (T, H_orig, W_orig, C)

                # Apply crop and resize
                frames = self._apply_video_transforms(batch)
                views.append(frames)
            else:
                logger.warning(f"Video not found: {video_path}")
                views.append(np.zeros((len(frame_indices), H, W, 3), dtype=np.uint8))

        # Pad missing views with black
        while len(views) < self.num_views:
            views.append(np.zeros((len(frame_indices), H, W, 3), dtype=np.uint8))

        # Tile into 2x2 grid: (T, 2*H, 2*W, C)
        T = len(frame_indices)
        tiled = np.zeros((T, 2 * H, 2 * W, 3), dtype=np.uint8)

        # View 0 -> top-left (head/main camera)
        tiled[:, :H, :W, :] = views[0]
        # View 1 -> bottom-left (wrist camera)
        if len(views) > 1:
            tiled[:, H:, :W, :] = views[1]
        # Top-right and bottom-right stay black (matching reference for 2-view)

        return tiled

    def _apply_video_transforms(self, frames: np.ndarray) -> np.ndarray:
        """Apply crop, resize, and optional color jitter. Returns uint8 (T, H, W, 3)."""
        from PIL import Image

        H, W = self.image_size
        T, h_orig, w_orig, C = frames.shape

        result = np.empty((T, H, W, C), dtype=np.uint8)

        # Random crop params (same for all frames in sample)
        if self.training and self.crop_ratio < 1.0:
            crop_h = int(h_orig * self.crop_ratio)
            crop_w = int(w_orig * self.crop_ratio)
            top = random.randint(0, h_orig - crop_h)
            left = random.randint(0, w_orig - crop_w)
        else:
            # Center crop
            crop_h = int(h_orig * self.crop_ratio)
            crop_w = int(w_orig * self.crop_ratio)
            top = (h_orig - crop_h) // 2
            left = (w_orig - crop_w) // 2

        for i in range(T):
            frame = frames[i]
            # Crop
            frame = frame[top:top+crop_h, left:left+crop_w]
            # Resize
            img = Image.fromarray(frame)
            img = img.resize((W, H), Image.BILINEAR)
            result[i] = np.array(img)

        # Color jitter (simple brightness/contrast/saturation)
        if self.training and self.color_jitter:
            result = self._apply_color_jitter(result)

        return result

    def _apply_color_jitter(self, frames: np.ndarray) -> np.ndarray:
        """Simple color jitter matching reference VideoColorJitter defaults."""
        # brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        brightness = 1.0 + random.uniform(-0.1, 0.1)
        contrast = 1.0 + random.uniform(-0.1, 0.1)

        frames = frames.astype(np.float32)
        # Brightness
        frames = frames * brightness
        # Contrast
        mean = frames.mean(axis=(1, 2), keepdims=True)
        frames = (frames - mean) * contrast + mean
        # Clip and convert back
        frames = np.clip(frames, 0, 255).astype(np.uint8)
        return frames

    def _load_actions(
        self, table, action_indices: np.ndarray, state_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load actions, apply relative action and q99 normalization.

        Returns: (96, action_dim) float32, (96, action_dim) bool
        """
        n_rows = table.num_rows
        action_modality = self._modality.get("action", {})

        # Gather raw action components
        raw_actions = []
        for idx in action_indices:
            safe_idx = min(int(idx), n_rows - 1)
            row_val = table.column("action")[safe_idx].as_py()
            raw_actions.append(row_val)

        actions = np.array(raw_actions, dtype=np.float32)  # (96, raw_action_dim)
        raw_dim = actions.shape[1]

        # Apply relative action (per-chunk anchor subtraction)
        if self.relative_action:
            # Load states at anchor frames for relative computation
            state_arr = []
            for idx in state_indices:
                safe_idx = min(int(idx), n_rows - 1)
                row_val = table.column("observation.state")[safe_idx].as_py()
                state_arr.append(row_val)
            state_arr = np.array(state_arr, dtype=np.float32)  # (4, state_dim)

            # Per-chunk subtraction for relative_action_keys
            for key in self.relative_action_keys:
                if key not in action_modality:
                    continue
                action_slice = slice(
                    action_modality[key]["start"],
                    action_modality[key]["end"],
                )
                # Find corresponding state slice
                state_modality = self._modality.get("state", {})
                if key in state_modality:
                    state_slice = slice(
                        state_modality[key]["start"],
                        state_modality[key]["end"],
                    )
                else:
                    continue

                a_start = action_modality[key]["start"]
                a_end = action_modality[key]["end"]
                s_start = state_modality[key]["start"]
                s_end = state_modality[key]["end"]
                d = min(a_end - a_start, s_end - s_start)

                for c in range(self.max_chunk_size):
                    rs = c * self.action_horizon
                    re = rs + self.action_horizon
                    ref = state_arr[c, s_start:s_start + d]
                    actions[rs:re, a_start:a_start + d] -= ref

        # Q99 normalization
        actions = self._normalize_q99(actions, action_modality)

        # Pad to action_dim
        if raw_dim < self.action_dim:
            pad = np.zeros((actions.shape[0], self.action_dim - raw_dim), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=1)
        elif raw_dim > self.action_dim:
            actions = actions[:, :self.action_dim]

        # Clip to [-1, 1]
        actions = np.clip(actions, -1.0, 1.0)

        # Action mask: True for real dims, False for padding
        mask = np.zeros((self.total_action_steps, self.action_dim), dtype=bool)
        mask[:, :raw_dim] = True

        return actions, mask

    def _load_states(
        self, table, state_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load states and normalize.

        Returns: (4, max_state_dim) float32, (4, max_state_dim) bool
        """
        n_rows = table.num_rows

        raw_states = []
        for idx in state_indices:
            safe_idx = min(int(idx), n_rows - 1)
            row_val = table.column("observation.state")[safe_idx].as_py()
            raw_states.append(row_val)

        states = np.array(raw_states, dtype=np.float32)  # (4, raw_state_dim)
        raw_dim = states.shape[1]

        # Q99 normalization for states
        state_modality = self._modality.get("state", {})
        states = self._normalize_q99_state(states, state_modality)

        # Pad to max_state_dim
        if raw_dim < self.max_state_dim:
            pad = np.zeros((states.shape[0], self.max_state_dim - raw_dim), dtype=np.float32)
            states = np.concatenate([states, pad], axis=1)
        elif raw_dim > self.max_state_dim:
            states = states[:, :self.max_state_dim]

        # Clip
        states = np.clip(states, -1.0, 1.0)

        # State mask
        mask = np.zeros((self.total_state_steps, self.max_state_dim), dtype=bool)
        mask[:, :raw_dim] = True

        return states, mask

    def _normalize_q99(self, data: np.ndarray, modality_info: dict) -> np.ndarray:
        """Apply q99 normalization: maps [q01, q99] -> [-1, 1]."""
        if not self._norm_stats:
            return data

        for key, meta in modality_info.items():
            if key not in self._norm_stats:
                continue
            stats = self._norm_stats[key]
            if "q01" not in stats or "q99" not in stats:
                continue

            start = meta["start"]
            end = meta["end"]
            q01 = np.array(stats["q01"], dtype=np.float32)
            q99 = np.array(stats["q99"], dtype=np.float32)

            denom = q99 - q01
            valid = denom > 1e-8

            # Normalize: 2 * (x - q01) / (q99 - q01) - 1
            d = end - start
            for i in range(d):
                if valid[i]:
                    data[:, start + i] = 2.0 * (data[:, start + i] - q01[i]) / denom[i] - 1.0

        return data

    def _normalize_q99_state(self, data: np.ndarray, modality_info: dict) -> np.ndarray:
        """Apply q99 normalization to states (uses main stats.json if available)."""
        # States use absolute stats, not relative
        meta_dir = self.data_path / "meta"
        stats_path = meta_dir / "stats.json"
        if not stats_path.exists():
            return data

        with open(stats_path) as f:
            stats = json.load(f)

        # Look for observation.state stats
        obs_key = "observation.state"
        if obs_key not in stats:
            return data

        obs_stats = stats[obs_key]
        if "q01" not in obs_stats or "q99" not in obs_stats:
            return data

        q01 = np.array(obs_stats["q01"], dtype=np.float32)
        q99 = np.array(obs_stats["q99"], dtype=np.float32)
        denom = q99 - q01
        valid = denom > 1e-8

        raw_dim = min(data.shape[1], len(q01))
        for i in range(raw_dim):
            if valid[i]:
                data[:, i] = 2.0 * (data[:, i] - q01[i]) / denom[i] - 1.0

        return data

    def _load_language(self, table, frame_idx: int, ep_idx: int) -> str:
        """Load language instruction for this sample."""
        if "annotation.task" in table.column_names:
            val = table.column("annotation.task")[frame_idx].as_py()
            if val:
                return str(val)
        if "task_index" in table.column_names:
            task_idx = table.column("task_index")[frame_idx].as_py()
            return self._tasks.get(task_idx, "")
        return ""

    def _load_language(self, table, frame_idx: int, ep_idx: int) -> str:
        """Load language instruction for the sample."""
        if "annotation.task" in table.column_names:
            val = table.column("annotation.task")[frame_idx].as_py()
            if val:
                return str(val)

        if "task_index" in table.column_names:
            task_idx = table.column("task_index")[frame_idx].as_py()
            return self._tasks.get(task_idx, "")

        # Fallback from episodes metadata
        for ep in self._episodes:
            if ep["episode_index"] == ep_idx:
                tasks = ep.get("tasks", [])
                if tasks:
                    return tasks[0]
        return ""


class DreamZeroCollator:
    """Collate DreamZero samples into batches with text tokenization.

    Handles:
    - Stacking numpy arrays into tensors
    - Tokenizing language instructions
    - Creating embodiment_id tensor
    """

    def __init__(
        self,
        tokenizer_path: str,
        max_text_length: int = 512,
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_text_length = max_text_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of samples into a batch."""
        # Stack numpy arrays -> tensors
        images = torch.from_numpy(np.stack([f["video"] for f in features]))
        actions = torch.from_numpy(np.stack([f["action"] for f in features]))
        action_masks = torch.from_numpy(np.stack([f["action_mask"] for f in features]))
        states = torch.from_numpy(np.stack([f["state"] for f in features]))
        state_masks = torch.from_numpy(np.stack([f["state_mask"] for f in features]))

        has_real_action = torch.tensor(
            [f["has_real_action"] for f in features], dtype=torch.bool
        )

        # Tokenize language instructions
        languages = [f["language"] for f in features]
        tokenized = self.tokenizer(
            languages,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Embodiment IDs
        embodiment_ids = torch.tensor(
            [f["embodiment_id"] for f in features], dtype=torch.long
        )

        return {
            "images": images,                          # (B, 33, 2H, 2W, 3) uint8
            "action": actions,                         # (B, 96, 32) float32
            "action_mask": action_masks,               # (B, 96, 32) bool
            "has_real_action": has_real_action,         # (B,) bool
            "state": states,                           # (B, 4, 64) float32
            "state_mask": state_masks,                 # (B, 4, 64) bool
            "text": tokenized["input_ids"],            # (B, max_text_len) int64
            "text_attention_mask": tokenized["attention_mask"],  # (B, max_text_len) int64
            "embodiment_id": embodiment_ids,           # (B,) int64
        }


def build_dataloader(
    data_path: str,
    tokenizer_path: str,
    batch_size: int = 1,
    num_workers: int = 8,
    max_chunk_size: int = 4,
    macro_stride: int = 24,
    action_horizon: int = 24,
    state_horizon: int = 1,
    action_dim: int = 32,
    max_state_dim: int = 64,
    embodiment_tag: str = "libero",
    embodiment_tag_mapping: dict[str, int] | None = None,
    image_size: tuple[int, int] = (176, 320),
    max_text_length: int = 512,
    num_views: int = 2,
    shuffle: bool = True,
    distributed: bool = True,
    relative_action: bool = True,
    crop_ratio: float = 0.95,
    color_jitter: bool = True,
    training: bool = True,
) -> DataLoader:
    """Build DataLoader for DreamZero training with multi-anchor sampling.

    All samples are guaranteed to have exactly max_chunk_size chunks, so
    np.stack in the collator always succeeds for batch_size > 1.
    """
    dataset = DreamZeroDataset(
        data_path=data_path,
        max_chunk_size=max_chunk_size,
        macro_stride=macro_stride,
        action_horizon=action_horizon,
        state_horizon=state_horizon,
        action_dim=action_dim,
        max_state_dim=max_state_dim,
        embodiment_tag=embodiment_tag,
        embodiment_tag_mapping=embodiment_tag_mapping,
        image_size=image_size,
        num_views=num_views,
        relative_action=relative_action,
        crop_ratio=crop_ratio,
        color_jitter=color_jitter,
        training=training,
    )

    collator = DreamZeroCollator(
        tokenizer_path=tokenizer_path,
        max_text_length=max_text_length,
    )

    sampler = None
    if distributed and torch.distributed.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
        )
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collator,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logger.info(
        f"Built DreamZero DataLoader: {len(dataset)} samples, "
        f"batch_size={batch_size}, max_chunk_size={max_chunk_size}, "
        f"distributed={distributed}"
    )
    return dataloader


def get_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Move batch to device and cast dtypes for model consumption."""
    result = {}

    # Images stay as uint8 on device (VAE handles normalization internally)
    result["images"] = batch["images"].to(device, non_blocking=True)

    # Actions and states in compute dtype
    result["action"] = batch["action"].to(device, dtype=compute_dtype, non_blocking=True)
    result["action_mask"] = batch["action_mask"].to(device, non_blocking=True)
    result["state"] = batch["state"].to(device, dtype=compute_dtype, non_blocking=True)
    result["state_mask"] = batch["state_mask"].to(device, non_blocking=True)

    # Text tokens as int64
    result["text"] = batch["text"].to(device, non_blocking=True)
    result["text_attention_mask"] = batch["text_attention_mask"].to(device, non_blocking=True)

    # Scalar tensors
    result["has_real_action"] = batch["has_real_action"].to(device, non_blocking=True)
    result["embodiment_id"] = batch["embodiment_id"].to(device, non_blocking=True)

    return result

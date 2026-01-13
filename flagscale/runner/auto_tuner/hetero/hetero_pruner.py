import csv
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.runner.auto_tuner.prune.history import (
    _HISTORY_BASED_PRUNE_FUNC,
    prune_by_sequence_parallel as original_prune_by_sp,
)
from flagscale.runner.auto_tuner.prune.pruner import Pruner
from flagscale.runner.auto_tuner.utils import beside, compare_by_recompute


class HeteroPruner(Pruner):
    """
    A specialized Pruner for heterogeneous training environments.

    It evaluates strategies based on:
    1. Architectural validity (e.g., layer splits matching pipeline size).
    2. Heterogeneous historical data (checking if identical mesh configs failed previously).
    3. Theoretical memory models (comparing estimated usage against device limits).
    4. Standard heuristics (reusing homogeneous logic where applicable).
    """

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("FlagScale-AutoTuner")

        self.pruned_strategies: list[dict[str, Any]] = []
        self.pruned_idx_counter = 1

        # Robustly determine the path for the pruned history log
        exp_dir = getattr(config.experiment, "exp_dir", os.getcwd())
        self.pruned_history_path = os.path.join(exp_dir, "auto_tuner", "pruned_history.csv")

        # Ensure the directory exists
        try:
            os.makedirs(os.path.dirname(self.pruned_history_path), exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create directory for pruned history: {e}")

    def prune(self, strategy: dict[str, Any], history: list[dict[str, Any]] | None = None) -> bool:
        """
        Main entry point for pruning logic.
        Returns True if the strategy should be pruned (skipped), False otherwise.
        """
        if history is None:
            history = []

        # 1. Architectural Validity Checks (Type A)
        # Fast structural checks (e.g., split sums, tied embeddings).
        is_invalid, reason = self._check_architectural_validity(strategy)
        if is_invalid:
            return self._apply_prune(strategy, reason, "ARCH_INVALID")

        # 2. Hetero-specific History Checks (Type B)
        # Check if an identical hardware/batch-size configuration failed OOM in the past.
        if self._check_hetero_history_oom(strategy, history):
            reason = strategy.get("prune_reason", "OOM predicted by identical hetero history")
            return self._apply_prune(strategy, reason, "OOM_PREDICTED_HETERO")

        # 3. Theoretical Memory Model Checks (Type C)
        # Use the hetero memory model estimates to predict OOM.
        if "hetero_memory_model" in self.config.experiment.auto_tuner:
            is_oom, reason = self._check_memory_model_utilization(strategy)
            if is_oom:
                self.pruned_by_memory_model += 1
                # Determine if it was a calc failure (inf) or high utilization
                status = "OOM_CALC_FAIL" if "inf" in reason else "OOM_PREDICTED_UTIL_HIGH"
                return self._apply_prune(strategy, reason, status)

        # 4. Standard/Homogeneous Heuristics (Type D)
        # Reuse existing heuristic functions from the registry.
        for func in _HISTORY_BASED_PRUNE_FUNC:
            should_prune = False

            # Special handling for Sequence Parallelism in hetero environments
            if func.__name__ == original_prune_by_sp.__name__:
                if self._corrected_prune_by_sequence_parallel(self.config, strategy, history):
                    should_prune = True
            else:
                # Execute standard heuristic
                if func(self.config, strategy, history):
                    should_prune = True

            if should_prune:
                # Retrieve reason/status potentially set by the heuristic function
                reason = strategy.get("prune_reason", f"Pruned by heuristic: {func.__name__}")
                status = strategy.get("max_mem_per_device", "OOM_PREDICTED_HISTORY")
                return self._apply_prune(strategy, reason, status)

        # 5. Valid Strategy
        history.append(strategy)
        return False

    # Core Helper: Centralized Pruning Side-Effects

    def _apply_prune(
        self, strategy: dict[str, Any], reason: str, status_code: str | None = None
    ) -> bool:
        """
        Handles all side effects of pruning a strategy:
        - Marks the strategy dictionary.
        - Updates internal counters.
        - Logs the event.
        - Stores the strategy for CSV export.
        """
        # Update Strategy State
        strategy["pruned"] = True
        strategy["prune_reason"] = reason
        strategy["pruned_idx"] = self.pruned_idx_counter

        if status_code:
            strategy["max_mem_per_device"] = status_code
            # Ensure compatibility with older components expecting 'max_mem'
            if strategy.get("max_mem") != "OOM":
                strategy["max_mem"] = "OOM"
            strategy["performance"] = None

        # Update Counters
        self.pruned_idx_counter += 1
        self.pruned_count += 1

        # Log
        idx = strategy.get("idx", "N/A")
        self.logger.info(f"Pruning Strategy {idx}. Reason: {reason}")
        self.logger.debug(f"Full details for pruned strategy {idx}: {strategy}")

        # Store
        self.pruned_strategies.append(strategy)

        return True

    # Logic: Architectural Checks

    def _check_architectural_validity(self, strategy: dict[str, Any]) -> tuple[bool, str]:
        """Validates structural integrity of the strategy."""
        required_keys = [
            "pipeline_model_parallel_size",
            "hetero_pipeline_layer_split",
            "hetero_process_meshes",
            "hetero_device_types",
        ]

        # Check Keys
        if not all(key in strategy for key in required_keys):
            return True, "Invalid config - Missing essential keys."

        pp_size = strategy["pipeline_model_parallel_size"]
        layer_split = strategy["hetero_pipeline_layer_split"]
        meshes = strategy["hetero_process_meshes"]
        total_layers = self.config.train.model.num_layers

        # Check Layer Splits
        if not isinstance(layer_split, list) or len(layer_split) != pp_size:
            return (
                True,
                f"Layer split length ({len(layer_split) if isinstance(layer_split, list) else 0}) != PP size ({pp_size}).",
            )

        if sum(layer_split) != total_layers:
            return True, f"Layer split sum ({sum(layer_split)}) != Total Layers ({total_layers})."

        # Check Tied Embeddings
        untie = self.config.train.model.get("untie_embeddings_and_output_weights", False)
        if not untie and meshes:
            # If tied, the first stage (Mesh 0) and last stage (Mesh -1) must have equal TP
            tp_first = meshes[0][0]
            tp_last = meshes[-1][0]
            if tp_first != tp_last:
                return (
                    True,
                    f"Tied embeddings require TP_first ({tp_first}) == TP_last ({tp_last}).",
                )

        # Check Sequence Parallel Validity
        sp = strategy.get("sequence_parallel", False)
        if meshes:
            tp_list = [m[0] for m in meshes]
            has_mixed_tp = len(set(tp_list)) > 1
            all_tp_one = all(tp == 1 for tp in tp_list)

            if has_mixed_tp and not sp:
                return True, f"Mixed TP usage {tp_list} requires sequence_parallel=True."
            if all_tp_one and sp:
                return True, "SP=True is invalid when all Meshes have TP=1."

        return False, ""

    # Logic: Hetero History Checks

    def _check_hetero_history_oom(
        self, strategy: dict[str, Any], history: list[dict[str, Any]]
    ) -> bool:
        """Checks if an identical hardware assignment + MBS failed OOM in history."""
        current_meshes = strategy.get("hetero_process_meshes")
        current_mbs = strategy.get("micro_batch_size")

        if not current_meshes:
            return False

        for item in history:
            # Only look at failed items
            if item.get("max_mem_per_device") == "OOM" or item.get("max_mem") == "OOM":
                # Check for identical hardware and batch configuration
                if (
                    item.get("hetero_process_meshes") == current_meshes
                    and item.get("micro_batch_size") == current_mbs
                ):
                    # Check if the failed item had less or equal recompute usage.
                    # If a strategy with MORE recompute OOM'd, the current one (with less) will definitely OOM.
                    if compare_by_recompute(strategy, item):
                        strategy["prune_reason"] = (
                            f"History task {item.get('idx')} OOM'd with same meshes/mbs."
                        )
                        return True
        return False

    def _corrected_prune_by_sequence_parallel(self, config, strategy, history=[]):
        """
        Hetero-aware Sequence Parallel pruning heuristic.
        Replaces the standard homogeneous SP prune which assumes uniform TP.
        """
        sp = strategy.get("sequence_parallel", False)
        meshes = strategy.get("hetero_process_meshes", [])

        # If all meshes have TP=1, SP behavior is essentially no-op or invalid,
        # usually handled by architectural check, but we return False here to be safe.
        if meshes and all(m[0] == 1 for m in meshes):
            return False

        # Search history for the same configuration but with different SP setting
        retrieval = beside(["sequence_parallel"], strategy, history)
        if not retrieval:
            return False

        for item in retrieval:
            # Heuristic: If SP=True (memory saving) failed OOM, then SP=False (higher memory) will fail.
            if item.get("sequence_parallel") and not sp:
                if item.get("max_mem") == "OOM" or item.get("max_mem_per_device") == "OOM":
                    strategy["prune_reason"] = (
                        "Pruned by SP history: SP=True OOM'd, implying SP=False will OOM."
                    )
                    return True

        return False

    # Logic: Memory Model Checks

    def _check_memory_model_utilization(self, strategy: dict[str, Any]) -> tuple[bool, str]:
        """
        Compares predicted per-mesh memory against device capacity limits.
        Returns (is_oom, reason).
        """
        memory_per_mesh_list = strategy.get("hetero_memory_model")

        # Validate input
        if not isinstance(memory_per_mesh_list, list):
            if memory_per_mesh_list == float("inf"):
                return True, "Memory calculation failed (returned infinity)."
            return False, ""  # No valid data

        # Parse Config
        mem_config = self.config.experiment.auto_tuner.get("hetero_memory_model", {})
        limits = OmegaConf.to_container(mem_config.get("gpu_memory", {}), resolve=True)

        if not limits:
            return False, ""

        util_range = mem_config.get("gpu_utilization", [0.2, 0.9])
        # Use the upper bound of the utilization range
        max_util = (
            util_range[1]
            if isinstance(util_range, (list, ListConfig)) and len(util_range) > 1
            else 0.9
        )

        device_types = strategy.get("hetero_device_types", [])

        if len(device_types) != len(memory_per_mesh_list):
            self.logger.warning(
                "Mismatch between memory list length and device types length. Skipping check."
            )
            return False, ""

        # Check each mesh individually
        for i, pred_mem in enumerate(memory_per_mesh_list):
            dtype = device_types[i]

            if pred_mem == float("inf"):
                return True, f"Mesh {i} predicted infinite memory."

            if dtype in limits:
                capacity = limits[dtype]
                threshold = capacity * max_util
                if pred_mem > threshold:
                    return True, (
                        f"Est Memory (Mesh {i}/{dtype}: {pred_mem:.0f}MB) > "
                        f"Limit ({threshold:.0f}MB, Capacity={capacity} * Util={max_util})"
                    )

        return False, ""

    # Serialization & Saving (CSV Helper)

    def _to_str(self, v: Any) -> str:
        """Safely serializes mixed types for CSV output."""
        if v is None:
            return ""
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return str(v)
            return str(v)
        if isinstance(v, (bool, str)):
            return str(v)

        try:
            if isinstance(v, (DictConfig, ListConfig)):
                v = OmegaConf.to_container(v, resolve=True)
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    def save_pruned_history(self):
        """Saves the collected list of pruned strategies to a CSV file."""
        if not self.pruned_strategies:
            return

        self.logger.info(
            f"Saving {len(self.pruned_strategies)} pruned strategies to {self.pruned_history_path}..."
        )
        try:
            df = pd.DataFrame(self.pruned_strategies)

            # Remove operational columns that are irrelevant for the report
            cols_to_drop = [
                "pruned",
                "idx",
                "max_mem",
                "max_mem_per_device",
                "performance",
                "error",
                "stopped_by_tuner",
                "elapsed_time",
                "start_time",
                "hetero_memory_model_calibrated",
            ]
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

            # Reorder columns to put reason first
            front_cols = ["pruned_idx", "prune_reason"]
            cols = front_cols + [c for c in df.columns if c not in front_cols]
            df = df.reindex(columns=cols)

            # Serialize objects
            for c in df.columns:
                df[c] = df[c].map(self._to_str)

            df.to_csv(self.pruned_history_path, index=False, quoting=csv.QUOTE_ALL)
            self.logger.info("Pruned history saved successfully.")
        except Exception as e:
            self.logger.exception(f"Failed to save pruned history: {e}")

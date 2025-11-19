import csv
import json
import logging
import os

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
    Extends the base Pruner for heterogeneous environments.

    This Pruner decides if a strategy should run and manages two lists:
    1. Valid strategies are appended to the 'history' list (passed by reference)
       to be run by the Tuner and saved by the Recorder.
    2. Pruned strategies (Type A or B) are collected internally and saved
       to 'pruned_history.csv' by this class via `save_pruned_history()`.
    """

    def __init__(self, config):
        """Initializes the HeteroPruner."""
        super().__init__(config)
        self.logger = logging.getLogger("FlagScale-AutoTuner")

        # List to store strategies that are pruned
        self.pruned_strategies = []
        # Counter for the 'pruned_idx' column in pruned_history.csv
        self.pruned_idx_counter = 1
        # Path for the separate pruned strategies log
        try:
            self.pruned_history_path = os.path.join(
                config.experiment.exp_dir, "auto_tuner", "pruned_history.csv"
            )
        except Exception:
            self.logger.error("Failed to set 'pruned_history_path'. CWD will be used.")
            self.pruned_history_path = "pruned_history.csv"

    def _assign_pruned_index_and_mark(self, strategy: dict, reason: str):
        """
        Helper to consistently mark a strategy as pruned and assign it
        a sequential 'pruned_idx' for the pruned_history.csv log.
        """
        strategy['pruned'] = True
        strategy['prune_reason'] = reason
        strategy['pruned_idx'] = self.pruned_idx_counter
        self.pruned_idx_counter += 1

    def prune(self, strategy: dict, history: list = None) -> bool:
        """
        Determines if a given strategy should be pruned.

        - If pruned (Type A or B), logs the reason, adds to
          'self.pruned_strategies', and returns True.
        - If valid, adds to 'history' list and returns False.
        """
        if history is None:
            history = []

        strategy_idx_str = f"Strategy {strategy.get('idx', 'N/A')}"

        # 1. Prune architecturally "invalid" strategies.
        if not self._is_strategy_architecturally_valid(strategy, strategy_idx_str):
            reason = strategy.get('prune_reason', 'Architecturally invalid')

            # Log full strategy dictionary ***
            self.logger.info(
                f"Pruning {strategy_idx_str}. Reason: {reason}. " f"Strategy: {strategy}"
            )

            self._assign_pruned_index_and_mark(strategy, reason)
            self.pruned_strategies.append(strategy)
            self.pruned_count += 1
            return True  # Prune

        # 2. Hetero-specific OOM prediction
        if self.prune_by_meshes_mbs_recompute_oom(strategy, history):
            reason = strategy.get('prune_reason', 'OOM predicted by history')

            #  Log full strategy dictionary ***
            self.logger.info(
                f"Pruning {strategy_idx_str}. Reason: {reason}. " f"Strategy: {strategy}"
            )

            self._assign_pruned_index_and_mark(strategy, reason)
            self.pruned_strategies.append(strategy)
            self.pruned_count += 1
            return True  # Prune

        # 3.Static memory model checks
        if "hetero_memory_model" in self.config.experiment.auto_tuner:
            if self.prune_by_memory_model_utilization(strategy, history):
                reason = strategy.get('prune_reason', 'Pruned by memory model utilization')

                #  Log full strategy dictionary ***
                self.logger.info(
                    f"Pruning {strategy_idx_str}. Reason: {reason}. " f"Strategy: {strategy}"
                )

                self._assign_pruned_index_and_mark(strategy, reason)
                self.pruned_strategies.append(strategy)
                self.pruned_count += 1
                self.pruned_by_memory_model += 1
                return True  # Prune

        # 4. History-based checks (SP, etc.)
        for func in _HISTORY_BASED_PRUNE_FUNC:
            pruned_by_history = False
            if func.__name__ == original_prune_by_sp.__name__:
                if self._corrected_prune_by_sequence_parallel(self.config, strategy, history):
                    pruned_by_history = True
            else:
                if func(self.config, strategy, history):
                    pruned_by_history = True

            if pruned_by_history:
                reason = strategy.get('prune_reason', 'Pruned by history function')

                #  Log full strategy dictionary ***
                self.logger.info(
                    f"Pruning {strategy_idx_str}. Reason: {reason}. " f"Strategy: {strategy}"
                )

                self._assign_pruned_index_and_mark(strategy, reason)
                self.pruned_strategies.append(strategy)
                self.pruned_count += 1
                return True  # Prune

        # 5. Passed all checks
        history.append(strategy)
        return False  # Do not prune

    def _to_str(self, v):
        """Serializes a Python value into a string suitable for CSV storage."""
        if isinstance(v, (ListConfig, DictConfig)):
            try:
                plain_v = OmegaConf.to_container(v, resolve=True)
                return json.dumps(plain_v, ensure_ascii=False)
            except Exception:
                return str(v)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return str(v)
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    def save_pruned_history(self):
        """
        Saves all collected pruned strategies to 'pruned_history.csv'.
        """
        if not self.pruned_strategies:
            self.logger.info("No strategies were pruned, 'pruned_history.csv' will not be created.")
            return

        self.logger.info(
            f"Saving {len(self.pruned_strategies)} pruned strategies to {self.pruned_history_path}..."
        )
        try:
            df = pd.DataFrame(self.pruned_strategies)
        except Exception as e:
            self.logger.error(f"Error creating DataFrame for pruned history: {e}. Cannot save.")
            return

        # Reorder columns
        cols = df.columns.tolist()
        try:
            if "pruned_idx" in cols:
                cols.insert(0, cols.pop(cols.index("pruned_idx")))
            if "prune_reason" in cols:
                cols.insert(1, cols.pop(cols.index("prune_reason")))

            # Added all irrelevant run-time columns to drop list
            cols_to_drop = [
                'pruned',
                'idx',
                'max_mem',
                'max_mem_per_device',
                'performance',
                'error',
                'stopped_by_tuner',
                'elapsed_time',
                'start_time',
                'hetero_memory_model_calibrated',
            ]

            final_cols = [c for c in cols if c not in cols_to_drop]
            df = df.reindex(columns=final_cols)
        except Exception as e:
            self.logger.warning(f"Error reordering/dropping columns for pruned history: {e}.")
            df = df.drop(
                columns=[col for col in cols_to_drop if col in df.columns], errors='ignore'
            )

        # Apply serialization
        try:
            for c in df.columns:
                df[c] = df[c].map(self._to_str)
        except Exception:
            pass  # Try to save raw

        # Save to CSV
        try:
            df.to_csv(self.pruned_history_path, index=False, escapechar="\\", quoting=csv.QUOTE_ALL)
            self.logger.info(
                f"Successfully saved {len(self.pruned_strategies)} pruned records to {self.pruned_history_path}"
            )
        except Exception as e:
            self.logger.error(f"Error saving pruned history to CSV: {e}")

    # --- (Omitted the unchanged pruning functions for brevity) ---
    # --- (_mark_as_pruned, prune_by_meshes_mbs_recompute_oom, ...) ---
    # --- (_corrected_prune_by_sequence_parallel, _get_device_type_for_stage, ...) ---
    # --- (prune_by_memory_model_utilization, _is_strategy_architecturally_valid) ---

    def _mark_as_pruned(self, strategy: dict, reason: str):
        strategy['pruned'] = True
        strategy['prune_reason'] = reason

    def prune_by_meshes_mbs_recompute_oom(self, strategy: dict, history: list) -> bool:
        current_meshes = strategy.get('hetero_process_meshes')
        current_mbs = strategy.get('micro_batch_size')
        if not current_meshes:
            return False
        for item in history:
            if item.get('max_mem_per_device') == 'OOM' or item.get('max_mem') == 'OOM':
                item_meshes = item.get('hetero_process_meshes')
                if item_meshes == current_meshes:
                    item_mbs = item.get('micro_batch_size')
                    if item_mbs == current_mbs:
                        if compare_by_recompute(strategy, item):
                            reason = (
                                f"Identical meshes/mbs OOM: History task {item.get('idx', 'N/A')} "
                                f"OOM'd with same meshes/mbs and <= demanding recompute."
                            )
                            self.logger.debug(f"Strategy {strategy.get('idx', 'N/A')}: {reason}")
                            self._mark_as_pruned(strategy, reason)
                            strategy['max_mem_per_device'] = 'OOM_PREDICTED_MESH_MBS'
                            strategy['performance'] = None
                            return True
        return False

    def _corrected_prune_by_sequence_parallel(self, config, strategy, history=[]):
        sequence_parallel = strategy.get("sequence_parallel", False)
        all_tp_are_one = False
        meshes = strategy.get("hetero_process_meshes")
        if meshes:
            all_tp_are_one = all(mesh[0] == 1 for mesh in meshes)
        if all_tp_are_one:
            return False
        retrieval = beside(["sequence_parallel"], strategy, history)
        if retrieval:
            for item in retrieval:
                if (
                    item.get("sequence_parallel")
                    and item.get("performance") is not None
                    and not sequence_parallel
                ):
                    reason = "Pruned by sequence_parallel performance (TP>1)"
                    self.logger.debug(f"Strategy {strategy.get('idx', 'N/A')}: {reason}.")
                    self._mark_as_pruned(strategy, reason)
                    strategy["performance"] = item["performance"]
                    strategy["max_mem_per_device"] = item.get("max_mem_per_device") or item.get(
                        "max_mem"
                    )
                    return True
                if (
                    item.get("sequence_parallel")
                    and (item.get("max_mem_per_device") == "OOM" or item.get("max_mem") == "OOM")
                    and not sequence_parallel
                ):
                    reason = "Pruned by sequence_parallel memory (TP>1)"
                    self.logger.debug(f"Strategy {strategy.get('idx', 'N/A')}: {reason}.")
                    self._mark_as_pruned(strategy, reason)
                    strategy["max_mem_per_device"] = "OOM_PREDICTED_SP"
                    strategy["performance"] = None
                    return True
        return False

    def prune_by_memory_model_utilization(self, strategy: dict, history: list = None) -> bool:
        """
        Prunes based on the theoretical memory model and a
        *maximum* utilization threshold (e.g., 0.8) from the config.

        This version reads a *per-mesh* memory list.
        """
        # 1. Get the *per-mesh* memory list (e.g., [62000, 26000])
        memory_per_mesh_list = strategy.get("hetero_memory_model")

        if memory_per_mesh_list is None:
            return False

        # Handle calculation failure (inf)
        if isinstance(memory_per_mesh_list, float) and memory_per_mesh_list == float('inf'):
            reason = "Pruned due to memory calculation failure (inf)."
            self.logger.debug(f"Strategy {strategy.get('idx', 'N/A')}: {reason}")
            self._mark_as_pruned(strategy, reason)
            strategy['max_mem_per_device'] = 'OOM_CALC_FAIL'
            strategy['performance'] = None
            return True
        if not isinstance(memory_per_mesh_list, list):
            return False

        # Get device memory limits
        memory_limits_config = self.config.experiment.auto_tuner.get("hetero_memory_model", {}).get(
            "gpu_memory", None
        )
        if not isinstance(memory_limits_config, (dict, DictConfig)):
            self.logger.warning(
                "config...hetero_memory_model.gpu_memory is not a dict. Cannot perform hetero prune."
            )
            return False
        memory_limits_dict = OmegaConf.to_container(memory_limits_config, resolve=True)

        # Get device types for each mesh (e.g., ['A800', 'mlu590'])
        device_types = strategy.get("hetero_device_types")
        if not isinstance(device_types, list) or len(device_types) != len(memory_per_mesh_list):
            self.logger.error(
                f"Strategy {strategy.get('idx', 'N/A')}: Mismatch between memory list ({len(memory_per_mesh_list)}) and device types ({len(device_types)}). Cannot prune."
            )
            return False

        # Get *max* utilization threshold (e.g., [0.2, 0.8])
        util_range = self.config.experiment.auto_tuner.get("hetero_memory_model", {}).get(
            "gpu_utilization", [0.2, 0.8]
        )
        try:
            if not isinstance(util_range, (list, ListConfig)) or len(util_range) < 2:
                raise ValueError("Expected a list of 2 floats.")
            max_util = float(util_range[1])
        except (TypeError, ValueError, IndexError):
            self.logger.warning(
                f"Invalid 'gpu_utilization' range {util_range}. Using default max_util [0.8]."
            )
            max_util = 0.8

        # 2. Check each *mesh* against its device's capacity
        for mesh_idx, theoretical_mem_mb in enumerate(memory_per_mesh_list):

            if theoretical_mem_mb == float('inf'):
                reason = f"Pruned due to memory calculation failure (Mesh {mesh_idx} == inf)."
                self.logger.debug(f"Strategy {strategy.get('idx', 'N/A')}: {reason}")
                self._mark_as_pruned(strategy, reason)
                strategy['max_mem_per_device'] = 'OOM_CALC_FAIL'
                strategy['performance'] = None
                return True

            # 3. Get the device type for this mesh
            device_type = device_types[mesh_idx]
            if device_type not in memory_limits_dict:
                self.logger.warning(
                    f"No memory limit set for device type '{device_type}' in config. Skipping prune check."
                )
                continue

            # 4. Get the limit for this device
            device_total_capacity_mb = memory_limits_dict[device_type]
            upper_bound_mb = device_total_capacity_mb * max_util

            # 5. Check OOM (Upper Bound)
            if theoretical_mem_mb > upper_bound_mb:
                reason = (
                    f"Pruned by Theoretical Memory (Mesh {mesh_idx}, Device {device_type}): "
                    f"Est ({theoretical_mem_mb:.0f} MB) > Limit ({upper_bound_mb:.0f} MB)"
                )
                self.logger.debug(f"Strategy {strategy.get('idx', 'N/A')}: {reason}")
                self._mark_as_pruned(strategy, reason)
                strategy['max_mem_per_device'] = 'OOM_PREDICTED_UTIL_HIGH'
                strategy['performance'] = None
                return True

        return False  # Not pruned

    def _is_strategy_architecturally_valid(self, strategy: dict, strategy_idx: str) -> bool:
        required_keys = [
            "pipeline_model_parallel_size",
            "hetero_pipeline_layer_split",
            "hetero_process_meshes",
            "hetero_device_types",
        ]
        if not all(key in strategy for key in required_keys):
            reason = "Pruned (Type A): Invalid config - Missing essential keys."
            self.logger.debug(f"{strategy_idx}: {reason}")
            self._mark_as_pruned(strategy, reason)
            return False
        pp_size = strategy["pipeline_model_parallel_size"]
        layer_split = strategy["hetero_pipeline_layer_split"]
        meshes = strategy["hetero_process_meshes"]
        if not isinstance(layer_split, list) or len(layer_split) != pp_size:
            reason = f"Pruned (Type A): Invalid config - Layer split length ({len(layer_split)}) != PP size ({pp_size})."
            self.logger.debug(f"{strategy_idx}: {reason}")
            self._mark_as_pruned(strategy, reason)
            return False
        if sum(layer_split) != self.config.train.model.num_layers:
            reason = f"Pruned (Type A): Invalid config - Layer split sum ({sum(layer_split)}) != Total Layers ({self.config.train.model.num_layers})."
            self.logger.debug(f"{strategy_idx}: {reason}")
            self._mark_as_pruned(strategy, reason)
            return False
        untie_embeddings = self.config.train.model.get("untie_embeddings_and_output_weights", False)
        if not untie_embeddings and meshes:
            if meshes[0][0] != meshes[-1][0]:
                reason = f"Pruned (Type A): Invalid config - Embeddings are tied, but TP-First ({meshes[0][0]}) != TP-Last({meshes[-1][0]})."
                self.logger.debug(f"{strategy_idx}: {reason}")
                self._mark_as_pruned(strategy, reason)
                return False
        sp = strategy.get("sequence_parallel", False)
        if not meshes:
            return True
        tp_list = [mesh[0] for mesh in meshes]
        all_tp_are_one = all(tp == 1 for tp in tp_list)
        tps_are_mixed = len(set(tp_list)) > 1
        if tps_are_mixed and not sp:
            reason = f"Pruned (Type A): Invalid config - TPs are mixed ({tp_list}), which requires sequence_parallel=True."
            self.logger.debug(f"{strategy_idx}: {reason}")
            self._mark_as_pruned(strategy, reason)
            return False
        if all_tp_are_one and sp:
            reason = f"Pruned (Type A): Invalid config - SP=True but ALL meshes have TP=1."
            self.logger.debug(f"{strategy_idx}: {reason}")
            self._mark_as_pruned(strategy, reason)
            return False
        return True

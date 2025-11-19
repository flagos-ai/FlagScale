import csv
import json
import logging
import os
import re
import subprocess

import numpy as np
import pandas as pd

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.runner.auto_tuner.record.recorder import Recorder
from flagscale.runner.utils import parse_hostfile


class HeteroRecorder(Recorder):
    """
    Recorder for heterogeneous tasks.

    This class extends the base Recorder to:
    1. Override memory gathering to report 'max_mem_per_device'.
    2. Save *only* the run strategies it receives to 'history.csv'.

    It assumes the 'history' list it receives from the Tuner only contains
    strategies that were approved for running by the Pruner.
    """

    def __init__(self, config):
        """Initializes the HeteroRecorder."""
        super().__init__(config)
        self.host_to_type_map = {}
        self.memory_patterns = {}
        self.default_mem_pattern_str = "max reserved"

        # 1. Load device-specific memory grep patterns
        try:
            if "hetero_memory_model" in self.config.experiment.auto_tuner:
                patterns_dict = self.config.experiment.auto_tuner.hetero_memory_model.get(
                    "memory_grep_patterns", None
                )
                if isinstance(patterns_dict, (dict, DictConfig)):
                    self.memory_patterns = OmegaConf.to_container(patterns_dict, resolve=True)
                    self.logger.info(
                        f"Loaded device-specific memory patterns: {self.memory_patterns}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to read 'memory_grep_patterns' from config: {e}.")

        # 2. Build host -> device type map from hostfile
        hostfile_path = self.config.experiment.runner.get("hostfile", None)
        if hostfile_path and os.path.exists(hostfile_path):
            try:
                resources = parse_hostfile(hostfile_path)
                if resources:
                    self.host_to_type_map = {
                        host: res.get('type', 'default') for host, res in resources.items()
                    }
                self.logger.info(f"Loaded host-to-type map: {self.host_to_type_map}")
            except Exception as e:
                self.logger.error(f"Failed to parse hostfile {hostfile_path}: {e}")
        else:
            self.logger.warning(
                f"Hostfile not found at {hostfile_path}. Cannot map hosts to "
                "device types for memory reporting. Will report per-host memory."
            )

    def record(self, task, strategy):
        """
        Records the task results using hetero-aware memory gathering.
        (Overrides the base `record` method)
        """
        self.cur_strategy = strategy
        peformance_path, host_path = self.get_all_performance_and_host_paths(task)
        errors = self.grep_error(host_path)

        if errors:
            if "OOM" in errors:
                strategy["performance"] = None
                oom_dict = {
                    dtype: "OOM" for dtype in strategy.get("hetero_device_types", ["unknown"])
                }
                strategy["max_mem_per_device"] = oom_dict
                strategy["error"] = "|".join(list(errors))

            elif self.cur_strategy.get("stopped_by_tuner", False):
                performace = self.grep_performance(peformance_path, self.metric)
                strategy["performance"] = performace
                strategy["max_mem_per_device"] = self.grep_max_memory(host_path)
                strategy["error"] = None

            else:  # Task failed with other errors
                performace = self.grep_performance(peformance_path, self.metric)
                strategy["performance"] = performace
                strategy["max_mem_per_device"] = self.grep_max_memory(host_path)
                strategy["error"] = "|".join(list(errors))

        else:
            # Task ended properly
            strategy["max_mem_per_device"] = self.grep_max_memory(host_path)
            performace = self.grep_performance(peformance_path, self.metric)
            strategy["performance"] = performace
            strategy["error"] = None

        if (
            "airs_switch" in self.config.experiment.auto_tuner.platform
            and self.config.experiment.auto_tuner.platform.airs_switch
            and strategy.get("performance")
        ):
            self.pass_back_to_platform(strategy)

        if strategy.get("error"):
            error_string = str(strategy["error"])
            sanitized_error = error_string.replace("\n", " ").replace("\r", "")
            strategy["error"] = sanitized_error

    def grep_max_memory(self, path, pattern=None):
        """
        Reads log files, finds the peak memory *per host*, and aggregates
        them *per device type* using the host-to-type map.
        (Overrides the base `grep_max_memory` method)
        """
        details_path = os.path.join(path, "details")
        if not os.path.exists(details_path):
            self.logger.warning(
                f"Log 'details' path not found: {details_path}. Cannot find host logs."
            )
            return {}

        max_memory_per_host = {}
        for host_dir in os.listdir(details_path):
            if not host_dir.startswith("host_"):
                continue
            try:
                parts = host_dir.split('_', 2)
                if len(parts) < 3:
                    continue
                hostname = parts[2]
            except Exception:
                continue
            device_type = self.host_to_type_map.get(hostname, "unknown_host")
            pattern_str = self.memory_patterns.get(
                device_type, self.memory_patterns.get("default", self.default_mem_pattern_str)
            )
            memory_pattern = pattern_str + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + pattern_str
            # self.logger.debug(f"Grepping host: {hostname} (Type: {device_type}) using pattern: '{pattern_str}'")
            host_max_mem = 0.0
            host_full_path = os.path.join(details_path, host_dir)
            for root, dirs, files in os.walk(host_full_path):
                for file in files:
                    if file != "stdout.log":
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "rb") as f:
                            for line_bytes in f:
                                try:
                                    line = line_bytes.decode("utf-8")
                                except UnicodeDecodeError:
                                    continue
                                memory = re.findall(memory_pattern, line, re.IGNORECASE)
                                if memory:
                                    for item in memory[0]:
                                        try:
                                            value = float(item)
                                            if value > host_max_mem:
                                                host_max_mem = value
                                            break
                                        except (ValueError, TypeError):
                                            continue
                    except Exception as e:
                        self.logger.warning(f"Could not read log file {file_path}: {e}")
            if host_max_mem > 0:
                max_memory_per_host[hostname] = host_max_mem
            else:
                self.logger.warning(
                    f"No memory pattern '{pattern_str}' found for host {hostname} (Type: {device_type})."
                )

        max_mem_per_type = {}
        if not self.host_to_type_map:
            self.logger.warning("No host-to-type map found. Returning memory per host.")
            return max_memory_per_host
        for host, max_mem in max_memory_per_host.items():
            device_type = self.host_to_type_map.get(host, "unknown_host")
            if max_mem > max_mem_per_type.get(device_type, 0.0):
                max_mem_per_type[device_type] = max_mem
        self.logger.debug(f"Max memory found per host: {max_memory_per_host}")
        self.logger.debug(f"Aggregated max memory per device type: {max_mem_per_type}")
        if self.cur_strategy and 'idx' in self.cur_strategy:
            self.logger.info(
                f"task_{self.cur_strategy['idx']} max_memory_per_device: {max_mem_per_type}"
            )
        return max_mem_per_type

    def to_str(self, v):
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

    def save(self, history: list):
        """
        Saves the processed history of *run* strategies to 'history.csv'.

        This method relies on the inherited base `sort()` method, which
        filters out any `pruned=True` strategies. This ensures 'history.csv'
        only contains strategies that were actually run.
        """
        if not history:
            self.logger.warning("No *run* strategies found. 'history.csv' will be empty.")
            try:
                df_empty = pd.DataFrame()
                df_empty.to_csv(self.path, index=False, escapechar="\\")
            except Exception as e:
                self.logger.error(f"Error creating empty history file: {e}")
            return

        # 1. Sort (using the inherited base sort method, which filters 'pruned')
        try:
            processed_history = self.sort(history)
        except Exception as e:
            self.logger.error(f"Error sorting run history: {e}. Saving unsorted.")
            processed_history = [s for s in history if not s.get("pruned", False)]

        if not processed_history:
            self.logger.warning(
                "Run history is empty after filtering. 'history.csv' will be empty."
            )
            pd.DataFrame().to_csv(self.path, index=False, escapechar="\\")
            return

        # 2. Create DataFrame
        try:
            df = pd.DataFrame(processed_history)
        except Exception as e:
            self.logger.error(f"Error creating DataFrame from run history: {e}. Cannot save.")
            return

        # 3. Reorder 'idx' and drop irrelevant columns
        cols = df.columns.tolist()
        columns_to_drop = []
        if "idx" in cols:
            try:
                cols.insert(0, cols.pop(cols.index("idx")))
            except ValueError:
                self.logger.warning("Could not reorder 'idx' column.")
        if "stopped_by_tuner" in cols:
            columns_to_drop.append("stopped_by_tuner")
        if "max_mem" in cols and "max_mem_per_device" in cols:
            columns_to_drop.append("max_mem")

        # Clean up pruning columns (irrelevant for run history)
        columns_to_drop.extend(
            ['pruned', 'prune_reason', 'pruned_idx', 'hetero_memory_model_calibrated']
        )
        final_cols = [c for c in cols if c not in columns_to_drop]

        try:
            df = df.reindex(columns=final_cols)
        except Exception as e:
            self.logger.warning(f"Error reordering/dropping columns: {e}.")
            df = df.drop(
                columns=[col for col in columns_to_drop if col in df.columns], errors='ignore'
            )

        # 4. Apply 'to_str'
        try:
            for c in df.columns:
                df[c] = df[c].map(self.to_str)
        except Exception as e:
            self.logger.error(f"Error applying string conversion: {e}. Attempting to save raw.")

        # 5. Save
        try:
            df.to_csv(self.path, index=False, escapechar="\\", quoting=csv.QUOTE_ALL)
            self.logger.info(f"Saved {len(df)} run records to {self.path}")
        except Exception as e:
            self.logger.error(f"Error saving run history to CSV: {e}")

import csv
import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.runner.auto_tuner.record.recorder import Recorder
from flagscale.runner.utils import parse_hostfile


class HeteroRecorder(Recorder):
    """
    Recorder for heterogeneous tasks.

    Extends base Recorder to:
    1. Report 'max_mem_per_device' aggregated by device type.
    2. Save only executing strategies to 'history.csv' (filtering out pruned ones).
    """

    def __init__(self, config):
        super().__init__(config)
        self.host_to_type_map = {}
        self.memory_patterns = {}
        self.default_mem_pattern_str = "max reserved"

        # 1. Load device-specific memory grep patterns
        self._load_memory_patterns()

        # 2. Build host -> device type map
        self._build_host_map()

    def _load_memory_patterns(self):
        """Loads memory grep patterns from config safely."""
        try:
            auto_tuner_cfg = getattr(self.config.experiment, "auto_tuner", {})
            hetero_mem_cfg = getattr(auto_tuner_cfg, "hetero_memory_model", {})
            patterns = hetero_mem_cfg.get("memory_grep_patterns", None)

            if isinstance(patterns, (dict, DictConfig)):
                self.memory_patterns = OmegaConf.to_container(patterns, resolve=True)
                self.logger.info(f"Loaded memory patterns: {self.memory_patterns}")
        except Exception as e:
            self.logger.error(f"Failed to load memory patterns: {e}")

    def _build_host_map(self):
        """Parses hostfile to map hostnames to device types."""
        hostfile_path = self.config.experiment.runner.get("hostfile", None)
        if hostfile_path and os.path.exists(hostfile_path):
            try:
                resources = parse_hostfile(hostfile_path)
                if resources:
                    self.host_to_type_map = {
                        host: res.get("type", "default") for host, res in resources.items()
                    }
                self.logger.info(f"Loaded host-to-type map: {self.host_to_type_map}")
            except Exception as e:
                self.logger.error(f"Failed to parse hostfile {hostfile_path}: {e}")
        else:
            self.logger.warning(
                f"Hostfile not found at {hostfile_path}. Reporting will be per-host."
            )

    def record(self, task, strategy):
        """
        Records task results with hetero-aware memory gathering.
        """
        self.cur_strategy = strategy
        performance_path, host_path = self.get_all_performance_and_host_paths(task)

        # Gather Metrics
        max_mem = self.grep_max_memory(host_path)
        performance = self.grep_performance(performance_path, self.metric)
        errors = self.grep_error(host_path)

        # Update Strategy
        strategy["max_mem_per_device"] = max_mem
        strategy["performance"] = performance

        # Handle Errors
        if errors:
            # OOM Special Handling
            if "OOM" in errors:
                strategy["performance"] = None
                # Mark all devices as OOM for safety if generic OOM detected
                strategy["max_mem_per_device"] = {
                    dtype: "OOM" for dtype in strategy.get("hetero_device_types", ["unknown"])
                }

            # Format error string
            err_str = "|".join(list(errors))
            strategy["error"] = err_str.replace("\n", " ").replace("\r", "")
        else:
            strategy["error"] = None

        # Platform Callback (if enabled)
        platform_cfg = self.config.experiment.auto_tuner.get("platform", {})
        if platform_cfg.get("airs_switch", False) and strategy.get("performance"):
            self.pass_back_to_platform(strategy)

    def grep_max_memory(self, path, pattern=None) -> dict[str, Any]:
        """
        Scans logs for peak memory usage, aggregating by device type.
        Returns: Dict[device_type, max_memory_float]
        """
        details_path = os.path.join(path, "details")
        if not os.path.exists(details_path):
            self.logger.warning(f"Log details not found at {details_path}")
            return {}

        max_memory_per_host = {}

        # Iterate over host directories
        for host_dir in os.listdir(details_path):
            if not host_dir.startswith("host_"):
                continue

            try:
                parts = host_dir.split("_", 2)
                if len(parts) < 3:
                    continue
                hostname = parts[2]
            except IndexError:
                continue

            # Determine grep pattern for this host
            device_type = self.host_to_type_map.get(hostname, "unknown_host")
            pattern_str = self.memory_patterns.get(
                device_type, self.memory_patterns.get("default", self.default_mem_pattern_str)
            )

            # Regex: "pattern: 1234.56" or "1234.56 pattern"
            regex = rf"{re.escape(pattern_str)}:* *(\d+(?:\.\d*)?)|(\d+(?:\.\d*)?) *{re.escape(pattern_str)}"

            host_max_mem = 0.0
            host_full_path = os.path.join(details_path, host_dir)

            # Walk through logs for this host
            for root, _, files in os.walk(host_full_path):
                if "stdout.log" not in files:
                    continue

                file_path = os.path.join(root, "stdout.log")
                try:
                    with open(file_path, "rb") as f:
                        for line_bytes in f:
                            try:
                                line = line_bytes.decode("utf-8", errors="ignore")
                                matches = re.findall(regex, line, re.IGNORECASE)
                                for match in matches:
                                    # match tuple can be ('123', '') or ('', '123')
                                    val_str = match[0] or match[1]
                                    if val_str:
                                        val = float(val_str)
                                        if val > host_max_mem:
                                            host_max_mem = val
                            except ValueError:
                                continue
                except Exception as e:
                    self.logger.warning(f"Error reading {file_path}: {e}")

            if host_max_mem > 0:
                max_memory_per_host[hostname] = host_max_mem

        # Aggregate by Device Type
        if not self.host_to_type_map:
            return max_memory_per_host

        max_mem_per_type = {}
        for host, mem in max_memory_per_host.items():
            dtype = self.host_to_type_map.get(host, "unknown_host")
            if mem > max_mem_per_type.get(dtype, 0.0):
                max_mem_per_type[dtype] = mem

        self.logger.debug(f"Aggregated Max Memory: {max_mem_per_type}")
        return max_mem_per_type

    @staticmethod
    def _to_str(v: Any) -> str:
        """Safely serializes values for CSV."""
        if v is None:
            return ""

        # OmegaConf types
        if isinstance(v, (DictConfig, ListConfig)):
            try:
                v = OmegaConf.to_container(v, resolve=True)
            except Exception:
                return str(v)

        # Numbers
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return str(v)
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return str(v)

        if isinstance(v, bool):
            return str(v)
        if isinstance(v, str):
            return v

        # JSON fallback
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    def save(self, history: list):
        """
        Saves executed strategies to 'history.csv'.
        Filters out pruned strategies using the base sort method.
        """
        if not history:
            self._save_empty_csv()
            return

        # 1. Filter & Sort
        try:
            # Use base sort if available, else manual filter
            processed_history = self.sort(history)
        except Exception as e:
            self.logger.error(f"Sort failed: {e}. Saving unsorted.")
            processed_history = [s for s in history if not s.get("pruned", False)]

        if not processed_history:
            self.logger.warning("No valid run history found.")
            self._save_empty_csv()
            return

        # 2. Create DataFrame
        try:
            df = pd.DataFrame(processed_history)
        except Exception as e:
            self.logger.error(f"DataFrame creation failed: {e}")
            return

        # 3. Column Management
        # Drop irrelevant columns
        drop_cols = [
            "pruned",
            "prune_reason",
            "pruned_idx",
            "hetero_memory_model_calibrated",
            "stopped_by_tuner",
            "max_mem",
        ]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # Reorder 'idx' to front
        if "idx" in df.columns:
            cols = ["idx"] + [c for c in df.columns if c != "idx"]
            df = df.reindex(columns=cols)

        # 4. Serialize
        for c in df.columns:
            df[c] = df[c].map(self._to_str)

        # 5. Save
        try:
            df.to_csv(self.path, index=False, escapechar="\\", quoting=csv.QUOTE_ALL)
            self.logger.info(f"Saved {len(df)} records to {self.path}")
        except Exception as e:
            self.logger.error(f"Failed to save history CSV: {e}")

    def _save_empty_csv(self):
        """Helper to create an empty CSV file."""
        try:
            pd.DataFrame().to_csv(self.path, index=False)
        except Exception as e:
            self.logger.error(f"Failed to save empty CSV: {e}")

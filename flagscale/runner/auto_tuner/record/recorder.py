import json
import logging
import os
import re
import subprocess

import numpy as np
import pandas as pd


class Recorder:
    def __init__(self, config):
        self.config = config
        self.path = os.path.join(config.experiment.exp_dir, "auto_tuner", "history.csv")
        # Metric to grep in the last rank of last node log file
        if (
            "auto_tuner" in self.config.experiment
            and "performance" in self.config.experiment.auto_tuner
        ):
            self.metric = self.config.experiment.auto_tuner.performance.get(
                "name", "elapsed time per iteration \(ms\):"
            )
        else:
            self.metric = "elapsed time per iteration \(ms\):"

        # Sort order of performance, order just in [ascend, and descend], default ascend
        if (
            "auto_tuner" in self.config.experiment
            and "performance" in self.config.experiment.auto_tuner
        ):
            self.sorted_order = self.config.experiment.auto_tuner.performance.get("order", "ascend")
        else:
            self.sorted_order = "ascend"

        self.logger = logging.getLogger("FlagScale-AutoTuner")
        self.cur_strategy = None

    def record(self, task, strategy):
        """Record the performance and max memory of task"""
        self.cur_strategy = strategy
        performance_path, host_path = self.get_all_performance_and_host_paths(task)

        errors = self.grep_error(host_path)
        if errors:
            # If OOM in errors, the task must fail.
            if "OOM" in errors:
                strategy["performance"] = None
                strategy["max_mem"] = "OOM"
                strategy["error"] = "|".join(list(errors))

            # If task is stopped by autotuner, task may not be failed,just hang or too slow.
            elif self.cur_strategy.get("stopped_by_tuner", False):
                performance = self.grep_performance(performance_path, self.metric)
                strategy["performance"] = performance
                strategy["max_mem"] = self.grep_max_memory(host_path)
                strategy["error"] = None

            # Task failed and the code may have logical errors
            else:
                # HACK: record the performance when task exits in the last allreduce of training
                performance = self.grep_performance(performance_path, self.metric)
                strategy["performance"] = performance
                strategy["max_mem"] = self.grep_max_memory(host_path)
                strategy["error"] = "|".join(list(errors))

        # Task ended properly
        else:
            strategy["max_mem"] = self.grep_max_memory(host_path)
            performance = self.grep_performance(performance_path, self.metric)
            strategy["performance"] = performance
            strategy["error"] = None

        # Pass back to platform if need
        if (
            "airs_switch" in self.config.experiment.auto_tuner.platform
            and self.config.experiment.auto_tuner.platform.airs_switch
            and strategy["performance"]
        ):
            self.pass_back_to_platform(strategy)

    def pass_back_to_platform(self, strategy):
        gbs = int(self.config.train.model.global_batch_size)
        seq_len = int(self.config.train.model.seq_length)
        throughput = gbs * seq_len / (strategy["performance"] / 1000)
        day = round(
            self.config.train.model.train_samples * seq_len / (throughput * 60 * 60 * 24), 2
        )
        command = [
            "airsctl job performance",
            "-D",
            f"{strategy['data_parallel_size']}",
            "--distributed_optimizer",
            (
                f"{strategy['use_distributed_optimizer']}"
                if strategy["use_distributed_optimizer"] is not None
                else "False"
            ),
            "-E",
            f"{strategy['expert_model_parallel_size']}",
            "-C",
            f"{strategy['context_parallel_size']}",
            "-M",
            f"{strategy['micro_batch_size']}",
            "-L",
            f"{strategy['pipeline_model_parallel_size']}",
            "-G",
            (
                f"{strategy['recompute_granularity']}"
                if strategy["recompute_granularity"]
                else "None"
            ),
            "-R",
            (f"{strategy['recompute_method']}" if strategy["recompute_granularity"] else "None"),
            "-N",
            (f"{strategy['recompute_num_layers']}" if strategy["recompute_num_layers"] else "0"),
            "-S",
            (
                f"{strategy['sequence_parallel']}"
                if strategy["sequence_parallel"] is not None
                else "False"
            ),
            "-T",
            f"{strategy['tensor_model_parallel_size']}",
            "-V",
            (
                f"{strategy['num_layers_per_virtual_pipeline_stage']}"
                if strategy["num_layers_per_virtual_pipeline_stage"]
                else "0"
            ),
            "--throughput",
            f"{int(throughput)}",
            "--day",
            f"{day}",
            "--time",
            f"{int(strategy['performance'])}",
        ]
        joined_command = " ".join(command)
        self.logger.info(f"Pass back to platform: {joined_command}")
        try:
            subprocess.run(joined_command, shell=True, check=True)
        except Exception as e:
            self.logger.info(f"Failed to pass back to platform: {e}")

    def grep_max_memory(self, path, pattern="max reserved"):
        """Read the log file and return the max memory."""
        if not os.path.exists(path):
            raise ValueError(f"The path do not exist: {path}")
        memory_pattern = pattern + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + pattern
        max_memory = None
        for item in os.listdir(path):
            if not item.startswith("host_") and not item.endswith(".output"):
                continue
            file_path = os.path.join(path, item)
            with open(file_path, "rb") as f:
                for _ in f:
                    try:
                        line = _.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    memory = re.findall(memory_pattern, line, re.IGNORECASE)
                    if memory:
                        value = None
                        for item in memory[0]:
                            try:
                                value = float(item)
                                if max_memory is None:
                                    max_memory = value
                                if value > max_memory:
                                    max_memory = value
                                break
                            except:
                                continue
                        assert value is not None, "Can't grep the max memory"
        self.logger.info(f"task_{self.cur_strategy['idx']} max_memory: {max_memory}")
        return max_memory

    def get_all_performance_and_host_paths(self, task):
        """Get all log paths of task across all hosts and ranks."""
        logs = os.path.join(task.experiment.exp_dir, "logs")
        details = os.path.join(logs, "details")

        if not os.path.exists(details):
            raise ValueError("The task detail folder does not exist.")

        all_log_paths = []

        for host_item in os.listdir(details):
            if not host_item.startswith("host_"):
                continue

            host_path = os.path.join(details, host_item)

            for output_item in os.listdir(host_path):
                output_path = os.path.join(host_path, output_item)

                for sub_item in os.listdir(output_path):
                    sub_path = os.path.join(output_path, sub_item)

                    attempt_path = os.path.join(sub_path, "attempt_0")
                    if not os.path.exists(attempt_path):
                        continue

                    for rank_item in os.listdir(attempt_path):
                        rank_path = os.path.join(attempt_path, rank_item)

                        if not rank_item.isdigit():
                            continue

                        log_path = os.path.join(rank_path, "stdout.log")
                        if os.path.exists(log_path):
                            all_log_paths.append(log_path)
        return all_log_paths, logs

    def grep_performance(self, paths, pattern="elapsed time per iteration \(ms\):"):
        """Read the log file and return the performance."""
        metric_pattern = pattern + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + pattern
        if not paths:
            return None
        performance = []
        for path in paths:
            if not path or not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                for _ in f:
                    try:
                        line = _.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    metric = re.findall(metric_pattern, line)
                    if metric:
                        value = None
                        for item in metric[0]:
                            try:
                                value = float(item)
                                performance.append(value)
                                break
                            except:
                                continue
            if performance:
                break
        if not performance:
            self.logger.info(f"task_{self.cur_strategy['idx']} performance: {None}")
            return None
        if len(performance) == 1:
            self.logger.info(f"task_{self.cur_strategy['idx']} performance: {performance[0]} ms")
            return round(performance[0], 3)
        else:
            average = sum(performance[1:]) / (len(performance) - 1)
            self.logger.info(f"task_{self.cur_strategy['idx']} performance: {average} ms")
            return round(average, 3)

    def grep_error(self, path, pattern="Error:"):
        """Read the log file and return the error"""
        if not os.path.exists(path):
            raise ValueError(f"The path do not exist: {path}")

        oom = "out of memory"
        errors_info = set()
        for item in os.listdir(path):
            if not item.startswith("host_") and not item.endswith(".output"):
                continue
            file_path = os.path.join(path, item)
            with open(file_path, "rb") as f:
                for _ in f:
                    try:
                        line = _.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    error = re.findall(pattern, line, re.IGNORECASE)
                    if error:
                        if oom in line:
                            errors_info.add("OOM")
                            errors_info.add(line)
                        else:
                            errors_info.add(line)

        self.logger.info(f"task_{self.cur_strategy['idx']} error: {errors_info}")
        return errors_info

    def sort(self, history):
        no_pruned_history = []
        for strategy in history:
            if not strategy.get("pruned", False):
                no_pruned_history.append(strategy)

        sorted_history = None
        if self.sorted_order == "ascend":
            sorted_history = sorted(
                no_pruned_history,
                key=lambda x: (x["performance"] if x["performance"] is not None else float("inf")),
            )
        elif self.sorted_order == "descend":
            sorted_history = sorted(
                no_pruned_history,
                key=lambda x: (x["performance"] if x["performance"] is not None else float("-inf")),
                reverse=True,
            )
        else:
            raise ValueError(f"The sorted order {self.sorted_order} is not supported.")
        assert sorted_history is not None
        return sorted_history

    def to_str(self, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (int, float, bool, str)):
            return str(v)
        return json.dumps(v)

    def save(self, history):
        sorted_history = self.sort(history)
        df = pd.DataFrame(sorted_history)
        cols = df.columns.tolist()
        if "idx" in cols:
            cols.insert(0, cols.pop(cols.index("idx")))
        if "stopped_by_tuner" in cols:
            df = df.drop(columns=["stopped_by_tuner"])
        df = df.reindex(columns=cols)
        for c in df.columns:
            df[c] = df[c].map(self.to_str)
        df.to_csv(self.path, index=False, escapechar="\\")

    def parse_value(self, s):
        if s == "":
            return None
        ls = s.lower()
        if ls == "true":
            return True
        if ls == "false":
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        try:
            # Check for JSON string start/end characters as a heuristic
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass
        return s

    def read(self):
        if not os.path.exists(self.path):
            return []
        df = pd.read_csv(
            self.path, dtype=str, keep_default_na=False, na_filter=False, escapechar="\\"
        )
        for c in df.columns:
            df[c] = df[c].map(self.parse_value)
        return df.to_dict(orient="records")


class ServeRecorder(Recorder):
    def __init__(self, config):
        self.config = config
        if (
            "auto_tuner" in self.config.experiment
            and "performance" in self.config.experiment.auto_tuner
        ):
            self.metric = self.config.experiment.auto_tuner.performance.get("metric", "itl")
        else:
            self.metric = "itl"

        # Sort order of performance, order just in [ascend, and descend], default ascend
        if (
            "auto_tuner" in self.config.experiment
            and "performance" in self.config.experiment.auto_tuner
        ):
            self.sorted_order = self.config.experiment.auto_tuner.performance.get("order", "ascend")
        else:
            self.sorted_order = "ascend"
        self.logger = logging.getLogger("FlagScale-AutoTuner")
        self.cur_strategy = None
        self.path = os.path.join(config.experiment.exp_dir, "auto_tuner", "history.csv")

    def record(self, strategy, performance):
        strategy["e2e_latency"] = round(performance["mean_e2el_ms"], 2)
        strategy["request_throughput"] = round(performance["request_throughput"], 2)
        strategy["token_throughput"] = round(performance["total_token_throughput"], 2)
        strategy["ttft"] = round(performance["mean_ttft_ms"], 2)
        strategy["itl"] = round(performance["mean_itl_ms"], 2)
        strategy["topt"] = round(performance["mean_tpot_ms"], 2)
        self.cur_strategy = strategy

    def sort(self, history):
        sorted_history = None
        if self.sorted_order == "ascend":
            sorted_history = sorted(
                history,
                key=lambda x: (x[self.metric] if x[self.metric] is not None else float("inf")),
            )
        elif self.sorted_order == "descend":
            sorted_history = sorted(
                history,
                key=lambda x: (x[self.metric] if x[self.metric] is not None else float("-inf")),
                reverse=True,
            )
        else:
            raise ValueError(f"The sorted order {self.sorted_order} is not supported.")
        assert sorted_history is not None
        return sorted_history

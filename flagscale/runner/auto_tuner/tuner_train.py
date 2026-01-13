import copy
import datetime
import json
import logging
import os
import re
import shutil
import sys
import time

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.auto_tuner.generate import Generator
from flagscale.runner.auto_tuner.hetero import (
    HeteroGenerator,
    HeteroPruner,
    HeteroRecorder,
    HeteroSearcher,
)
from flagscale.runner.auto_tuner.platform import set_jiuding_platform_args
from flagscale.runner.auto_tuner.prune.pruner import Pruner
from flagscale.runner.auto_tuner.record.recorder import Recorder
from flagscale.runner.auto_tuner.search.searcher import Searcher
from flagscale.runner.auto_tuner.tuner import AutoTunerBase
from flagscale.runner.utils import JobStatus, parse_hostfile

FLAGSCALE_USE_V1 = os.environ.get("FLAGSCALE_USE_V1", "1").lower() in ("1", "true")

if FLAGSCALE_USE_V1:
    from flagscale.runner.runner_base import Runner

    SSHTrainRunner = Runner
else:
    from flagscale.runner.runner_train import SSHTrainRunner


class TrainAutoTuner(AutoTunerBase):
    def __init__(self, config: DictConfig):
        # Set logger
        OmegaConf.set_struct(config, False)
        logger = logging.getLogger("FlagScale-AutoTuner")
        logger.setLevel(logging.INFO)

        dir_path = os.path.join(config.experiment.exp_dir, "auto_tuner")
        log_path = os.path.join(dir_path, "tuner.log")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        handler = logging.FileHandler(log_path, mode="a")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger

        # Deepcopy the original config to isolate from each task config
        # Modify the orig config when run best task
        self.orig_config = config
        self.config = copy.deepcopy(config)

        # Set config of auto tuner
        if "auto_tuner" not in self.config.experiment:
            self.config.experiment.auto_tuner = {}

        # Add nodes, nproc_per_node, cards to build search space or prune
        assert "experiment" in config, "experiment is not in yaml file."
        assert "runner" in config.experiment, "runner is not in yaml file."
        nnodes = config.experiment.runner.get("nnodes", 1)
        nproc_per_node = config.experiment.runner.get("nproc_per_node", 8)
        self.config.experiment.auto_tuner.nproc_per_node = nproc_per_node
        self.config.experiment.auto_tuner.nnodes = nnodes

        # Set tuner configs
        # The interval of task monitoring
        if "control" not in self.config.experiment.auto_tuner:
            self.config.experiment.auto_tuner.control = {}
        self.interval = self.config.experiment.auto_tuner.control.get("interval", 10)

        # Set platform envs
        if "platform" not in self.config.experiment.auto_tuner:
            self.config.experiment.auto_tuner.platform = {}

        # As long as AIRS_SWITCH has value it means running on the jiuding platform
        if os.environ.get("AIRS_SWITCH", None):
            set_jiuding_platform_args(self.config, self.orig_config)
        # If on other platform, set the platform envs

        self.config.experiment.auto_tuner.cards = (
            self.config.experiment.auto_tuner.nnodes
            * self.config.experiment.auto_tuner.nproc_per_node
        )

        # Build core sub modules, such as Searcher, Pruner, Generator and Recorder
        is_hetero_enabled = self.config.train.system.get("hetero", {}).get("enable_hetero", False)

        if is_hetero_enabled:
            self.logger.info("Initializing in Heterogeneous Mode.")
            hostfile_path = self.config.experiment.runner.get("hostfile", None)
            resources = parse_hostfile(hostfile_path)
            if not resources:
                raise ValueError(
                    "Heterogeneous tuning requires a valid hostfile, but none was found or it was empty."
                )
            total_cards = sum(info["slots"] for info in resources.values())
            self.config.experiment.auto_tuner.cards = total_cards
            self.searcher = HeteroSearcher(self.config, resources)
            self.pruner = HeteroPruner(self.config)
            self.generator = HeteroGenerator(self.config)
            self.recorder = HeteroRecorder(self.config)
        else:
            self.logger.info("Initializing in Homogeneous Mode.")
            self.searcher = Searcher(self.config)
            self.pruner = Pruner(self.config)
            self.generator = Generator(self.config)
            self.recorder = Recorder(self.config)

        # check configuration file state
        config_path = os.path.join(dir_path, "config.yaml")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                load_config = json.load(f)
            assert load_config == self.config, (
                "The configuration file has changed and cannot be resumed from breakpoint"
            )
        else:
            with open(config_path, "w", encoding="utf-8") as f:
                pure = OmegaConf.to_container(copy.deepcopy(self.config), resolve=True)
                json.dump(pure, f, ensure_ascii=False, indent=2)

        # History strategy
        self.history = self.recorder.read()

        # resume searcher idx
        self.searcher.algo.idx = max(0, int(self.find_search_num_value(log_path)) - 1)

        # Each task has its own runner
        self.runner = None

        # The max time per task, unit: second
        # NOTE: The task will be stopped if the time is reached or done.
        self.max_time_per_task = self.config.experiment.auto_tuner.control.get(
            "max_time_per_task", 300
        )

        # The max time of auto tuner, if None, no limit.
        self.max_time = self.config.experiment.auto_tuner.control.get("max_time", None)

        # The start time of each task, used to control each task when stop
        self.start_task_time = None

        # The start time of tuner, used to control the tuner when stop
        self.start_time = time.time()

        # The history pruned count
        self.pruner.pruned_count = int(self.find_pruned_num_value(log_path))

        # Task id
        self.idx = self.searcher.algo.idx - self.pruner.pruned_count

        # clear breakpoint task log
        if self.searcher.algo.idx >= 0:
            breakpoint_task_path = os.path.join(dir_path, "task_" + str(self.idx + 1))
            self.clear_log(breakpoint_task_path)
            self.searcher.algo.idx = self.idx - 1

        # Checkout search mode on the platform
        self.has_checkout = False

    # clear break task log
    def clear_log(self, folder_path):
        try:
            shutil.rmtree(folder_path)
        except FileNotFoundError:
            return
        except PermissionError:
            self.logger.error("no permission to clear breakpoint task log")
            sys.exit()
        except OSError as e:
            self.logger.info(f"cannot clear breakpoint task log, due {e}")
            sys.exit()

    # get pruned num from log
    def find_pruned_num_value(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                matches = re.findall(r"Pruned (.*?) strategy", file.read())
                return matches[-1] if matches else "0"
        except (OSError, FileNotFoundError, IndexError):
            return "0"

    # get serarch num from log
    def find_search_num_value(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                matches = re.findall(r"Searching (.*?) /", file.read())
                return matches[-1] if matches else "0"
        except (OSError, FileNotFoundError, IndexError):
            return "0"

    def tune(self):
        """
        Tune the model performance, the steps are:
            Step1. Generate the task to run
            Step2. Run task and monitor
            Step3. Record task
            Step4. Loop 1-3 until stop
            Step5. Run the best task
        """
        tuner_start_time = time.time()
        while not self.need_stop():
            self.gen()
            if not self.cur_strategy:
                break
            self.logger.info(f"Run task_{self.idx}: {self.cur_strategy}")
            self.run()
            self.logger.info(f"Monitor task_{self.idx}:")
            self.monitor()
            self.logger.info(f"Record task_{self.idx}:")
            self.record()

            if (
                self.cur_strategy["performance"]
                and self.config.experiment.auto_tuner.platform.get("airs_switch", False)
                and not self.has_checkout
            ):
                self.checkout()

            # get best strategy
            best_strategy = self.get_best()
            if best_strategy:
                self.logger.info(
                    f"Best strategy tuned so far: {best_strategy}, and performance is {best_strategy['performance']}."
                )
            else:
                self.logger.info("No strategy can run so far.")
        tuner_end_time = time.time()
        self.logger.info(f"AutoTuner Ended in {tuner_end_time - tuner_start_time} seconds.")

        # Run the best task
        if self.config.experiment.auto_tuner.control.get("run_best", True):
            best_strategy = self.get_best()
            if best_strategy:
                self.logger.info(f"Run best Strategy: {best_strategy}")
            else:
                raise ValueError("No strategy can run.")
            best_task = self.generator.gen_best_task(best_strategy, self.orig_config)
            best_task.action = "run"
            runner = SSHTrainRunner(best_task)
            enable_monitoring = best_task.experiment.runner.get("enable_monitoring", False)
            runner.run(monitor=True, interval=60, enable_monitoring=enable_monitoring)

    def checkout(self, mode="performance"):
        if not self.has_checkout:
            self.searcher.algo.checkout(mode)
            self.has_checkout = True

    def run(self, task=None):
        # Instantiate a runner and run the task
        if task is None:
            task = self.cur_task
        self.runner = SSHTrainRunner(task)
        enable_monitoring = task.experiment.runner.get("enable_monitoring", False)
        self.runner.run(enable_monitoring=enable_monitoring)
        # set start time
        self.task_start_time = time.time()

    def monitor(self):
        """Monitor the task until task timeout or completed."""
        # Sleep 3s to ensure the task is started
        time.sleep(3)
        running = False
        sub_process_running = False
        while True:
            # If the task timeout, stop monitoring
            end_time = time.time()
            # To increase the time to 600s for the first task with data processing and cache.
            if self.idx == 1:
                max_time_per_task = 2 * self.max_time_per_task
            else:
                max_time_per_task = self.max_time_per_task
            if end_time - self.task_start_time > max_time_per_task:
                self.runner.stop()
                self.cur_strategy["stopped_by_tuner"] = True
                break
            # If the task is completed or idle, stop monitoring
            try:
                if FLAGSCALE_USE_V1:
                    status = self.runner.launcher._query_status()
                else:
                    status = self.runner._query_status()
                self.logger.info(f"task_{self.cur_strategy['idx']} status: {status.name}")
                if status == JobStatus.COMPLETED_OR_IDLE:
                    break
                if status == JobStatus.RUNNING:
                    running = True
                if status == JobStatus.TRANSITIONAL:
                    if running:
                        self.runner.stop()
                        break

                # Add sub process monitor
                if FLAGSCALE_USE_V1:
                    sub_process = self.runner.launcher._query_sub_process_status()
                else:
                    sub_process = self.runner._query_sub_process_status()
                if sub_process:
                    sub_process_running = True

                elif not sub_process:
                    if sub_process_running:
                        self.logger.info("Sub process not working, stop the task.")
                        self.runner.stop()
                        self.cur_strategy["stopped_by_tuner"] = True
                        break

            except Exception as e:
                self.logger.info(e)
                time.sleep(self.interval)
            time.sleep(self.interval)

        end_time = time.time()

        # Add elapsed time
        self.cur_strategy["elapsed_time"] = round(end_time - self.task_start_time, 2)
        # Add start time
        readable_task_start_time = datetime.datetime.fromtimestamp(self.task_start_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.cur_strategy["start_time"] = readable_task_start_time

        self.logger.info(
            f"task_{self.cur_strategy['idx']} monitor time: {self.cur_strategy['elapsed_time']:.2f}s"
        )

    def record(self):
        """Record the task result to csv"""
        self.recorder.record(self.cur_task, self.cur_strategy)
        self.recorder.save(self.history)

    def get_best(self):
        sorted_history = self.recorder.sort(self.history)
        if sorted_history and sorted_history[0] and sorted_history[0]["performance"]:
            return sorted_history[0]
        return None

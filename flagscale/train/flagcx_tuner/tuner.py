import os
import subprocess

import torch

from flagscale.train.flagcx_tuner.recorder import FlagCXTuneRecorder


class FlagCXTuner:

    def __init__(self):
        self.iter = -1
        self.config_id = -1
        self.best_config_id = -1
        self.recorder = FlagCXTuneRecorder()
        self.perf_map = {}
        self.best_perf = float("inf")
        self.finished_tuning = False
        self.best_config_set = False
        os.environ["FLAGCX_TUNER_CONFIG_ID"] = str(self.config_id)
        os.environ["FLAGCX_TUNER_BEST_CONFIG_ID"] = str(self.best_config_id)

        # Get autotuner log directory from environment variable
        tune_file_path = os.environ.get("FLAGCX_TUNE_FILE")
        assert tune_file_path is not None, "Environment variable FLAGCX_TUNE_FILE must be set."
        log_dir = os.path.dirname(os.path.abspath(tune_file_path))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def tuning_done(self):
        return self.finished_tuning

    def check_flagcx_done(self):
        # os.envirion cannot read env variables set outside of python
        # so execute a shell command to read it
        result = subprocess.check_output("echo $FLAGCX_TUNER_DONE", shell=True).decode().strip()
        if result == "1":
            self.finished_tuning = True

    def need_config_update(self):
        if self.iter % 5 == 0:
            return True
        return False

    def update_iter(self):
        self.iter += 1
        self.recorder.record()

    def update_config(self):
        # synchronize ranks before updating config to make sure all ranks are done with current config
        torch.cuda.synchronize()
        torch.distributed.barrier()
        self.config_id += 1
        os.environ["FLAGCX_TUNER_CONFIG_ID"] = str(self.config_id)

    def need_eval(self):
        if self.iter > 0 and self.iter % 5 == 0:
            return True
        return False

    def eval_e2e_perf(self):
        records = self.recorder.get_records()
        assert len(records) >= 5
        # calculate the average time of last 4 iters
        self.perf_map[self.config_id] = sum(records[-4:]) / 4.0
        if self.perf_map[self.config_id] < self.best_perf:
            self.best_perf = self.perf_map[self.config_id]
            self.best_config_id = self.config_id
        self.recorder.reset()

    def best_config_set(self):
        return self.best_config_set

    def set_best_config(self):
        os.environ["FLAGCX_TUNER_BEST_CONFIG_ID"] = str(self.best_config_id)
        self.best_config_set = True

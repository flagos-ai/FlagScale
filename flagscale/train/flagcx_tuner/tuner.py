import os
import subprocess

import torch
import torch.distributed as dist

from megatron.core import mpu

from flagscale.train.flagcx_tuner.recorder import FlagCXTuneRecorder

pg_map = {
    "cp": mpu.get_context_parallel_group,
    "mp": mpu.get_model_parallel_group,
    "tp": mpu.get_tensor_model_parallel_group,
    "pp": mpu.get_pipeline_model_parallel_group,
    "ep": mpu.get_expert_model_parallel_group,
}


class FlagCXTuner:

    def __init__(self, flagcx_tune_groups):
        self.iter = -1
        self.config_id = -1
        self.best_config_id = -1
        self.recorder = FlagCXTuneRecorder()
        self.perf_map = {}
        self.best_perf = float("inf")
        self.tune_groups = flagcx_tune_groups
        self.cur_group_idx = 0
        self.tune_group_size = len(self.tune_groups)
        self.finished_tuning = [False for _ in range(self.tune_group_size)]
        self.best_config_set = [False for _ in range(self.tune_group_size)]
        self.finished_all_tuning = False
        self.need_reset = False
        os.environ["FLAGCX_TUNER_CONFIG_ID"] = str(self.config_id)
        os.environ["FLAGCX_TUNER_BEST_CONFIG_ID"] = str(self.best_config_id)
        os.environ["FLAGCX_TUNE_GROUP_IDX"] = str(self.cur_group_idx)

        # Get autotuner log directory from environment variable
        tune_file_path = os.environ.get("FLAGCX_TUNE_FILE")
        assert tune_file_path is not None, "Environment variable FLAGCX_TUNE_FILE must be set."
        log_dir = os.path.dirname(os.path.abspath(tune_file_path))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def tuning_done(self):
        return self.finished_all_tuning

    def cur_group_tuning_done(self):
        return self.finished_tuning[self.cur_group_idx]

    def check_flagcx_done(self):
        # os.envirion cannot read env variables set outside of python
        # so execute a shell command to read it
        result = subprocess.check_output("echo $FLAGCX_TUNER_DONE", shell=True).decode().strip()
        if result == "1":
            self.finished_tuning[self.cur_group_idx] = True

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
        perf_tensor = torch.tensor(sum(records[-4:]) / 4.0)
        # synchronize perf across all ranks
        group = pg_map[self.tune_groups[self.cur_group_idx]]()
        size = dist.get_world_size(group=group)
        dist.all_reduce(perf_tensor, op=dist.ReduceOp.SUM, group=group)
        perf = perf_tensor.item() / size
        self.perf_map[self.config_id] = perf
        if self.perf_map[self.config_id] < self.best_perf:
            self.best_perf = self.perf_map[self.config_id]
            self.best_config_id = self.config_id
        self.recorder.reset()

    def cur_group_best_config_used(self):
        return self.best_config_set[self.cur_group_idx]

    def set_cur_group_best_config(self):
        os.environ["FLAGCX_TUNER_BEST_CONFIG_ID"] = str(self.best_config_id)
        self.best_config_set[self.cur_group_idx] = True
        self.need_update_tune_group = True
        if self.cur_group_idx == self.tune_group_size - 1:
            self.finished_all_tuning = True

    def need_update_tune_group(self):
        return self.need_update_tune_group

    def update_tune_group(self):
        self.iter = -1
        self.config_id = -1
        self.best_config_id = -1
        self.perf_map = {}
        self.best_perf = float("inf")
        self.cur_group_idx += 1
        self.need_reset = False
        os.environ["FLAGCX_TUNER_CONFIG_ID"] = str(self.config_id)
        os.environ["FLAGCX_TUNER_BEST_CONFIG_ID"] = str(self.best_config_id)
        os.environ["FLAGCX_TUNE_GROUP_IDX"] = str(self.cur_group_idx)
        os.environ["FLAGCX_TUNER_DONE"] = "0"

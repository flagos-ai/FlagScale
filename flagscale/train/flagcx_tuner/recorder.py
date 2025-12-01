from megatron.training.global_vars import get_timers

class FlagCXTuneRecorder:

    def __init__(self):
        self.timer = get_timers()("flagcx-tuner-time")
        self.records = []
        self.timer.start(barrier=True)

    def record(self):
        time = self.timer.elapsed()
        self.records.append(time)

    def get_records(self):
        return self.records

    def reset(self):
        self.records = []
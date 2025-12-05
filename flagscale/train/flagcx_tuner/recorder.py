from megatron.training.global_vars import get_timers

class FlagCXTuneRecorder:

    def __init__(self):
        self.timer = get_timers()("flagcx-tuner-time", log_level=0)
        self.records = []
        self.timer.start(barrier=True)

    def record(self):
        print("flagcxTuner recording time")
        time = self.timer.elapsed()
        print(f"flagcxTuner recorded time: {time}")
        self.records.append(time)

    def get_records(self):
        return self.records

    def reset(self):
        self.records = []
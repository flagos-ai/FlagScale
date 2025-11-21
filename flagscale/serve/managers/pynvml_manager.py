import pynvml


class PynvmlManager:
    def __init__(self):
        pynvml.nvmlInit()

    def get_gpu_info(self):
        gpu_info = {}
        if pynvml.nvmlDeviceGetCount():
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info = {
                "name": pynvml.nvmlDeviceGetName(handle),
                "num": pynvml.nvmlDeviceGetCount(),
                "memory_total": f"{mem_info.total / 1024**3:.2f}GB",
                "memory_used": f"{mem_info.used / 1024**3:.2f}GB",
                "memory_free": f"{mem_info.free / 1024**3:.2f}GB",
            }
        return gpu_info


PYNVML_MANAGER = PynvmlManager()

"""Best-effort process and GPU snapshots with clean CPU-only behavior."""
import os

import psutil


def snapshot() -> dict[str, float | int | None]:
    process = psutil.Process()
    gpu_utilization = None
    gpu_used_mb = None
    gpu_total_mb = None
    try:
        import pynvml
    except ImportError:
        pynvml = None
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_utilization = float(utilization.gpu)
            gpu_used_mb = memory.used / (1024 * 1024)
            gpu_total_mb = memory.total / (1024 * 1024)
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
    if gpu_used_mb is None:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        except (ImportError, RuntimeError):
            pass
    return {
        "process_id": os.getpid(),
        "process_rss_mb": process.memory_info().rss / (1024 * 1024),
        "process_cpu_percent": process.cpu_percent(interval=None),
        "gpu_utilization_percent": gpu_utilization,
        "gpu_memory_used_mb": gpu_used_mb,
        "gpu_memory_total_mb": gpu_total_mb,
    }

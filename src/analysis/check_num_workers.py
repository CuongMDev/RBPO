import os
import torch

def suggest_max_workers(batch_size: int, use_gpu: bool = False, hard_cap: int = 6):
    cores = os.cpu_count() or 2

    # giới hạn theo API pressure
    api_cap = max(1, hard_cap // batch_size)

    # giới hạn theo CPU
    cpu_cap = max(1, cores // 2)

    if use_gpu and torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # giả định ~3GB VRAM / worker cho embedding/rerank
        gpu_cap = max(1, int(total_mem // 3))
        return min(api_cap, cpu_cap, gpu_cap, hard_cap)
    else:
        return min(api_cap, cpu_cap, hard_cap)


print("Suggested max_workers =", suggest_max_workers(2, use_gpu=True))

import gc
import torch


def free_gpu(*objects) -> None:
    """Delete references and reclaim GPU memory."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
"""GPU memory management helpers.

The original notebook leaked VRAM between committee models because local
references kept tensors alive past `del model`. `free_gpu` accepts any
number of positional objects, drops their references, and forces Python's
garbage collector and the CUDA caching allocator to reclaim memory.
"""

from __future__ import annotations

import gc

import torch


def free_gpu(*objects) -> None:
    """Delete references to ``objects`` and reclaim GPU memory.

    Accepts zero or more positional arguments. Passing nothing is a valid
    call and simply flushes the CUDA cache.
    """
    for obj in objects:
        try:
            del obj
        except Exception:
            # Deleting a non-existent local or a value we don't own is
            # never fatal; we only care about dropping the reference.
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


__all__ = ["free_gpu"]
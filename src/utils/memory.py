"""GPU memory management helpers.

`free_gpu` is deliberately signature-tolerant so it can be called either as
`free_gpu()` (preferred) or `free_gpu(a, b, c)` (legacy). The actual release
of Python references must happen at the call site via `del`, because a
function cannot drop the caller's bindings.
"""

from __future__ import annotations

import gc
import torch


def free_gpu(*_objects) -> None:
    """Reclaim cached GPU memory.

    Parameters
    ----------
    *_objects
        Optional references retained for call-site expressiveness. They are
        accepted but not deleted here — the caller is responsible for
        ``del model, tokenizer``. Passing them is a no-op that keeps older
        call sites (``free_gpu(model, tokenizer)``) working.

    Notes
    -----
    This function performs two actions:

    1. Runs a full cyclic GC pass so that any Python objects whose ``__del__``
       holds CUDA tensors release them.
    2. Empties PyTorch's CUDA caching allocator, returning memory to the
       driver so the next model load can use it.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # release any inter-process shared handles
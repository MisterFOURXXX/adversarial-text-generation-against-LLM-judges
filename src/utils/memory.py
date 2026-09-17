"""GPU memory reclamation helpers.

Python reference counting means a helper function *cannot* free an object
held by the caller — ``del obj`` inside a function only removes the local
binding. The correct pattern is::

    model = None
    tokenizer = None
    free_gpu()

This module therefore exposes ``free_gpu`` as a zero-argument reclaim step,
and callers are responsible for dropping their references first.
"""

from __future__ import annotations

import gc

import torch


def free_gpu() -> None:
    """Run Python GC and release cached CUDA blocks.

    Call *after* setting the relevant model/tokenizer variables to ``None``
    (or letting them fall out of scope). Safe to call even when CUDA is
    unavailable — the ``torch.cuda`` calls are no-ops in that case.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
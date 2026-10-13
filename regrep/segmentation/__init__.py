"""TotalSegmentator and LION segmentation, and the lesion supports built from them."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch


@contextlib.contextmanager
def cudnn_flags(benchmark: bool, deterministic: bool,
                threads: int | None = None) -> Iterator[None]:
    """Set the cuDNN flags, and optionally the CPU threads of torch, in a block.

    The segmentors run with the settings of nnU-Net: ``benchmark`` on and
    ``deterministic`` off. On exit the previous values come back, so code that
    runs later in the same process keeps its own settings.
    """
    saved = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.get_num_threads())
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[:2]
        torch.set_num_threads(saved[2])

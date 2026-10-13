"""Organ and lesion segmentation of one pair, in-process or as a worker process.

:func:`segment_organs` and :func:`segment_lesions` run the two segmentors and
hand each map to a callback as soon as it is ready. The serial mode calls
them in the main process. The concurrent mode runs each of them in its own
process with

.. code-block:: text

    python -m regrep.segmentation.worker organs --workspace DIR \\
        --fixed-ct FIXED_CT --moving-ct MOVING_CT --config CONFIG.json
    python -m regrep.segmentation.worker lesions --workspace DIR \\
        --fixed-pet FIXED_PET --moving-pet MOVING_PET \\
        --fixed-ct FIXED_CT --moving-ct MOVING_CT --config CONFIG.json

The working grids come from the headers of the CTs, so the lesion worker
gets their paths too. A worker writes every map to the workspace with an
atomic rename and exits with status 0. On any error it prints the traceback
and exits with status 1. It also exits when its parent process ends, so a
worker never outlives the registration that started it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable

import numpy as np

from regrep.config import ChainConfig, load_config
from regrep.io import (
    Volume,
    Workspace,
    load_volume,
    orient_like,
    read_grid,
    to_working_grid,
    working_grid,
)

log = logging.getLogger(__name__)

LESION_MAPS = ("lion_moving", "lion_fixed")
"""The maps of the lesion segmentation, in the order in which they are written."""

Emit = Callable[[str, np.ndarray], None]


def _peak_gib(device) -> float:
    import torch

    return torch.cuda.max_memory_allocated(device) / 2**30


def segment_organs(fixed_ct: Volume, moving_ct: Volume, config: ChainConfig,
                   device, emit: Emit) -> None:
    """Make the four TotalSegmentator maps of a pair.

    ``emit(name, array)`` receives the maps of ``ORGAN_MAPS``
    (``regrep.segmentation.labels``); each array is uint8 ``(X, Y, Z)``
    on the grid of its CT. The models are released before the function
    returns, and the caller's cuDNN flags are restored. The weights are read
    from TotalSegmentator's weights folder (``TOTALSEG_WEIGHTS_PATH`` or
    ``TOTALSEG_HOME_DIR``).
    """
    from regrep.segmentation.totalseg import OrganSegmenter

    segmenter = OrganSegmenter(device, config.organs)
    try:
        segmenter.segment_pair(fixed_ct, moving_ct, emit)
    finally:
        segmenter.release()


def segment_lesions(fixed_pet: Volume, moving_pet: Volume, config: ChainConfig,
                    device, emit: Emit) -> None:
    """Make the LION lesion masks of the moving and then the fixed PET.

    ``emit(name, array)`` receives ``lion_moving`` and then ``lion_fixed``,
    each a uint8 ``(X, Y, Z)`` mask on the grid of its PET. The model is
    released before the function returns, and the caller's cuDNN flags are
    restored. The weights are read from the LION models folder
    (``LIONZ_MODELS_DIRECTORY``).
    """
    from regrep.segmentation.lion import LionSegmenter

    segmenter = LionSegmenter(config.lesions, device)
    try:
        for name, pet in zip(LESION_MAPS, (moving_pet, fixed_pet), strict=True):
            emit(name, segmenter.segment(pet))
    finally:
        segmenter.release()


def _exit_with_parent() -> None:
    """End this process soon after its parent process ends."""
    parent = os.getppid()

    def watch() -> None:
        while True:
            if os.getppid() != parent:
                os._exit(1)
            time.sleep(1.0)

    threading.Thread(target=watch, daemon=True).start()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m regrep.segmentation.worker",
        description="Segment one pair and write the maps to a workspace.")
    parser.add_argument("task", choices=("organs", "lesions"),
                        help="organs: TotalSegmentator on both CTs; lesions: LION "
                             "on both PETs")
    parser.add_argument("--workspace", required=True,
                        help="the directory that receives the maps")
    parser.add_argument("--config", required=True,
                        help="chain settings (YAML or JSON)")
    parser.add_argument("--fixed-ct", required=True,
                        help="baseline CT (the fixed image)")
    parser.add_argument("--moving-ct", required=True,
                        help="follow-up CT (the moving image)")
    parser.add_argument("--fixed-pet", help="baseline PET (the fixed image)")
    parser.add_argument("--moving-pet", help="follow-up PET (the moving image)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one worker; return its exit status."""
    args = _parser().parse_args(argv)
    _exit_with_parent()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format=f"[worker {args.task}] %(message)s")
    started = time.perf_counter()
    try:
        import torch

        # The parent process leaves the worker exactly one visible GPU.
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        config = load_config(args.config)
        workspace = Workspace(path=args.workspace, keep=True)

        # Each volume is resampled onto the working grid of the CT of its
        # timepoint, and the moving one is reordered to the axis orientation
        # of the fixed one, as the main process does.
        cts = (args.fixed_ct, args.moving_ct)
        paths = cts if args.task == "organs" else (args.fixed_pet, args.moving_pet)
        fixed, moving = (
            to_working_grid(load_volume(path),
                            working_grid(*read_grid(ct), config.grid))
            for path, ct in zip(paths, cts, strict=True))
        moving = orient_like(moving, fixed.affine)

        def emit(name: str, array: np.ndarray) -> None:
            like = fixed if name in ("labels_fixed", "lion_fixed") else moving
            workspace.write_labels(name, array, like)
            log.info("%s written after %.1f s, peak GPU memory %.2f GiB", name,
                     time.perf_counter() - started, _peak_gib(device))

        if args.task == "organs":
            segment_organs(fixed, moving, config, device, emit)
        else:
            segment_lesions(fixed, moving, config, device, emit)
        log.info("done after %.1f s", time.perf_counter() - started)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""GPU selection, memory estimates and the run plan of one pair.

A pair runs in one of two modes; the ``auto`` mode chooses one of them
(see item 1 below).

``serial``
    Every step runs in this process, one after the other. The segmentors run
    first, and their models are released before the features are computed.
``concurrent``
    TotalSegmentator and LION run in two worker processes
    (``python -m regrep.segmentation.worker``) while this process loads the
    anatomix network and computes the features. Stage 1 waits for the maps
    it needs, and stage 2 waits for the rest.

Both modes call the same functions on the same inputs, so their results
differ only by the order of floating-point sums.

How the plan is chosen
----------------------
All memory figures below are in GiB, measured on the chain itself at the
reference size of 192 x 192 x 288 voxels of 2.7344 x 2.7344 x 3.27 mm. Each
figure is scaled to the pair in one of three ways: by the voxel count of the
fixed or of the moving grid (the data that the registration stages keep on
each grid), by the voxel count of the isotropic grids of the feature pyramid
(which grows with the physical extent of a CT and with the fineness of the
fixed CT), or by the physical extent in mm (the organ segmentation, which
works on grids of a fixed spacing). A step fits on a GPU
when ``SAFETY_FACTOR * estimate + workers + SAFETY_MARGIN_GIB`` is at most the
free memory of the GPU, measured when the plan is made. ``estimate`` is the
peak of this process and ``workers`` the memory of the segmentation
processes that run on the GPU at the same time (0 for most steps). The
figures assume PyTorch's ``expandable_segments`` allocator setting, which the
command line sets.

1. GPUs and mode (``auto``). The first GPU listed is the main GPU: it holds
   the features and the transform, and stage 1 always runs on it, alone
   when it fits. In the ``auto`` mode, every other GPU on which the two
   segmentors do not fit is left out. With two or more GPUs left the mode is
   ``concurrent``. With one GPU it is ``concurrent`` when the feature
   extraction and both segmentors fit on it at the same time, and ``serial``
   otherwise.
2. Segmentation GPU. In the concurrent mode with two or more GPUs the
   segmentors run on the GPU other than the main GPU with the most free
   memory (the last one listed on a tie), and stage 1 never uses that GPU.
   Stage 2 may use every listed GPU; it starts on the segmentation GPU only
   after both segmentation processes have exited. In the concurrent mode
   with one GPU, the segmentors share the GPU with the feature extraction,
   and stage 1 starts only after both segmentation processes have exited.
   In the serial mode everything starts on the main GPU.
3. Channel chunks. A stage computes its image term in channel chunks spread
   over one or more GPUs (see ``regrep.registration.chunked``). Fewer
   chunks give results closer to those of a single chunk on one GPU. For
   each stage the candidates are one chunk per GPU and chunks of
   ``CHUNK_SIZES`` channels, on the first ``n`` GPUs available to the
   stage.

   The estimate of a candidate on several GPUs includes the extra
   allocator memory of the main GPU (``MULTI_GPU_MAIN_GIB``).

   * ``concurrent``: the candidate with the fewest chunks that fits; on a
     tie, the one with fewer GPUs.
   * ``serial``: the fewest GPUs on which a candidate fits, and on those
     GPUs the candidate with the fewest chunks that fits. A stage uses more
     than one GPU only when no chunk size fits on one GPU.

   A chunk size given by the user replaces the candidates. The stage then
   uses, among the GPU counts on which that size fits, the one with the
   fewest chunks (concurrent) or the fewest GPUs (serial). If it fits on
   none, the stage uses all its GPUs with that size and a warning is logged,
   because the estimates are conservative.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from regrep.config import FeatureConfig

log = logging.getLogger(__name__)

MODES = ("auto", "serial", "concurrent")

REFERENCE_SHAPE = (192, 192, 288)
REFERENCE_SPACING = (2.7344, 2.7344, 3.27)
"""The volume size and voxel size (mm) at which the memory figures were measured."""

GIB = 2**30

# Memory of one channel of one stack at the reference size (float32).
CHANNEL_GIB = math.prod(REFERENCE_SHAPE) * 4 / GIB

# The fit rule: SAFETY_FACTOR * estimate + workers + SAFETY_MARGIN_GIB <= free
# memory. The margin holds the CUDA context of this process and allocator
# slack.
SAFETY_FACTOR = 1.05
SAFETY_MARGIN_GIB = 1.5

# Peak of the feature extraction (the moving side, with the fixed features
# resident): a fixed part (the network and its activations), a part that
# grows with the voxel count of the native grids (the features of both
# sides), and the peak of the pyramid level that needs the most memory. At
# the reference size that is the 1 mm level of the anatomix pyramid, which is
# also the MIND-SSC grid; its part grows with the voxel count of the 1 mm
# grid. The three parts were fitted to the peaks measured at the reference
# size, on the same extent with 0.3 times the voxels, and on half the extent.
FEATURES_FIXED_GIB = 10.2
FEATURES_GRID_GIB = 16.8
FEATURES_NATIVE_GIB = 2.0
# The 1 mm level of the anatomix pyramid, which is also the MIND-SSC grid. The
# memory figures were measured with the default feature settings, so the
# estimates follow the default spacing, not a spacing set in a config file.
FEATURE_GRID_MM = FeatureConfig.mind_spacing_mm
# The pyramid levels are summed, in float32 with 32 channels, on the
# isotropic grid whose spacing is the smallest voxel size of the fixed CT;
# that grid is also the third pyramid level. While the 1 mm level is
# computed, the sum takes FEATURE_SUM_BYTES per voxel of that grid; the part
# beyond its size at the reference is added to the 1 mm level. While the
# third level is computed, it holds the sum, a float32 copy of the level and
# its float16 window sums and weights.
FEATURE_SUM_BYTES = 32 * 4
FEATURE_SUM_LEVEL_BYTES = 2 * 32 * 4 + 33 * 2

# Peak memory of each segmentation process on its GPU, with its CUDA context
# and allocator cache. The values include a margin; both processes together
# were measured at 8.0 GiB at the reference size. The organ segmentation runs
# its models on grids of 6, 3 and 1.5 mm over the field of view, so the
# larger part of its memory grows with the physical extent of the CT, down to
# a floor for small scans (2.9 GiB allocated on a scan of 400 x 300 x 320 mm,
# plus the CUDA context). The memory of LION hardly changes with the input.
ORGAN_WORKER_FIXED_GIB = 1.6
ORGAN_WORKER_EXTENT_GIB = 5.4
ORGAN_WORKER_MIN_GIB = 3.5
LESION_WORKER_GIB = 2.0

# Memory that the main GPU holds beyond the allocated peak of this process
# when the chunks of a stage are spread over several GPUs. Measured, with the
# CUDA context: 6.0 GiB with one chunk on each of two GPUs, 5.3 GiB with one
# chunk on each of three.
MULTI_GPU_MAIN_GIB = 5.0

# Chunk sizes tried, largest first, after one chunk per GPU.
CHUNK_SIZES = (32, 24, 16, 8, 4)

FULL = "full"
"""A chunk setting: one chunk per GPU."""
AUTO = "auto"
"""A chunk setting: chosen by the plan."""


@dataclass(frozen=True)
class StageMemory:
    """Linear memory model of one registration stage.

    The peak on a GPU is the sum of: the resident data of the chain (main
    GPU only); the channel chunks that live on the GPU (fixed and moving,
    each chunk with its own copy of the mask); and the working memory of the
    largest chunk, ``label_gib`` per label channel plus ``image_gib`` per
    image channel. A single chunk on one GPU is the stack itself.

    Attributes
    ----------
    labels, images
        Number of label channels (first) and image channels (after them).
    label_gib, image_gib
        Working memory per channel of the chunk in flight.
    resident_gib
        Data that stays on the main GPU during the stage.
    moving_resident_gib
        The part of ``resident_gib`` (and of ``floor_gib``) that lives on the
        moving grid.
    moving_stack
        Whether the moving stack is on the moving grid. Otherwise both stacks
        are on the fixed grid.
    floor_gib
        A lower bound of the peak on the main GPU (the stack assembly).
    inputs_gib, whole_label_gib
        Before the optimization, stage 2 evaluates its terms at the identity
        in pieces of the chunk size. With one chunk per GPU this happens in
        one piece on the main GPU, next to both full stacks and the inputs
        (``inputs_gib``), with ``whole_label_gib`` per label channel.
    """

    labels: int
    images: int
    label_gib: float
    image_gib: float
    resident_gib: float
    moving_resident_gib: float
    moving_stack: bool = False
    floor_gib: float = 0.0
    inputs_gib: float = 0.0
    whole_label_gib: float = 0.0

    @property
    def channels(self) -> int:
        return self.labels + self.images


# Stage 1: 46 image channels under the masked local NCC. The moving stack is
# on the moving grid. The resident data are the features of both sides.
STAGE1 = StageMemory(labels=0, images=46, label_gib=0.0, image_gib=0.62,
                     resident_gib=4.7, moving_resident_gib=2.35,
                     moving_stack=True)
# Stage 2: at most 117 structures and the lesion as label channels, then 32
# anatomix channels under the local NCC. Both stacks are on the fixed grid.
# The resident data are the anatomix features, the masks, the stage-1 grid,
# the region term and the optimizer; the moving anatomix features and the
# distance channels of the region term are on the moving grid.
STAGE2 = StageMemory(labels=118, images=32, label_gib=0.30, image_gib=0.60,
                     resident_gib=8.9, floor_gib=25.0, inputs_gib=2.8,
                     whole_label_gib=0.16, moving_resident_gib=5.9)


Grid = tuple[Sequence[int], Sequence[float]]
"""The shape and the voxel size (mm) of a CT."""


def _iso_voxels(grid: Grid, spacing: float) -> int:
    """Voxel count of the isotropic grid of ``spacing`` over the extent of a CT.

    The shape follows ``regrep.features.anatomix.grid_shape``.
    """
    shape, voxel = grid
    return math.prod(int(round((int(n) - 1) * float(s) / spacing)) + 1
                     for n, s in zip(shape, voxel, strict=True))


def _extent(grid: Grid) -> float:
    """Physical volume (mm^3) from the first to the last voxel centre."""
    shape, voxel = grid
    return math.prod((int(n) - 1) * float(s) for n, s in zip(shape, voxel, strict=True))


REFERENCE_GRID = (REFERENCE_SHAPE, REFERENCE_SPACING)


@dataclass(frozen=True)
class PairSize:
    """The size of a pair relative to the reference size.

    Attributes
    ----------
    fixed_voxels, moving_voxels
        Voxel count of the fixed and of the moving CT, each divided by the
        reference voxel count.
    feature_grid
        Voxel count of the 1 mm grid of the feature pyramid, for the
        timepoint where it is largest, divided by that of the reference.
    feature_sum
        Voxel count of the isotropic grid on which the pyramid levels are
        summed, for the timepoint where it is largest. Its spacing is the
        smallest voxel size of the fixed CT, for both timepoints.
    extent
        Physical volume of the larger field of view of the two CTs (from the
        first to the last voxel centre) divided by that of the reference.
    """

    fixed_voxels: float
    moving_voxels: float
    feature_grid: float
    feature_sum: int
    extent: float

    @classmethod
    def of(cls, fixed: Grid, moving: Grid) -> PairSize:
        """The size of a pair of CTs, each given as (shape, voxel size in mm)."""
        reference_voxels = math.prod(REFERENCE_SHAPE)
        sum_mm = min(float(s) for s in fixed[1])
        sides = (fixed, moving)
        return cls(
            fixed_voxels=math.prod(int(n) for n in fixed[0]) / reference_voxels,
            moving_voxels=math.prod(int(n) for n in moving[0]) / reference_voxels,
            feature_grid=max(_iso_voxels(g, FEATURE_GRID_MM) for g in sides)
            / _iso_voxels(REFERENCE_GRID, FEATURE_GRID_MM),
            feature_sum=max(_iso_voxels(g, sum_mm) for g in sides),
            extent=max(_extent(g) for g in sides) / _extent(REFERENCE_GRID))

    def features_gib(self) -> float:
        """Estimated peak of the feature extraction."""
        reference_sum = _iso_voxels(REFERENCE_GRID, min(REFERENCE_SPACING))
        grid_level = (FEATURES_GRID_GIB * self.feature_grid
                      + max(self.feature_sum - reference_sum, 0)
                      * FEATURE_SUM_BYTES / GIB)
        sum_level = self.feature_sum * FEATURE_SUM_LEVEL_BYTES / GIB
        return (FEATURES_FIXED_GIB + max(grid_level, sum_level)
                + FEATURES_NATIVE_GIB * (self.fixed_voxels + self.moving_voxels) / 2.0)

    def workers_gib(self) -> float:
        """Estimated memory of both segmentation processes on one GPU."""
        organs = ORGAN_WORKER_FIXED_GIB + ORGAN_WORKER_EXTENT_GIB * self.extent
        return max(organs, ORGAN_WORKER_MIN_GIB) + LESION_WORKER_GIB


@dataclass(frozen=True)
class StagePlan:
    """Where and in how many chunks one registration stage runs.

    Attributes
    ----------
    devices
        The GPUs of the channel chunks; the first is the main GPU.
    chunk_channels
        Largest number of channels in one chunk, or ``None`` for one chunk
        per GPU.
    estimate_gib
        The estimated peak memory on each GPU, at the size of the pair.
    """

    devices: tuple[torch.device, ...]
    chunk_channels: int | None
    estimate_gib: tuple[float, ...] = ()

    def describe(self) -> str:
        gpus = ",".join(str(d.index) for d in self.devices)
        chunks = "one per GPU" if self.chunk_channels is None else (
            f"{self.chunk_channels} channels")
        estimate = ", ".join(f"{g:.1f}" for g in self.estimate_gib)
        return f"GPUs {gpus}, chunks {chunks}, estimated peak GiB [{estimate}]"


@dataclass(frozen=True)
class RunPlan:
    """How one pair runs: the mode, the GPUs of each role and the chunks.

    Attributes
    ----------
    mode
        ``"serial"`` or ``"concurrent"``.
    main_device
        The GPU that holds the features and the transform.
    segmentation_device
        The GPU of the segmentors.
    stage1, stage2
        The plans of the two registration stages.
    """

    mode: str
    main_device: torch.device
    segmentation_device: torch.device
    stage1: StagePlan
    stage2: StagePlan

    def waits_for_segmentation(self, stage: StagePlan) -> bool:
        """Whether a stage must wait for the segmentation processes to exit.

        This is the case in the concurrent mode when the stage uses the
        segmentation GPU.
        """
        return self.mode == "concurrent" and self.segmentation_device in stage.devices

    def describe(self) -> str:
        return (f"mode {self.mode}; segmentation on GPU "
                f"{self.segmentation_device.index}; main GPU "
                f"{self.main_device.index}; stage 1: {self.stage1.describe()}; "
                f"stage 2: {self.stage2.describe()}")


def _fits(estimate: float, free: float, workers: float = 0.0) -> bool:
    """The fit rule of the module docstring."""
    return SAFETY_FACTOR * estimate + workers + SAFETY_MARGIN_GIB <= free


def _near_equal(start: int, stop: int, parts: int) -> list[tuple[int, int]]:
    """Cut ``[start, stop)`` into ``parts`` contiguous ranges of near-equal size."""
    bounds = [start + round(i * (stop - start) / parts) for i in range(parts + 1)]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def channel_ranges(total: int, parts: int,
                   chunk: int | None) -> list[list[tuple[int, int]]]:
    """Split ``total`` channels over ``parts`` GPUs, and each GPU's block into chunks.

    Each GPU gets a contiguous block of near-equal size, and each block is cut
    into the fewest near-equal chunks of at most ``chunk`` channels (one
    chunk when ``chunk`` is ``None``). The run plan and
    ``regrep.registration.chunked.split_channels`` both use this split.

    Returns
    -------
    list
        One list per GPU of ``(start, stop)`` channel ranges.
    """
    if parts < 1 or parts > total:
        raise ValueError(f"{parts} GPUs for {total} channels")
    if chunk is not None and chunk < 1:
        raise ValueError(f"the chunk size must be positive, got {chunk}")
    ranges = []
    for lo, hi in _near_equal(0, total, parts):
        count = 1 if chunk is None else math.ceil((hi - lo) / chunk)
        ranges.append(_near_equal(lo, hi, count))
    return ranges


def stage_estimate(model: StageMemory, gpus: int, chunk: int | None,
                   size: PairSize) -> list[float]:
    """Estimated peak memory (GiB) on each GPU of a stage.

    Parameters
    ----------
    model
        The memory model of the stage.
    gpus
        Number of GPUs the channels are split over.
    chunk
        Largest number of channels per chunk, or ``None`` for one chunk per
        GPU.
    size
        The size of the pair.
    """
    fixed = size.fixed_voxels
    moving = size.moving_voxels if model.moving_stack else fixed
    resident = ((model.resident_gib - model.moving_resident_gib) * fixed
                + model.moving_resident_gib * size.moving_voxels)
    floor = (model.floor_gib * fixed
             + model.moving_resident_gib * (size.moving_voxels - fixed))
    peaks = []
    for index, chunks in enumerate(channel_ranges(model.channels, gpus, chunk)):
        work = 0.0
        for start, stop in chunks:
            labels = max(0, min(stop, model.labels) - start)
            images = (stop - start) - labels
            work = max(work, labels * model.label_gib + images * model.image_gib)
        # Each chunk carries its own copy of the mask channel.
        stack = ((chunks[-1][1] - chunks[0][0]) + len(chunks)) * CHANNEL_GIB
        peak = stack * (fixed + moving) + work * fixed
        if index == 0:
            peak = max(peak + resident, floor)
            if chunk is None and model.whole_label_gib:
                identity = (2 * (model.channels + 1) * CHANNEL_GIB + model.inputs_gib
                            + model.labels * model.whole_label_gib)
                peak = max(peak, identity * fixed)
            if gpus > 1:
                peak += MULTI_GPU_MAIN_GIB
        peaks.append(peak)
    return peaks


def choose_stage(model: StageMemory, devices: Sequence[torch.device],
                 free_gib: dict[torch.device, float], size: PairSize,
                 prefer_fewer_gpus: bool, chunk: int | str = AUTO) -> StagePlan:
    """Choose the GPUs and the chunk size of one stage.

    Parameters
    ----------
    model
        The memory model of the stage.
    devices
        The GPUs the stage may use, main GPU first. The stage uses the first
        ``n`` of them.
    free_gib
        Free memory of each GPU when the stage starts.
    size
        The size of the pair.
    prefer_fewer_gpus
        True for the serial rule (fewest GPUs first), False for the
        concurrent rule (fewest chunks first).
    chunk
        ``"auto"``, ``"full"`` (one chunk per GPU) or a channel count.

    Raises
    ------
    RuntimeError
        If no candidate fits and ``chunk`` is ``"auto"``.
    """
    if chunk == AUTO:
        sizes = [None, *CHUNK_SIZES]
    else:
        sizes = [None if chunk == FULL else int(chunk)]
    candidates = []
    for gpus in range(1, len(devices) + 1):
        if gpus > model.channels:
            break
        for channels in sizes:
            estimate = stage_estimate(model, gpus, channels, size)
            if all(_fits(e, free_gib[d])
                   for e, d in zip(estimate, devices[:gpus], strict=True)):
                chunks = sum(len(c) for c in channel_ranges(model.channels, gpus,
                                                            channels))
                key = (gpus, chunks) if prefer_fewer_gpus else (chunks, gpus)
                candidates.append((key, channels, gpus, estimate))
    if not candidates:
        listing = ", ".join(f"{d} {free_gib[d]:.1f} GiB free" for d in devices)
        if chunk == AUTO:
            raise RuntimeError(
                f"a registration stage does not fit on the GPUs given ({listing}); "
                "give more GPUs")
        # The user's setting is used even when the estimate says it does not
        # fit, because the estimate is conservative.
        estimate = stage_estimate(model, len(devices), sizes[0], size)
        log.warning("chunk setting %r may not fit on %s; estimated peak GiB %s",
                    chunk, listing, ", ".join(f"{e:.1f}" for e in estimate))
        return StagePlan(tuple(devices), sizes[0], tuple(estimate))
    _, channels, gpus, estimate = min(candidates, key=lambda c: c[0])
    return StagePlan(tuple(devices[:gpus]), channels, tuple(estimate))


def free_memory(devices: Sequence[torch.device]) -> dict[torch.device, float]:
    """Free memory (GiB) of each GPU, as the driver reports it now."""
    free = {}
    for device in devices:
        available, _ = torch.cuda.mem_get_info(device)
        free[device] = available / GIB
    return free


def _parse_gpus(spec: str, count: int | None) -> list[int]:
    """Turn ``"0,1"`` into GPU indices.

    The indices are checked against ``count``, the number of visible GPUs,
    unless it is ``None``.
    """
    indices = [int(part) for part in spec.split(",") if part.strip()]
    if not indices:
        raise ValueError(f"no GPU index in {spec!r}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"a GPU is listed twice: {spec}")
    for index in indices:
        if index < 0:
            raise ValueError(f"GPU {index} does not exist")
        if count is not None and index >= count:
            raise ValueError(f"GPU {index} does not exist; {count} GPU(s) visible")
    return indices


def gpu_indices(spec: str | None) -> list[int]:
    """Turn ``"0,1"`` or ``None`` (every visible GPU) into GPU indices.

    The indices are those of this process's visible GPUs.
    """
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("no CUDA device is visible")
    if spec is None:
        return list(range(count))
    return _parse_gpus(spec, count)


def visible_devices_value(indices: Sequence[int], environ: dict[str, str]) -> str:
    """The ``CUDA_VISIBLE_DEVICES`` value that shows only the given GPUs.

    ``indices`` refer to the GPUs visible under ``environ``.
    """
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [v.strip() for v in visible.split(",") if v.strip()]
        return ",".join(entries[i] for i in indices)
    return ",".join(str(i) for i in indices)


def select_gpus(spec: str | None) -> list[torch.device]:
    """Restrict this process to the GPUs in ``spec`` and return them.

    The GPUs become the only visible ones, as ``cuda:0``, ``cuda:1`` and so
    on in the order given. Some copies between two GPUs make PyTorch open a
    CUDA context on every visible GPU; with the other GPUs hidden, a run
    takes no memory on GPUs it was not given. ``None`` keeps every visible
    GPU. This must run before CUDA is used in this process.
    """
    if spec is not None:
        if torch.cuda.is_initialized():
            raise RuntimeError("the GPUs must be selected before CUDA is used")
        # CUDA is not asked for the GPU count before the selection, because
        # CUDA would then keep the visible GPUs it found. The indices are
        # checked against CUDA_VISIBLE_DEVICES when it is set, and against
        # the GPUs that CUDA finds after the selection otherwise.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        count = len([v for v in visible.split(",") if v.strip()]) if visible else None
        indices = _parse_gpus(spec, count)
        value = visible_devices_value(indices, os.environ)
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        if torch.cuda.device_count() != len(indices):
            raise ValueError(f"GPUs {spec} are not all available "
                             f"(CUDA_VISIBLE_DEVICES={value})")
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("no CUDA device is visible")
    return [torch.device("cuda", i) for i in range(count)]


def _segmentation_gpu(devices: Sequence[torch.device],
                      free_gib: dict[torch.device, float],
                      workers: float) -> torch.device | None:
    """Return the GPU for the segmentors, or ``None``.

    This is the GPU other than the main GPU with the most free memory on
    which the segmentors fit (the last one listed on a tie).
    """
    usable = [d for d in devices[1:] if _fits(0.0, free_gib[d], workers)]
    if not usable:
        return None
    return max(reversed(usable), key=lambda d: free_gib[d])


def plan(devices: Sequence[torch.device], mode: str, fixed: Grid, moving: Grid,
         chunk_stage1: int | str = AUTO, chunk_stage2: int | str = AUTO,
         free_gib: dict[torch.device, float] | None = None) -> RunPlan:
    """Choose the run plan of one pair (see the module docstring).

    Parameters
    ----------
    devices
        The GPUs given for the pair, main GPU first.
    mode
        ``"auto"``, ``"serial"`` or ``"concurrent"``.
    fixed, moving
        The shape and the voxel size (mm) of the fixed and of the moving CT on
        their working grids (``regrep.io.working_grid_size``).
    chunk_stage1, chunk_stage2
        ``"auto"``, ``"full"`` or a channel count for each stage.
    free_gib
        Free memory of each GPU. ``None`` asks the driver.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    devices = list(devices)
    if not devices:
        raise ValueError("at least one GPU is needed")
    if free_gib is None:
        free_gib = free_memory(devices)
    size = PairSize.of(fixed, moving)
    main = devices[0]
    workers = size.workers_gib()
    features = size.features_gib()

    if mode == "auto":
        left_out = [d for d in devices[1:] if not _fits(0.0, free_gib[d], workers)]
        for device in left_out:
            log.info("GPU %d is left out: %.1f GiB free", device.index,
                     free_gib[device])
        devices = [d for d in devices if d not in left_out]
        if len(devices) >= 2 or _fits(features, free_gib[main], workers):
            mode = "concurrent"
        else:
            mode = "serial"

    segmentation = main
    stage1_devices = devices
    if mode == "concurrent" and len(devices) >= 2:
        segmentation = _segmentation_gpu(devices, free_gib, workers)
        if segmentation is None:
            raise RuntimeError(
                "the segmentors do not fit on any GPU other than the main GPU ("
                + ", ".join(f"{d} {free_gib[d]:.1f} GiB free" for d in devices[1:])
                + ")")
        stage1_devices = [d for d in devices if d != segmentation]
        if not _fits(features, free_gib[main]):
            raise RuntimeError(f"the features do not fit on {main} (estimated "
                               f"peak {features:.1f} GiB, {free_gib[main]:.1f} GiB "
                               "free)")
    elif mode == "concurrent":
        if not _fits(features, free_gib[main], workers):
            raise RuntimeError(
                f"the features and the segmentors do not fit on {main} at the "
                f"same time (estimated peaks {features:.1f} and {workers:.1f} GiB, "
                f"{free_gib[main]:.1f} GiB free); use the serial mode or give a "
                "second GPU")
    elif not _fits(features, free_gib[main]):
        raise RuntimeError(f"the features do not fit on {main} (estimated peak "
                           f"{features:.1f} GiB, {free_gib[main]:.1f} GiB free)")

    prefer_fewer_gpus = mode == "serial"
    stage1 = choose_stage(STAGE1, stage1_devices, free_gib, size, prefer_fewer_gpus,
                          chunk_stage1)
    stage2 = choose_stage(STAGE2, devices, free_gib, size, prefer_fewer_gpus,
                          chunk_stage2)
    return RunPlan(mode=mode, main_device=main, segmentation_device=segmentation,
                   stage1=stage1, stage2=stage2)


# CPU threads of each segmentation process, so that the two processes do not
# compete with the registration for the CPU.
WORKER_CPU_THREADS = 3


def child_environment(device: torch.device, environ: dict[str, str]) -> dict[str, str]:
    """Environment of a segmentation process that may use only ``device``.

    ``CUDA_VISIBLE_DEVICES`` is set to the one GPU behind ``device`` (taken
    from this process's own ``CUDA_VISIBLE_DEVICES`` when it is set), so the
    child sees it as ``cuda:0``. The CPU thread pools of the child are
    limited to ``WORKER_CPU_THREADS``.
    """
    env = dict(environ)
    env["CUDA_VISIBLE_DEVICES"] = visible_devices_value([device.index], environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "nnUNet_n_proc_DA"):
        env[name] = str(WORKER_CPU_THREADS)
    return env

"""The registration of one pair, from the four input files to the field.

:func:`register_pair` runs these steps:

1. Read the fixed and moving CT and PET, and resample the axes of each
   timepoint that are much finer than the working voxel size
   (``regrep.io.working_grid``). All later steps run on these working grids.
2. Segment: the TotalSegmentator maps of both CTs and the LION masks of
   the moving and then the fixed PET. In the serial mode this happens here,
   before the next step; in the concurrent mode two worker processes do it
   while the next steps run (see ``regrep.runtime``).
3. Load the anatomix network, seed every random generator with
   ``config.seed`` and make cuDNN deterministic.
4. Compute the features of the fixed and then the moving timepoint.
5. Build the stage-1 lesion support from the 6 mm organ map and the moving
   LION mask, and run stage 1.
6. Build the stage-2 lesion support from the 3 mm organ map and the moving
   LION mask, and apply the volume projection to the stage-1 transform.
7. Run stage 2 with the label maps and the LION masks of both timepoints.
8. Apply the final volume projection, the low pass and the fold repair,
   bring the field to the grid of the fixed CT file and the voxels of the
   moving CT file, and write it.

The field is written once, at the end, with an atomic rename. On any error
the worker processes are stopped, the workspace is removed (unless it is
kept), no output file is written and the error is raised again.

The memory figures of the plan assume the ``expandable_segments`` setting
of PyTorch's CUDA allocator (``PYTORCH_CUDA_ALLOC_CONF``), which the command
line sets before it imports torch.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterator

import numpy as np
import torch

from regrep.config import ChainConfig, GridConfig
from regrep.io import (
    PairInputs,
    Volume,
    Workspace,
    check_axis_lengths,
    check_output,
    load_volume,
    orient_like,
    same_grid,
    to_working_grid,
    working_grid,
)
from regrep.runtime import RunPlan, child_environment
from regrep.segmentation import masks
from regrep.segmentation.labels import ORGAN_MAPS
from regrep.segmentation.worker import LESION_MAPS, segment_lesions, segment_organs

log = logging.getLogger(__name__)

# Longest time a worker process may take for all its maps.
WORKER_TIMEOUT_S = 900.0
# Interval at which the main process looks for a worker's output.
POLL_S = 0.1


class _Clock:
    """Logs the time and the peak GPU memory of each step on the given GPUs."""

    def __init__(self, devices: list[torch.device]):
        self.devices = devices

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        for device in self.devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        yield
        for device in self.devices:
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        memory = ", ".join(
            f"cuda:{d.index} {torch.cuda.max_memory_allocated(d) / 2**30:.2f}"
            for d in self.devices)
        log.info("%-18s %6.1f s   peak GiB %s", name, seconds, memory)


class Models:
    """Models that stay loaded from one pair to the next.

    Only the anatomix network is kept, because it is small. The
    segmentation models are loaded for each pair and released before the
    registration, so that their memory is free for it.
    """

    def __init__(self):
        self.backbone_model = None

    def backbone(self, config: ChainConfig, device: torch.device) -> torch.nn.Module:
        """The anatomix network of ``config`` on ``device``, loaded on first use.

        A process uses one network on one device.
        """
        from regrep.features import load_backbone

        if self.backbone_model is None:
            self.backbone_model = load_backbone(config.features, device)
        return self.backbone_model


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, and make cuDNN deterministic.

    The TF32 settings of PyTorch are left at their defaults.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _InProcessMaps:
    """Segmentation maps computed in this process (the serial mode)."""

    def __init__(self):
        self._maps: dict[str, np.ndarray] = {}

    def put(self, name: str, array: np.ndarray) -> None:
        self._maps[name] = np.ascontiguousarray(array, dtype=np.uint8)

    def get(self, name: str) -> np.ndarray:
        """Return one map as a uint8 ``(X, Y, Z)`` array."""
        return self._maps[name]

    def wait_exit(self) -> None:
        """Nothing to wait for: the maps are complete."""

    def stop(self) -> None:
        """Nothing to stop."""

    def log_lines(self) -> list[str]:
        return []


def worker_commands(inputs: PairInputs, workspace: Workspace,
                    config_path: str) -> dict[str, list[str]]:
    """The command lines of the two segmentation workers of a pair.

    Each worker sees only its own GPU, as ``cuda:0``. Both workers read the
    working grids from the headers of the CTs.
    """
    base = [sys.executable, "-m", "regrep.segmentation.worker"]
    common = ["--fixed-ct", inputs.fixed_ct, "--moving-ct", inputs.moving_ct,
              "--workspace", workspace.root, "--config", config_path]
    return {
        "organs": base + ["organs"] + common,
        "lesions": base + ["lesions", "--fixed-pet", inputs.fixed_pet,
                           "--moving-pet", inputs.moving_pet] + common,
    }


class _WorkerMaps:
    """Segmentation maps written by two worker processes (the concurrent mode).

    The processes start when the object is created; the output of each goes
    to ``<task>.log`` in the workspace. :meth:`get` waits for one map and
    raises as soon as the worker that writes it has failed.

    Parameters
    ----------
    workspace
        The workspace the workers write to.
    commands
        The command line of the ``"organs"`` and the ``"lesions"`` worker.
    env
        The environment of the workers.
    """

    def __init__(self, workspace: Workspace, commands: dict[str, list[str]],
                 env: dict[str, str]):
        self.workspace = workspace
        self.started = time.perf_counter()
        self.processes: dict[str, subprocess.Popen] = {}
        self.logs: dict[str, str] = {}
        try:
            for task, command in commands.items():
                path = os.path.join(workspace.root, f"{task}.log")
                self.logs[task] = path
                with open(path, "wb") as handle:
                    self.processes[task] = subprocess.Popen(
                        command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL)
        except BaseException:
            # A worker that has started must not outlive a failed start.
            self.stop()
            raise

    @staticmethod
    def _task(name: str) -> str:
        if name in ORGAN_MAPS:
            return "organs"
        if name in LESION_MAPS:
            return "lesions"
        raise KeyError(name)

    def _tail(self, task: str, lines: int = 30) -> str:
        with open(self.logs[task], errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])

    def _check(self, task: str) -> None:
        """Raise if a worker has exited with an error."""
        code = self.processes[task].poll()
        if code not in (None, 0):
            raise RuntimeError(f"the {task} segmentation failed with exit status "
                               f"{code}:\n{self._tail(task)}")

    def get(self, name: str) -> np.ndarray:
        """Wait for one map and read it as a uint8 ``(X, Y, Z)`` array."""
        task = self._task(name)
        process = self.processes[task]
        while not self.workspace.exists(name):
            self._check(task)
            if process.poll() == 0 and not self.workspace.exists(name):
                raise RuntimeError(f"the {task} segmentation ended without writing "
                                   f"{name}:\n{self._tail(task)}")
            if time.perf_counter() - self.started > WORKER_TIMEOUT_S:
                raise TimeoutError(f"the {task} segmentation did not write {name} "
                                   f"within {WORKER_TIMEOUT_S:.0f} s")
            time.sleep(POLL_S)
        return self.workspace.read_labels(name)

    def wait_exit(self) -> None:
        """Wait until both workers have exited; raise if one failed."""
        for task, process in self.processes.items():
            remaining = WORKER_TIMEOUT_S - (time.perf_counter() - self.started)
            try:
                process.wait(timeout=max(remaining, 1.0))
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(f"the {task} segmentation did not finish within "
                                   f"{WORKER_TIMEOUT_S:.0f} s") from error
            self._check(task)

    def stop(self) -> None:
        """Stop the workers that are still running."""
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self.processes.values():
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def log_lines(self) -> list[str]:
        """The lines that the workers printed."""
        lines = []
        for path in self.logs.values():
            with open(path, errors="replace") as handle:
                lines += [line.rstrip() for line in handle
                          if line.startswith("[worker")]
        return lines


def _read_inputs(inputs: PairInputs, cfg: GridConfig
                 ) -> tuple[Volume, tuple[Volume, Volume, Volume, Volume]]:
    """Read the four volumes and bring them to the working grids.

    Each CT and PET must share one grid (``regrep.io.same_grid``); the fixed
    and the moving timepoint may have different grids. A timepoint whose CT
    has an axis much finer than the working voxel size
    (``regrep.io.working_grid``) has its CT and PET resampled onto the
    working grid of the CT. Each working grid needs at least 32 voxels along
    each axis (``regrep.io.check_axis_lengths``). Then the array axes of the
    moving CT and PET are reordered to the axis orientation of the fixed CT
    (``regrep.io.orient_like``). The segmentation workers do the same.

    Returns
    -------
    tuple
        The fixed CT as read, whose grid is the grid of the output field, and
        the fixed CT and PET and the moving CT and PET on the working grids.
    """
    volumes = tuple(load_volume(path) for path in (
        inputs.fixed_ct, inputs.fixed_pet, inputs.moving_ct, inputs.moving_pet))
    for ct, pet in (volumes[:2], volumes[2:]):
        if not same_grid(ct.shape, ct.affine, pet.shape, pet.affine):
            raise ValueError(
                f"the CT and the PET of a timepoint must share one grid: {ct.path} "
                f"has shape {ct.shape} and affine\n{ct.affine}\n{pet.path} has "
                f"shape {pet.shape} and affine\n{pet.affine}")
    working = []
    for side, (ct, pet) in (("fixed", volumes[:2]), ("moving", volumes[2:])):
        grid = working_grid(ct.shape, ct.affine, cfg)
        check_axis_lengths(ct.shape if grid is None else grid[0], ct.path)
        if grid is not None:
            log.info("%s timepoint: %s voxels of %s mm are resampled to %s voxels "
                     "of %s mm (the working grid)", side, _size(ct.shape),
                     _size(ct.spacing), _size(grid[0]),
                     _size(np.linalg.norm(grid[1][:3, :3], axis=0)))
        working += [to_working_grid(ct, grid), to_working_grid(pet, grid)]
    fixed_ct, fixed_pet, moving_ct, moving_pet = working
    return volumes[0], (fixed_ct, fixed_pet, orient_like(moving_ct, fixed_ct.affine),
                        orient_like(moving_pet, fixed_ct.affine))


def _size(values) -> str:
    """Format a shape or a voxel size as ``a x b x c``."""
    return " x ".join(f"{float(v):g}" for v in values)


def _to_file_grids(field: np.ndarray, fixed_file: Volume, fixed_ct: Volume,
                   moving_ct: Volume, device: torch.device) -> np.ndarray:
    """Bring a field from the working grids to the grids of the input files.

    ``field`` is on the working fixed grid and points to the voxels of the
    working moving CT. The result is on the grid of ``fixed_file``, the fixed
    CT as read, and points to the voxels of the moving CT file.
    """
    from regrep.posthoc.export import resample_field, to_file_voxels

    if fixed_ct is not fixed_file:
        field = resample_field(field, fixed_ct.affine, moving_ct.affine,
                               fixed_file.shape, fixed_file.affine, device)
    if moving_ct.file_index is not None:
        field = to_file_voxels(field, moving_ct.file_index)
    return field


def _empty_cache(devices) -> None:
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def register_pair(inputs: PairInputs, output_path: str, plan: RunPlan,
                  config: ChainConfig | None = None, keep_workspace: bool = False,
                  models: Models | None = None) -> float:
    """Register one pair and write the displacement field.

    The call changes process-wide state of torch and does not restore it: it
    seeds the global random generators of Python, NumPy and torch, makes
    cuDNN deterministic and sets the current CUDA device to the main GPU of
    the plan.

    Parameters
    ----------
    inputs
        The fixed (baseline) and moving (follow-up) CT and PET files.
    output_path
        The field file to write: ``(3, X, Y, Z)`` float32 on the grid of the
        fixed CT file, a displacement in voxels: fixed voxel ``x`` maps to
        voxel ``x + u(x)`` of the moving CT file. It must not be one of the
        input files.
    plan
        The run plan from ``regrep.runtime.plan``.
    config
        The chain settings; ``None`` uses the defaults.
    keep_workspace
        Keep the workspace (worker logs and label maps) after a concurrent
        run. The workspace is created in ``TMPDIR``.
    models
        Models kept between pairs; ``None`` loads them for this pair only.

    Returns
    -------
    float
        The time of the registration in seconds.
    """
    from regrep.registration.chunked import check_fused_ops

    check_output(output_path, (inputs.fixed_ct, inputs.fixed_pet, inputs.moving_ct,
                               inputs.moving_pet))
    config = config if config is not None else ChainConfig()
    models = models if models is not None else Models()
    devices = list(dict.fromkeys(
        [plan.main_device, *plan.stage1.devices, *plan.stage2.devices]))
    clock = _Clock(devices)
    started = time.perf_counter()
    workspace = maps = None
    log.info("plan: %s", plan.describe())
    try:
        torch.cuda.set_device(plan.main_device)
        check_fused_ops(plan.main_device)
        if plan.mode == "concurrent":
            workspace = Workspace(keep=keep_workspace)
            if keep_workspace:
                log.info("workspace: %s", workspace.root)
            config_path = os.path.join(workspace.root, "config.json")
            with open(config_path, "w") as handle:
                json.dump(dataclasses.asdict(config), handle)
            maps = _WorkerMaps(workspace,
                               worker_commands(inputs, workspace, config_path),
                               child_environment(plan.segmentation_device,
                                                 os.environ))
        with clock.step("read"):
            fixed_file, (fixed_ct, fixed_pet, moving_ct, moving_pet) = _read_inputs(
                inputs, config.grid)
        if plan.mode == "serial":
            maps = _InProcessMaps()
            with clock.step("segment organs"):
                segment_organs(fixed_ct, moving_ct, config, plan.segmentation_device,
                               maps.put)
            with clock.step("segment lesions"):
                segment_lesions(fixed_pet, moving_pet, config,
                                plan.segmentation_device, maps.put)
        field = _register(fixed_ct, fixed_pet, moving_ct, moving_pet, maps, plan,
                          config, models, clock, devices)
        field = _to_file_grids(field, fixed_file, fixed_ct, moving_ct,
                               plan.main_device)
        # A worker that fails after its last map still fails the pair.
        maps.wait_exit()
        worker_log = maps.log_lines()
        with clock.step("write"):
            from regrep.posthoc.export import write_field

            write_field(output_path, field, fixed_file)
    except BaseException:
        if maps is not None:
            maps.stop()
        raise
    finally:
        if workspace is not None:
            workspace.close()
    seconds = time.perf_counter() - started
    for line in worker_log:
        log.info("%s", line)
    log.info("done in %.1f s: %s", seconds, output_path)
    return seconds


def _register(fixed_ct: Volume, fixed_pet: Volume, moving_ct: Volume,
              moving_pet: Volume, maps, plan: RunPlan, config: ChainConfig,
              models: Models, clock: _Clock, devices) -> np.ndarray:
    """The steps from the features to the finished field."""
    from regrep.features import extract_side
    from regrep.posthoc import export, projection
    from regrep.registration import stage1, stage2
    from regrep.registration.transforms import VoxelMap

    main = plan.main_device

    def label_map(name: str) -> torch.Tensor:
        """One segmentation map as a uint8 ``(X, Y, Z)`` tensor on the main GPU."""
        return torch.from_numpy(maps.get(name)).to(main)

    with clock.step("backbone"):
        model = models.backbone(config, main)
    # Building the network draws random numbers, but only when it is not
    # loaded yet. Seeding after it gives every pair the same random state.
    seed_everything(config.seed)
    with clock.step("features fixed"):
        fixed = extract_side(fixed_ct, fixed_pet, model, config.features, main,
                             sum_mm=None)
    # The moving features use the pyramid of the fixed CT, so that both sides
    # are described at the same scales.
    with clock.step("features moving"):
        moving = extract_side(moving_ct, moving_pet, model, config.features, main,
                              sum_mm=min(fixed.spacing))
    del model
    _empty_cache(devices)

    waited = plan.waits_for_segmentation(plan.stage1)
    if waited:
        with clock.step("wait segmentation"):
            maps.wait_exit()
    with clock.step("support stage 1"):
        lion_moving = label_map("lion_moving")
        support1 = masks.lesion_support(label_map("organs6_moving"), moving_pet,
                                        lion_moving, config.lesions)
    with clock.step("stage 1"):
        grid1 = stage1.run(fixed, moving, support1, plan.stage1, config.stage1)
    del support1
    fixed.release_stage1_inputs()
    moving.release_stage1_inputs()
    _empty_cache(devices)

    with clock.step("support stage 2"):
        support2 = masks.lesion_support(label_map("organs3_moving"), moving_pet,
                                        lion_moving, config.lesions)
        labels_fixed = label_map("labels_fixed")
        labels_moving = label_map("labels_moving")
        lion_fixed = label_map("lion_fixed")
    voxel_map = VoxelMap.between(fixed_ct.affine, moving_ct.affine)
    with clock.step("project stage 1"):
        grid1 = projection.project_grid(grid1, support2, config.posthoc, voxel_map)
    _empty_cache(devices)

    if plan.waits_for_segmentation(plan.stage2) and not waited:
        with clock.step("wait segmentation"):
            maps.wait_exit()
    with clock.step("stage 2"):
        composed = stage2.run(grid1, fixed, moving, labels_fixed, labels_moving,
                              support2, lion_fixed, lion_moving, plan.stage2,
                              config.stage2)
    body = fixed.body[0, 0].permute(2, 1, 0) > 0
    del grid1, fixed, moving, labels_fixed, labels_moving, lion_fixed, lion_moving
    _empty_cache(devices)

    with clock.step("export"):
        return export.finalize(composed, fixed_ct, body, support2, config.posthoc,
                               voxel_map)

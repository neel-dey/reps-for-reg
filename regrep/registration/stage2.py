"""Stage 2: the label-driven deformable registration, and its composition.

Stage 2 starts from the stage-1 grid and registers two channel stacks on the
fixed grid. Each stack holds one banded distance channel per TotalSegmentator
structure present on both sides, one channel for the LION lesion mask when
both timepoints have a lesion, the 32 anatomix channels and the body mask.
The moving stack is built from the moving labels and features after the
stage-1 warp.

The objective is a weighted sum of a generalized Dice loss and a chamfer
distance on the structure channels, a masked local NCC on the anatomix
channels, a region-distance term and a volume penalty on the lesion support.
The Dice, chamfer, LNCC and region terms are scaled by their values at the
identity, so each starts at its configured share. The volume penalty has a
fixed multiplier.

The result is the composed sampling grid ``T1(T2(x))``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from regrep.config import Stage2Config
from regrep.registration import losses
from regrep.registration.chunked import (
    ChunkedGreedyRegistration,
    StackImages,
    masked_mean,
    split_channels,
)
from regrep.registration.distance import distance_channels
from regrep.registration.transforms import compose_grids, warp_by, xyz_to_zyx
from regrep.segmentation.labels import NUM_CLASSES


def _labels(labels_xyz: torch.Tensor) -> torch.Tensor:
    """A label map as an int64 ``(1, 1, Z, Y, X)`` tensor.

    The merged maps hold ids 0 to ``NUM_CLASSES`` only.
    """
    return xyz_to_zyx(labels_xyz).long()


def present_ids(fixed: torch.Tensor, moved: torch.Tensor) -> list[int]:
    """Structure ids with at least one voxel in both label maps, ascending."""
    def counts(volume):
        return torch.bincount(volume.reshape(-1), minlength=NUM_CLASSES + 1).cpu()

    fixed_counts, moved_counts = counts(fixed), counts(moved)
    return [i for i in range(1, NUM_CLASSES + 1)
            if fixed_counts[i] >= 1 and moved_counts[i] >= 1]


@dataclass
class Stage2Problem:
    """Everything the stage-2 optimization reads.

    :func:`optimize` sets the two stacks to ``None`` once the registration
    holds them, so that the registration can free them after it has copied
    its channel chunks.

    Attributes
    ----------
    fixed_stack, moving_stack
        ``(1, C, Z, Y, X)`` float32 stacks on the fixed grid: structure
        channels, then anatomix channels, then the body mask.
    objective
        The image term (Dice, chamfer and LNCC), evaluated in channel chunks.
    displacement_term
        The region term plus the volume penalty, a function of the stage-2
        displacement.
    """

    fixed_stack: torch.Tensor | None
    moving_stack: torch.Tensor | None
    objective: losses.Stage2Objective
    displacement_term: Callable[[torch.Tensor], torch.Tensor]


def _assemble(labels, ids, lesion, amix, mask, spacing, band) -> torch.Tensor:
    """Write the structure, anatomix and mask channels into one stack."""
    count = len(ids)
    structures = count + (0 if lesion is None else 1)
    extra = int(amix.shape[1])
    shape = tuple(int(s) for s in mask.shape[2:])
    stack = torch.empty((1, structures + extra + 1) + shape, device=mask.device,
                        dtype=torch.float32)
    distance_channels(labels, ids, spacing, band, out=stack[:, :count])
    if lesion is not None:
        distance_channels(lesion, [1], spacing, band, out=stack[:, count:structures])
    torch.mul(amix, (mask > 0).to(stack.dtype),
              out=stack[:, structures:structures + extra])
    stack[:, structures + extra:] = mask
    return stack


def _channel_chunks(start: int, stop: int, size: int | None) -> list[tuple[int, int]]:
    """Split the channel range ``[start, stop)`` into pieces of ``size``.

    All pieces but the last have ``size`` channels. The pieces only bound the
    memory of the values at the identity; the optimization uses the chunks
    of ``regrep.registration.chunked.split_channels``.
    """
    size = stop - start if size is None else int(size)
    return [(c, min(c + size, stop)) for c in range(start, stop, size)]


@torch.no_grad()
def _identity_values(fixed: torch.Tensor, moving: torch.Tensor, count: int,
                     dice_w: torch.Tensor, cham_w: torch.Tensor, shell: float,
                     chunk: int | None) -> tuple[float, float, float]:
    """Dice loss, chamfer distance (band units) and LNCC loss at the identity.

    The two stacks are compared without a stage-2 warp, under the product of
    their body-mask channels. ``count`` is the number of structure
    channels, and ``chunk`` the largest number of channels per piece.
    """
    mask = moving[:, -1:] * fixed[:, -1:]
    totals = torch.zeros(3, device=fixed.device)
    for c0, c1 in _channel_chunks(0, count, chunk):
        totals += losses.label_sums(moving[:, c0:c1], fixed[:, c0:c1], mask,
                                    dice_w[c0:c1], cham_w[c0:c1], shell)
    dice, chamfer = losses.dice_and_chamfer(totals, cham_w.sum())
    lncc = losses.make_lncc().to(fixed.device)
    numerator = torch.zeros(1, device=fixed.device)
    for c0, c1 in _channel_chunks(count, fixed.shape[1] - 1, chunk):
        numerator += losses.lncc_sum(lncc, moving[:, c0:c1], fixed[:, c0:c1], mask)
    cc = masked_mean(numerator, mask.sum())
    return float(dice), float(chamfer), float(cc)


def prepare(grid1: torch.Tensor, amix_fixed: torch.Tensor, amix_moving: torch.Tensor,
            body_fixed: torch.Tensor, body_moving: torch.Tensor,
            labels_fixed: torch.Tensor, labels_moving: torch.Tensor,
            support: torch.Tensor, lion_fixed: torch.Tensor,
            lion_moving: torch.Tensor, spacing_fixed: tuple[float, float, float],
            spacing_moving: tuple[float, float, float],
            cfg: Stage2Config, chunk: int | None) -> Stage2Problem:
    """Build the stacks, the weights and the terms of stage 2.

    All volumes in ``(X, Y, Z)`` order are tensors on the device of
    ``grid1``.

    Parameters
    ----------
    grid1
        Stage-1 sampling grid ``(1, Z, Y, X, 3)`` on the registration device.
    amix_fixed, amix_moving
        Anatomix features ``(1, 32, Z, Y, X)`` of each side on its own grid.
    body_fixed, body_moving
        Body masks ``(1, 1, Z, Y, X)``, float {0, 1}, on each side's grid.
    labels_fixed, labels_moving
        TotalSegmentator label maps ``(X, Y, Z)`` on each side's CT grid.
    support
        Stage-2 lesion support ``(X, Y, Z)``, boolean, on the moving grid.
    lion_fixed, lion_moving
        LION lesion masks ``(X, Y, Z)`` on each side's grid.
    spacing_fixed, spacing_moving
        Voxel size in mm along ``(Z, Y, X)`` of each side.
    cfg
        Stage-2 settings.
    chunk
        Largest number of channels per chunk for the values at the identity.
        ``None`` processes each block in one piece.
    """
    device = grid1.device
    band = cfg.band_mm
    fixed_zyx = tuple(float(s) for s in spacing_fixed)
    moving_zyx = tuple(float(s) for s in spacing_moving)

    fixed_labels = _labels(labels_fixed)
    moving_labels = _labels(labels_moving)
    warped_labels = warp_by(moving_labels.float(), grid1, "nearest").round().long()
    ids = present_ids(fixed_labels, warped_labels)
    if not ids:
        raise ValueError("stage 2 needs at least one TotalSegmentator class in both "
                         "CTs, but no class is in the fixed labels and in the "
                         "moving labels after stage 1")
    warped_mask = warp_by(body_moving, grid1, "nearest")

    lesion_fixed = (xyz_to_zyx(lion_fixed) > 0).float()
    lesion_moving = warp_by((xyz_to_zyx(lion_moving) > 0).float(), grid1, "nearest")
    if float(lesion_fixed.sum()) == 0 or float(lesion_moving.sum()) == 0:
        lesion_fixed = lesion_moving = None
    else:
        lesion_fixed = lesion_fixed[0, 0].to(torch.uint8)
        lesion_moving = lesion_moving[0, 0].to(torch.uint8)

    fixed_stack = _assemble(fixed_labels, ids, lesion_fixed, amix_fixed, body_fixed,
                            fixed_zyx, band)
    warped_amix = warp_by(amix_moving, grid1, "bilinear")
    moving_stack = _assemble(warped_labels, ids, lesion_moving, warped_amix,
                             warped_mask, fixed_zyx, band)
    has_lesion = lesion_fixed is not None
    del warped_amix, lesion_fixed, lesion_moving

    organs = len(ids)
    count = organs + int(has_lesion)
    sizes = losses.structure_sizes(fixed_stack[:, :count], moving_stack[:, :count])
    dice_w = losses.structure_weights(sizes[:organs], cfg.dice_size_exponent)
    # Every organ has chamfer weight 1.
    cham_w = torch.ones_like(dice_w)
    if has_lesion:
        tail = torch.tensor([cfg.lesion_channel_weight], dtype=dice_w.dtype,
                            device=device)
        dice_w = torch.cat([dice_w, tail])
        cham_w = torch.cat([cham_w, tail])

    shell = cfg.shell_voxels * min(fixed_zyx) / band
    dice0, cham0, cc0 = _identity_values(fixed_stack, moving_stack, count, dice_w,
                                         cham_w, shell, chunk)
    dice_scale = 1.0 / max(dice0, 1e-6)
    chamfer_scale = 1.0 / max(cham0, 1e-6)

    native = distance_channels(moving_labels, ids, moving_zyx, band)
    points = losses.interior_points(fixed_labels, ids, body_fixed,
                                    cfg.region_points, cfg.region_seed)
    region = losses.RegionDistance(points, native, grid1)
    del native, points
    with torch.no_grad():
        region0 = float(region.value())
    region.coeff = cfg.region_share / max(region0, 1e-6)

    weight = warp_by(xyz_to_zyx(support).float(), grid1, "nearest")
    penalty = losses.VolumePenalty(weight, cfg.volume_penalty)
    del weight

    def displacement_term(disp: torch.Tensor) -> torch.Tensor:
        return region(disp) + penalty(disp)

    share = 1.0 - cfg.region_share
    inner = cfg.chamfer_share / share
    objective = losses.Stage2Objective(
        count, dice_w, cham_w, shell,
        dice_coeff=(1.0 - inner) * (share * dice_scale),
        chamfer_coeff=inner * (share * chamfer_scale),
        lncc_coeff=cfg.amix_share / max(abs(cc0), 1e-6))
    return Stage2Problem(fixed_stack, moving_stack, objective, displacement_term)


def optimize(problem: Stage2Problem, torch2phy: torch.Tensor, phy2torch: torch.Tensor,
             devices, chunk_channels: int | None, cfg: Stage2Config) -> torch.Tensor:
    """Run the stage-2 optimization and return its sampling grid.

    Parameters
    ----------
    problem
        The output of :func:`prepare`. Its two stacks are handed over to the
        registration, and ``problem.fixed_stack`` and
        ``problem.moving_stack`` are set to ``None``.
    torch2phy, phy2torch
        Geometry of the fixed CT. Both stacks are on the fixed grid, so both
        use it.
    devices
        Devices for the channel chunks; the first must hold the stacks.
    chunk_channels
        Largest number of image channels in one chunk, or ``None`` for one
        chunk per device.
    cfg
        Stage-2 settings.

    Returns
    -------
    torch.Tensor
        ``grid2``, the ``(1, Z, Y, X, 3)`` stage-2 sampling grid.
    """
    fixed_images = StackImages(problem.fixed_stack, torch2phy, phy2torch)
    moving_images = StackImages(problem.moving_stack, torch2phy, phy2torch)
    chunks = split_channels(problem.fixed_stack.shape[1] - 1, devices, chunk_channels)
    # The registration owns the stacks from here on. Without these references
    # it can free them once it has copied its channel chunks.
    problem.fixed_stack = problem.moving_stack = None
    reg = ChunkedGreedyRegistration(
        [1], [cfg.iterations], fixed_images, moving_images, problem.objective, chunks,
        optimizer_lr=cfg.learning_rate, smooth_grad_sigma=cfg.grad_sigma,
        smooth_warp_sigma=cfg.warp_sigma,
        displacement_reg=problem.displacement_term, tolerance=cfg.tolerance,
        max_tolerance_iters=cfg.tolerance_window)
    reg.optimize()
    return reg.get_warped_coordinates(fixed_images, moving_images).detach()


def run(grid1: torch.Tensor, fixed, moving, labels_fixed: torch.Tensor,
        labels_moving: torch.Tensor, support2: torch.Tensor,
        lion_fixed: torch.Tensor, lion_moving: torch.Tensor, stage_plan,
        cfg: Stage2Config) -> torch.Tensor:
    """Run stage 2 and compose it with stage 1.

    All volumes in ``(X, Y, Z)`` order are tensors on the main registration
    device.

    Parameters
    ----------
    grid1
        Stage-1 sampling grid ``(1, Z, Y, X, 3)``, after the stage-1 volume
        projection, on the main registration device.
    fixed, moving
        ``SideFeatures`` of the two timepoints.
    labels_fixed, labels_moving
        TotalSegmentator label maps ``(X, Y, Z)``, uint8, on each side's grid.
    support2
        Stage-2 lesion support ``(X, Y, Z)``, boolean, on the moving grid.
    lion_fixed, lion_moving
        LION lesion masks ``(X, Y, Z)`` on each side's grid.
    stage_plan
        The ``StagePlan`` of stage 2: ``devices`` (the first holds the
        stage-1 grid) and ``chunk_channels``.
    cfg
        Stage-2 settings.

    Returns
    -------
    torch.Tensor
        The composed grid ``T1(T2(x))``, ``(1, Z, Y, X, 3)``.
    """
    chunk = stage_plan.chunk_channels
    problem = prepare(grid1, fixed.amix, moving.amix, fixed.body, moving.body,
                      labels_fixed, labels_moving, support2, lion_fixed, lion_moving,
                      fixed.spacing, moving.spacing, cfg, chunk)
    grid2 = optimize(problem, fixed.torch2phy, fixed.phy2torch,
                     stage_plan.devices, chunk, cfg)
    del problem
    return compose_grids(grid1, grid2).detach()

"""Volume projection: keep the volume of each lesion through the warp.

The challenge scores two PET biomarkers, the lesion volume (MTV) and the lesion
uptake (TLG), before and after the warp. Both are preserved when the warp
neither compresses nor expands the lesions. This module corrects a finished
displacement field so that the warped lesion support keeps its volume:

1. Gate. Each connected component of the lesion support is warped with the
   scorer's nearest-neighbour sampler. Components whose voxel count changes
   by less than a factor ``exp(gate)`` are left out of the next step.
2. Pointwise passes. On the warped support of the remaining components, the
   field is corrected so that ``log det J`` goes to 0. The correction is the
   gradient of the solution of a Poisson equation whose right-hand side is
   ``-log det J``, computed with an FFT.
3. Count feedback. The voxel count of each warped component (the quantity
   MTV reads) is compared with its count before the warp, and the same kind
   of correction drives the counts together. The pass with the smallest total
   count error is kept.

A component that reaches the edge of the fixed grid after the warp can
extend beyond the fixed field of view, and then its warped voxel count does
not measure its volume. Such components are found once, before the
corrections, and the gate and the count feedback leave them out.

Each correction ``v`` is composed with the field: ``u'(x) = u(x + v(x)) +
v(x)``. The measurements can be taken on a filtered copy of the field
(``measure``); the corrections are always composed into the unfiltered field.

The support lives on the moving grid, which may differ from the fixed grid
(``regrep.registration.transforms.VoxelMap``). Volumes are then compared in
physical units: ``det J`` of the field is divided by the Jacobian determinant
of the identity transform, and the voxel count of a component before the warp
is converted to fixed voxels.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from regrep.config import PosthocConfig
from regrep.registration.transforms import (
    VoxelMap,
    dilate_box,
    disp_to_grid,
    grid_to_disp,
    spatial_transform,
)

log = logging.getLogger(__name__)

Measure = Callable[[torch.Tensor], torch.Tensor]


def _det_jacobian(disp):
    """Return ``det J`` of ``x + u(x)`` with central differences (voxel units).

    ``torch.gradient`` uses one-sided differences at the volume faces.
    """
    gx, gy, gz = torch.gradient(disp, dim=(0, 1, 2))
    j00 = gx[..., 0] + 1.0
    j01 = gx[..., 1]
    j02 = gx[..., 2]
    j10 = gy[..., 0]
    j11 = gy[..., 1] + 1.0
    j12 = gy[..., 2]
    j20 = gz[..., 0]
    j21 = gz[..., 1]
    j22 = gz[..., 2] + 1.0
    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )


def _smooth(volume, passes=3, radius=2):
    """Blur a volume with repeated box filters.

    The padded zeros count in the mean, so values drop near the volume faces.
    """
    size = 2 * int(radius) + 1
    out = volume[None, None]
    for _ in range(passes):
        out = F.avg_pool3d(out, kernel_size=size, stride=1, padding=int(radius))
    return out[0, 0]


def _poisson_gradient(rhs):
    """Solve ``lap p = rhs`` on a periodic grid and return ``grad p``.

    The Laplacian is the central-difference divergence of the central-difference
    gradient, whose Fourier symbol is ``-(sin^2 kx + sin^2 ky + sin^2 kz)``.
    The modes where the symbol is 0 (the mean and the Nyquist modes) are set
    to 0. The returned gradient is ``(X, Y, Z, 3)``.
    """
    freq = torch.fft.fftn(rhs)
    sins = []
    for axis, n in enumerate(rhs.shape):
        k = torch.fft.fftfreq(n, device=rhs.device)
        view = [1, 1, 1]
        view[axis] = n
        sins.append(torch.sin(2.0 * torch.pi * k).view(view))
    symbol = -(sins[0] ** 2 + sins[1] ** 2 + sins[2] ** 2)
    null = symbol.abs() < 1e-8
    safe = torch.where(null, torch.ones_like(symbol), symbol)
    p_hat = torch.where(null, torch.zeros_like(freq), freq / safe)
    p = torch.fft.ifftn(p_hat).real
    return torch.stack(torch.gradient(p, dim=(0, 1, 2)), dim=-1)


def _compose(disp, v, voxel_map: VoxelMap, identity):
    """Return ``u(x + v(x)) + v(x)``, sampled trilinearly with zero padding.

    ``identity`` is the displacement of the identity transform
    (``VoxelMap.displacement``). When it is not ``None``, only ``u -
    identity`` is sampled, and the exact value of the identity part at ``x +
    v(x)`` is added, so that the zero padding acts on the deformation alone.
    """
    if identity is not None:
        rest = disp - identity
        moved = torch.stack(
            [spatial_transform(rest[..., c], v, "bilinear") for c in range(3)], dim=-1
        )
        return moved + identity + voxel_map.apply_matrix(v)
    moved = torch.stack(
        [spatial_transform(disp[..., c], v, "bilinear") for c in range(3)], dim=-1
    )
    return moved + v


def _correction(rhs, region, cfg: PosthocConfig, cap):
    """Return the smoothed, windowed and bounded correction for one pass.

    ``rhs`` is the right-hand side of the Poisson equation and ``region`` the
    fixed-grid voxels it describes. The correction fades out over
    ``cfg.envelope`` voxels around the region and each component is bounded
    by ``cap`` voxels before the box smoothing.
    """
    v = _poisson_gradient(rhs)
    grown = dilate_box(region, (cfg.envelope,) * 3)
    window = _smooth(grown.float()).clamp(0.0, 1.0)
    v = (v * window[..., None]).clamp(-cap, cap)
    return torch.stack(
        [_smooth(v[..., c], passes=2, radius=cfg.smooth) for c in range(3)], dim=-1
    )


def _components(mask):
    """Label the 6-connected components of a boolean mask.

    Returns the float32 id map on the mask's device and the component count.
    """
    labels, count = ndimage.label(mask.cpu().numpy())
    return torch.as_tensor(labels.astype(np.float32), device=mask.device), count


def _fixed_voxel_counts(counts, jacobian):
    """Convert moving voxel counts to the fixed voxel counts of the same volume.

    The result is rounded to whole voxels. With equal voxel sizes the counts
    are returned unchanged.
    """
    return torch.round(counts.double() / abs(jacobian)).long()


def _warped_counts(ids, disp, count):
    """Warp a component id map and count the fixed voxels of each id."""
    warped = spatial_transform(ids, disp, "nearest").round().long()
    return warped, torch.bincount(warped.flatten(), minlength=count + 1)


def _on_faces(warped, count):
    """Return, for each id, whether the warped id map has it on a face voxel.

    ``warped`` is a warped component id map on the fixed grid. Entry ``k`` of
    the result is True when component ``k`` covers a voxel on one of the six
    faces of the grid; entry 0 is the background.
    """
    faces = torch.cat([warped[0].flatten(), warped[-1].flatten(),
                       warped[:, 0].flatten(), warped[:, -1].flatten(),
                       warped[:, :, 0].flatten(), warped[:, :, -1].flatten()])
    return torch.bincount(faces, minlength=count + 1) > 0


def _gate(mask, ids, count, after, observed, gate, jacobian):
    """Return the part of ``mask`` whose components change volume under the warp.

    A component is kept when it is ``observed`` and ``|log((after + 1) /
    (before + 1))| > gate``, with both counts in fixed voxels. A mask with a
    single observed component is returned unchanged.

    Parameters
    ----------
    mask
        Boolean mask on the moving grid.
    ids, count
        The component id map of ``mask`` and the number of components.
    after
        The fixed voxel count of each id under the warp, entry 0 for the
        background.
    observed
        For each id, whether its count can be compared; entry 0 is not used.
    gate
        The threshold.
    jacobian
        The Jacobian determinant of the identity transform.
    """
    if count == 1 and bool(observed[1]):
        return mask
    before = _fixed_voxel_counts(
        torch.bincount(ids.long().flatten(), minlength=count + 1), jacobian)
    ratio = (after[1:].float() + 1.0) / (before[1:].float() + 1.0)
    keep = (ratio.log().abs() > gate) & observed[1:]
    kept = int(keep.sum())
    log.debug("projection gate: %d of %d components have |log(count ratio)| above "
              "%.3f", kept, count, gate)
    if kept == 0:
        return torch.zeros_like(mask)
    return torch.isin(ids.long(), torch.nonzero(keep).flatten() + 1) & mask


def _pointwise_passes(disp, mask, cfg: PosthocConfig, measure: Measure,
                      voxel_map: VoxelMap, identity):
    """Drive ``log det J`` to 0 on the warped support of ``mask``.

    ``det J`` is divided by the Jacobian determinant of the identity
    transform first, so that 0 means no change of the physical volume.
    """
    jacobian = voxel_map.jacobian
    mask_f = mask.float()
    for index in range(cfg.passes):
        field = measure(disp)
        warped = spatial_transform(mask_f, field, "nearest") > 0.5
        support = dilate_box(warped, (cfg.margin,) * 3)
        det = (_det_jacobian(field) / jacobian).clamp_min(cfg.det_floor)
        d = torch.where(support, det.log(), torch.zeros_like(det))
        residual = float(d[support].abs().mean()) if support.any() else 0.0
        log.debug("projection pass %d: mean |log det J| %.4f", index, residual)
        if residual < 1e-3:
            break
        disp = _compose(disp, _correction(-d, support, cfg, cfg.cap), voxel_map,
                        identity)
    return disp


def _count_feedback(disp, ids, count, observed, cfg: PosthocConfig,
                    measure: Measure, voxel_map: VoxelMap, identity):
    """Drive the warped voxel count of each observed component to its count.

    ``ids`` and ``count`` are the component id map of the support and the
    number of components, and ``observed`` tells, for each id, whether its
    count can be compared (entry 0 is not used). The count before the warp is
    converted to fixed voxels. Returns the field of the pass with the
    smallest total count error over the observed components.
    """
    before = _fixed_voxel_counts(
        torch.bincount(ids.long().flatten(), minlength=count + 1), voxel_map.jacobian)
    observed = observed[1:]
    best_disp, best_error = disp, None
    for index in range(cfg.count_passes):
        warped, after = _warped_counts(ids, measure(disp), count)
        error = torch.where(observed, (after[1:] - before[1:]).abs(), 0)
        total = float(error.sum())
        active = error > 0
        log.debug("projection count pass %d: total count error %d voxels over %d "
                  "components", index, int(total), count)
        if best_error is None or total < best_error:
            best_disp, best_error = disp, total
        if not bool(active.any()):
            break
        # On each active warped component the right-hand side is
        # log(after / before). A component that covers too many fixed voxels
        # gets a correction with det J > 1 there, so fewer fixed voxels map
        # into it.
        active_ids = torch.nonzero(active).flatten() + 1
        lookup = torch.zeros(count + 1, device=disp.device)
        ratio = (after[active_ids].float().clamp_min(1.0)
                 / before[active_ids].float().clamp_min(1.0))
        lookup[active_ids] = ratio.log().clamp(-0.5, 0.5)
        region = torch.isin(warped, active_ids)
        disp = _compose(disp, _correction(lookup[warped], region, cfg, cfg.count_cap),
                        voxel_map, identity)
    _, after = _warped_counts(ids, measure(disp), count)
    total = float(torch.where(observed, (after[1:] - before[1:]).abs(), 0).sum())
    if best_error is not None and best_error < total:
        log.debug("projection count: keep the pass with error %d instead of %d",
                  int(best_error), int(total))
        disp = best_disp
    return disp


def project(
    disp: torch.Tensor,
    support: torch.Tensor,
    cfg: PosthocConfig,
    voxel_map: VoxelMap,
    measure: Measure | None = None,
) -> torch.Tensor:
    """Correct a field so that the lesion support keeps its volume.

    Parameters
    ----------
    disp
        Displacement ``(X, Y, Z, 3)`` in voxels: fixed voxel ``x`` maps to
        moving voxel ``x + u(x)``.
    support
        Lesion support on the moving grid, ``(X', Y', Z')`` boolean, on the
        device of ``disp``.
    cfg
        Projection settings.
    voxel_map
        The identity transform between the fixed and the moving grid.
    measure
        Filter applied to the field before every measurement (gate, det J,
        counts). The corrections are composed into the unfiltered field.
        ``None`` measures the field itself.

    Returns
    -------
    torch.Tensor
        The corrected displacement, same shape as ``disp``.
    """
    if measure is None:
        def measure(field):
            return field
    if not bool(support.any()):
        log.debug("projection: the lesion support is empty; field unchanged")
        return disp
    identity = voxel_map.displacement(disp.shape[:3], disp.device)
    ids, count = _components(support)
    warped, after = _warped_counts(ids, measure(disp), count)
    observed = ~_on_faces(warped, count)
    clipped = count - int(observed[1:].sum())
    if clipped:
        log.info("volume projection: %d of %d lesion components reach the edge of "
                 "the fixed grid; their volume is not corrected", clipped, count)
    gated = _gate(support, ids, count, after, observed, cfg.gate, voxel_map.jacobian)
    disp = _pointwise_passes(disp, gated, cfg, measure, voxel_map, identity)
    return _count_feedback(disp, ids, count, observed, cfg, measure, voxel_map,
                           identity)


def project_grid(grid: torch.Tensor, support: torch.Tensor, cfg: PosthocConfig,
                 voxel_map: VoxelMap) -> torch.Tensor:
    """Apply ``project`` to a sampling grid and return a sampling grid.

    This is the projection of the stage-1 transform: the measurements read
    the field directly.

    Parameters
    ----------
    grid
        ``(1, Z, Y, X, 3)`` normalized sampling grid on the fixed grid.
    support
        Lesion support on the moving grid, ``(X', Y', Z')`` boolean, on the
        device of ``grid``.
    cfg
        Projection settings.
    voxel_map
        The identity transform between the fixed and the moving grid.
    """
    with torch.no_grad():
        disp = project(grid_to_disp(grid, support.shape), support, cfg, voxel_map)
        return disp_to_grid(disp, support.shape).to(grid.dtype)

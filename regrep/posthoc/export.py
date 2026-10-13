"""From the composed transform to the challenge displacement field.

``finalize`` runs the last steps on the composed sampling grid, on the
working grids (``regrep.io.working_grid``):

1. the volume projection of the lesion support (``regrep.posthoc.projection``).
   Its gate, ``det J`` and voxel counts are measured on a smoothed copy of
   the field: a trilinear downsampling by 2 and an upsampling back to the
   full size. The corrections are composed into the field itself,
2. a Gaussian low pass of the displacement (sigma ``export_sigma``),
3. the exterior fold repair: where the field folds outside the body, it is
   blurred locally until the folds go away. Voxels in the body and in a
   collar of ``repair_margin_mm`` around it never change.

``resample_field`` and ``to_file_voxels`` then bring the field from the
working grids to the grid of the fixed CT file and the voxels of the moving
CT file. ``write_field`` stores the result in the layout of the challenge
template. ``warp_to_fixed`` and ``write_moved`` warp the moving CT and PET
files with this field and store them on the grid of the fixed CT file.
"""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from regrep.config import PosthocConfig
from regrep.io import Volume, save_nifti
from regrep.posthoc.folds import ndv_map, scored_voxels
from regrep.posthoc.projection import project
from regrep.registration.transforms import (
    VoxelMap,
    box_radii,
    dilate_box,
    grid_to_disp,
    spatial_transform,
)

log = logging.getLogger(__name__)


def gaussian_blur(disp: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur each component of an ``(X, Y, Z, 3)`` field with a Gaussian.

    The kernel has radius ``max(1, int(3 * sigma + 0.5))`` voxels and is
    applied along X, then Y, then Z, with the edge values repeated at the
    volume faces.
    """
    radius = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=disp.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    field = disp.permute(3, 0, 1, 2).unsqueeze(0)
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        # F.pad lists the last axis first.
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = radius
        pad[2 * (2 - axis) + 1] = radius
        padded = F.pad(field, pad, mode="replicate")
        field = F.conv3d(padded, kernel.view(shape).expand(3, 1, *shape[2:]), groups=3)
    return field[0].permute(1, 2, 3, 0).contiguous()


def _deformation_filter(function, identity):
    """Return a filter that applies ``function`` to ``u - identity`` only.

    ``identity`` is the displacement of the identity transform
    (``VoxelMap.displacement``), which is linear in the voxel position. The
    filters of this module pad the field at the volume faces, and the padding
    would bend a linear field. With ``identity`` ``None`` (the same grid) the
    filter is ``function`` itself.
    """
    if identity is None:
        return function
    return lambda disp: function(disp - identity) + identity


def _half_resolution_round_trip(disp: torch.Tensor) -> torch.Tensor:
    """Downsample an ``(X, Y, Z, 3)`` field by 2 and upsample it back.

    Both steps are trilinear (``align_corners=False``). The upsampling
    returns exactly the input size, also along an axis of odd length. The
    final volume projection takes its measurements on this smoothed copy.
    """
    field = disp.permute(3, 0, 1, 2).unsqueeze(0)
    field = F.interpolate(field, scale_factor=0.5, mode="trilinear",
                          align_corners=False)
    field = F.interpolate(field, size=tuple(disp.shape[:3]), mode="trilinear",
                          align_corners=False)
    return field[0].permute(1, 2, 3, 0)


def repair_folds_outside_body(
    disp: torch.Tensor,
    fixed_ct: torch.Tensor,
    body: torch.Tensor,
    spacing,
    cfg: PosthocConfig,
    voxel_map: VoxelMap,
) -> torch.Tensor:
    """Remove the folds of the field that lie outside the body.

    Each pass finds the folded voxels (non-diffeomorphic volume above 0)
    outside the body and its ``cfg.repair_margin_mm`` collar, grows them by
    ``cfg.repair_reach`` voxels, and blends the field towards its Gaussian
    blur (sigma ``cfg.repair_sigma``) there, with a smooth weight that is 0 on
    the body and the collar. After at most ``cfg.repair_passes`` passes, the
    field with the smallest %NDV numerator (the non-diffeomorphic volume
    summed over the voxels the scorer counts) is returned. The input field is
    one of the candidates, so the result is never worse than the input.

    Parameters
    ----------
    disp
        Displacement ``(X, Y, Z, 3)`` in voxels.
    fixed_ct
        Raw (unclipped) fixed CT, ``(X, Y, Z)``, on the device of ``disp``.
    body
        Fixed body mask, ``(X, Y, Z)`` boolean, on the device of ``disp``.
    spacing
        Fixed voxel size in mm along ``(X, Y, Z)``.
    cfg
        Repair settings.
    voxel_map
        The identity transform between the fixed and the moving grid.
    """
    outside = ~dilate_box(body, box_radii(spacing, cfg.repair_margin_mm))
    reach = (int(cfg.repair_reach),) * 3

    def feather(mask):
        """Return a weight in [0, 1] that is 1 on ``mask`` and fades out around it."""
        grown = dilate_box(mask, reach).float()
        # The grown region is copied into three channels so that it goes
        # through the same three-channel blur as the field.
        w = grown[..., None].expand(-1, -1, -1, 3).contiguous()
        w = gaussian_blur(w, cfg.repair_sigma)[..., 0].clamp(0.0, 1.0)
        return torch.where(outside, w, torch.zeros_like(w))

    blur = _deformation_filter(lambda field: gaussian_blur(field, cfg.repair_sigma),
                               voxel_map.displacement(disp.shape[:3], disp.device))
    scored = scored_voxels(fixed_ct)
    best, best_score = disp, None
    for step in range(cfg.repair_passes + 1):
        volume = ndv_map(disp, voxel_map)
        bad = (volume > 0) & outside
        count = int(bad.sum())
        score = float(volume[1:-1, 1:-1, 1:-1][scored].sum())
        log.debug("repair pass %d: %d folded voxels outside the body", step, count)
        if best_score is None or score < best_score:
            best, best_score = disp, score
        del volume
        if count == 0 or step == cfg.repair_passes:
            break
        weight = feather(bad).unsqueeze(-1)
        disp = disp + weight * (blur(disp) - disp)
    return best


def finalize(
    composed: torch.Tensor,
    fixed_ct: Volume,
    fixed_body: torch.Tensor,
    support: torch.Tensor,
    cfg: PosthocConfig,
    voxel_map: VoxelMap,
) -> np.ndarray:
    """Turn the composed sampling grid into the challenge displacement field.

    Parameters
    ----------
    composed
        ``(1, Z, Y, X, 3)`` normalized sampling grid of the whole transform.
    fixed_ct
        The raw (unclipped) fixed CT on the working grid
        (``regrep.io.working_grid``); its grid is the grid of the result.
    fixed_body
        Fixed body mask ``(X, Y, Z)``, boolean, on the device of
        ``composed``.
    support
        Lesion support of the moving image, ``(X', Y', Z')`` boolean on the
        moving grid, on the device of ``composed``: the same support that the
        stage-1 projection used.
    cfg
        The post-hoc settings.
    voxel_map
        The identity transform between the fixed and the moving grid.

    Returns
    -------
    numpy.ndarray
        ``(3, X, Y, Z)`` float32 displacement in voxels on the working fixed
        grid: fixed voxel ``x`` maps to voxel ``x + u(x)`` of the working
        moving grid.
    """
    with torch.no_grad():
        disp = grid_to_disp(composed, support.shape)
        identity = voxel_map.displacement(disp.shape[:3], disp.device)
        disp = project(disp, support, cfg, voxel_map,
                       measure=_deformation_filter(_half_resolution_round_trip,
                                                   identity))
        low_pass = _deformation_filter(
            lambda field: gaussian_blur(field, cfg.export_sigma), identity)
        disp = low_pass(disp)
        del identity
        ct = torch.as_tensor(np.asarray(fixed_ct.array, dtype=np.float32),
                             device=composed.device)
        disp = repair_folds_outside_body(disp, ct, fixed_body, fixed_ct.spacing,
                                         cfg, voxel_map)
        field = np.ascontiguousarray(
            disp.cpu().numpy().astype(np.float32).transpose(3, 0, 1, 2)
        )
    if not np.isfinite(field).all():
        raise RuntimeError("the displacement field holds values that are not finite")
    return field


# Largest number of voxels that resample_field and to_file_voxels compute at
# once.
RESAMPLE_SLAB_VOXELS = 2**24


def to_file_voxels(field: np.ndarray, file_index: np.ndarray) -> np.ndarray:
    """Express a field in the voxels of the moving file.

    ``field`` maps fixed voxel ``x`` to voxel ``x + u(x)`` of a moving array
    that was reordered or resampled after reading, and ``file_index``
    (``regrep.io.Volume.file_index``) maps the voxels of that array to the
    voxels of the file. The result maps ``x`` to the file voxel of the same
    point. The map is applied in float64, in slabs along the first axis, so
    the extra memory is that of a few slabs.
    """
    index = np.asarray(file_index, dtype=np.float64)
    shape = field.shape[1:]
    out = np.empty(field.shape, dtype=np.float32)
    step = max(1, RESAMPLE_SLAB_VOXELS // (shape[1] * shape[2]))
    for start in range(0, shape[0], step):
        stop = min(start + step, shape[0])
        voxels = np.indices((stop - start, *shape[1:]), dtype=np.float64)
        voxels[0] += start
        moving = field[:, start:stop].astype(np.float64) + voxels
        in_file = (np.tensordot(index[:3, :3], moving, axes=1)
                   + index[:3, 3][:, None, None, None])
        out[:, start:stop] = in_file - voxels
    return out


def resample_field(field: np.ndarray, field_affine: np.ndarray,
                   moving_affine: np.ndarray, shape, affine: np.ndarray,
                   device) -> np.ndarray:
    """Resample a field onto another grid of the fixed image.

    The field is split into the identity transform between its grid and the
    moving grid (:class:`VoxelMap`) and the deformation, which is the rest.
    Only the deformation is interpolated, trilinearly at the voxels of the
    new grid; beyond the outermost voxel centres of the field it keeps its
    edge values. The identity part is computed exactly at the voxels of the
    new grid, so an affine transform stays affine.

    Parameters
    ----------
    field
        ``(3, X', Y', Z')`` float32 displacement on the grid of
        ``field_affine``: its voxel ``y`` maps to moving voxel ``y + u(y)``.
    field_affine
        Voxel-to-world affine of the grid of ``field``.
    moving_affine
        Voxel-to-world affine of the moving grid.
    shape, affine
        The shape and the voxel-to-world affine of the new grid.
    device
        The device of the interpolation.

    Returns
    -------
    numpy.ndarray
        ``(3, X, Y, Z)`` float32: voxel ``x`` of the new grid maps to moving
        voxel ``x + v(x)``.
    """
    new_to_field = VoxelMap.between(affine, field_affine)
    new_to_moving = VoxelMap.between(affine, moving_affine)
    shape = tuple(int(n) for n in shape)
    out = np.empty((3, *shape), dtype=np.float32)
    with torch.no_grad():
        disp = torch.as_tensor(field, device=device).permute(1, 2, 3, 0)
        identity = VoxelMap.between(field_affine, moving_affine).displacement(
            disp.shape[:3], device)
        deformation = disp if identity is None else disp - identity
        # (1, 3, X', Y', Z'). grid_sample reads the components of a point in
        # the reverse order of these axes.
        deformation = deformation.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        del disp, identity
        scale = torch.tensor([2.0 / max(int(n) - 1, 1) for n in field.shape[1:]],
                             device=device, dtype=torch.float64)
        step = max(1, RESAMPLE_SLAB_VOXELS // (shape[1] * shape[2]))
        for start in range(0, shape[0], step):
            stop = min(start + step, shape[0])
            axes = [torch.arange(start, stop, device=device, dtype=torch.float64)]
            axes += [torch.arange(n, device=device, dtype=torch.float64)
                     for n in shape[1:]]
            points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
            normalized = new_to_field.apply(points) * scale - 1.0
            sampled = F.grid_sample(
                deformation, normalized[..., [2, 1, 0]].float().unsqueeze(0),
                mode="bilinear", padding_mode="border", align_corners=True)[0]
            del normalized
            moved = (new_to_moving.apply(points) - points).permute(3, 0, 1, 2)
            out[:, start:stop] = (moved + sampled).float().cpu().numpy()
    return out


def write_field(path: str, field: np.ndarray, fixed_ct: Volume) -> str:
    """Write a ``(3, X, Y, Z)`` displacement field in the challenge layout.

    The image has an identity affine and ``pixdim[1:4]`` set to the fixed
    voxel size, as in the challenge template. A ``.nii.gz`` file is written
    without compression (gzip level 0), which is fast.
    """
    field = np.ascontiguousarray(field, dtype=np.float32)
    template = nib.Nifti1Image(field, np.eye(4))
    template.header["pixdim"][1:4] = fixed_ct.spacing
    return save_nifti(path, field, np.eye(4), template.header)


def warp_to_fixed(volume: Volume, field: np.ndarray, fill: float,
                  device: torch.device) -> np.ndarray:
    """Warp a moving file onto the grid of the fixed CT file.

    Parameters
    ----------
    volume
        The moving CT or PET as read from its file.
    field
        The ``(3, X, Y, Z)`` output field: on the grid of the fixed CT file,
        in the voxels of the moving files.
    fill
        The value of the fixed voxels that map outside the moving volume.
    device
        The device of the warp.

    Returns
    -------
    np.ndarray
        The float32 ``(X, Y, Z)`` moved volume, interpolated trilinearly.
    """
    disp = torch.from_numpy(np.moveaxis(field, 0, -1).copy()).to(device)
    # The warp fills with zeros, so the volume is shifted to make fill zero.
    shifted = torch.from_numpy(volume.array).to(device) - fill
    moved = spatial_transform(shifted, disp, "bilinear") + fill
    return moved.cpu().numpy()


def write_moved(path: str, moved: np.ndarray, fixed_ct: Volume) -> str:
    """Write a moved volume as float32, with the affine of the fixed CT file."""
    return save_nifti(path, np.ascontiguousarray(moved, dtype=np.float32),
                      fixed_ct.affine)

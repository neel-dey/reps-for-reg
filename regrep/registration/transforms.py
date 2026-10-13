"""Conversions between sampling grids and voxel displacements, and warps.

Two representations of a transform are used in regrep:

* A sampling grid ``(1, Z, Y, X, 3)`` on the fixed grid. Each entry holds the
  moving location of a fixed voxel in normalized ``[-1, 1]`` coordinates
  (``align_corners=True``), with the last axis ordered ``(x, y, z)``. This is
  the form FireANTs and ``torch.nn.functional.grid_sample`` use.
* A voxel displacement ``(X, Y, Z, 3)`` on the fixed grid, in nibabel axis
  order. Component ``a`` is the displacement along axis ``a``, in voxels.
  Fixed voxel ``i`` maps to moving voxel ``i + u(i)``. This is the form the
  challenge scorer reads.

The fixed and the moving grid may have different shapes, voxel sizes and
affines. A sampling grid is normalized by the shape of the moving grid, so the
conversions take that shape as an argument. :class:`VoxelMap` describes the
identity transform in voxel units between the two grids.

The module also holds small volume helpers that several steps share: the
change from nibabel order to FireANTs order, and a box dilation of a mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def xyz_to_zyx(volume: torch.Tensor) -> torch.Tensor:
    """Return an ``(X, Y, Z)`` volume as a contiguous ``(1, 1, Z, Y, X)`` tensor."""
    return volume.permute(2, 1, 0).contiguous()[None, None]


def box_radii(spacing, radius_mm: float) -> tuple[int, int, int]:
    """Return the half-widths in voxels of a box with the given radius in mm.

    Each half-width is ``radius_mm`` divided by the voxel size along that
    axis, rounded, and at least 1. ``spacing`` gives the voxel sizes in the
    axis order of the mask that the box dilates.
    """
    return tuple(max(1, int(round(radius_mm / float(s)))) for s in spacing)


def dilate_box(mask: torch.Tensor, radii: tuple[int, int, int]) -> torch.Tensor:
    """Dilate a 3-D boolean mask by a box with the given half-widths in voxels.

    Voxels outside the volume count as background.
    """
    kernel = tuple(2 * int(r) + 1 for r in radii)
    padding = tuple(int(r) for r in radii)
    pooled = F.max_pool3d(mask.float()[None, None], kernel, stride=1, padding=padding)
    return pooled[0, 0] > 0.5


def _identity_voxels(shape, device, dtype) -> torch.Tensor:
    """Voxel coordinates ``(X, Y, Z, 3)`` of a grid of the given shape."""
    axes = [torch.arange(int(s), device=device, dtype=dtype) for s in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


@dataclass(frozen=True)
class VoxelMap:
    """The identity transform, written from fixed voxels to moving voxels.

    A point that does not move is at fixed voxel ``x`` and at moving voxel
    ``matrix @ x + offset``. When the two grids have the same affine, this is
    the identity and the voxel displacement of the identity transform is 0.

    Attributes
    ----------
    matrix
        The 3 x 3 linear part, as rows.
    offset
        The translation in moving voxels.
    """

    matrix: tuple[tuple[float, float, float], ...]
    offset: tuple[float, float, float]

    @classmethod
    def between(cls, fixed_affine: np.ndarray, moving_affine: np.ndarray) -> VoxelMap:
        """The map between two grids given by their voxel-to-world affines."""
        fixed_affine = np.asarray(fixed_affine, dtype=np.float64)
        moving_affine = np.asarray(moving_affine, dtype=np.float64)
        if np.array_equal(fixed_affine, moving_affine):
            voxel = np.eye(4)
        else:
            voxel = np.linalg.solve(moving_affine, fixed_affine)
        return cls(tuple(tuple(float(v) for v in row) for row in voxel[:3, :3]),
                   tuple(float(v) for v in voxel[:3, 3]))

    @property
    def is_identity(self) -> bool:
        """Whether fixed voxel ``x`` is moving voxel ``x``."""
        return np.array_equal(self.matrix, np.eye(3)) and not any(self.offset)

    @property
    def jacobian(self) -> float:
        """``det(matrix)``: the fixed voxel volume divided by the moving one.

        It is negative when the two grids have opposite handedness, and
        exactly 1 for the identity. Volume changes and folds measured in
        voxel units are divided by it.
        """
        return float(np.linalg.det(np.asarray(self.matrix)))

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """``matrix @ p + offset`` for points ``(..., 3)`` in fixed voxels."""
        return self.apply_matrix(points) + points.new_tensor(self.offset)

    def apply_matrix(self, vectors: torch.Tensor) -> torch.Tensor:
        """``matrix @ v`` for vectors ``(..., 3)``, in the dtype of ``vectors``."""
        rows = [sum(float(a) * vectors[..., j] for j, a in enumerate(row))
                for row in self.matrix]
        return torch.stack(rows, dim=-1)

    def displacement(self, shape, device) -> torch.Tensor | None:
        """Return the voxel displacement of the identity transform.

        The result is ``(X, Y, Z, 3)`` float32 on a fixed grid of ``shape``,
        or ``None`` when it is 0.
        """
        if self.is_identity:
            return None
        voxels = _identity_voxels(shape, device, torch.float32)
        return self.apply(voxels) - voxels


def grid_to_disp(grid: torch.Tensor, moving_shape) -> torch.Tensor:
    """Convert a sampling grid to a voxel displacement.

    Parameters
    ----------
    grid
        ``(1, Z, Y, X, 3)`` normalized moving coordinates of the fixed voxels.
    moving_shape
        Shape ``(X', Y', Z')`` of the moving grid.

    Returns
    -------
    torch.Tensor
        ``(X, Y, Z, 3)`` displacement in voxels, in the dtype of ``grid``.
    """
    device, dtype = grid.device, grid.dtype
    normalized = grid[0].permute(2, 1, 0, 3)
    sizes = torch.tensor([float(s) for s in moving_shape], device=device, dtype=dtype)
    target = (normalized + 1.0) * (sizes - 1.0) / 2.0
    return target - _identity_voxels(normalized.shape[:3], device, dtype)


def disp_to_grid(disp: torch.Tensor, moving_shape) -> torch.Tensor:
    """Convert a voxel displacement to a sampling grid.

    This is the inverse of :func:`grid_to_disp`, exact up to float rounding.

    Parameters
    ----------
    disp
        ``(X, Y, Z, 3)`` displacement in voxels.
    moving_shape
        Shape ``(X', Y', Z')`` of the moving grid.

    Returns
    -------
    torch.Tensor
        ``(1, Z, Y, X, 3)`` normalized sampling grid, contiguous.
    """
    device, dtype = disp.device, disp.dtype
    identity = _identity_voxels(disp.shape[:3], device, dtype)
    sizes = torch.tensor([float(s) for s in moving_shape], device=device, dtype=dtype)
    normalized = (identity + disp) * 2.0 / (sizes - 1.0) - 1.0
    return normalized.permute(2, 1, 0, 3).unsqueeze(0).contiguous()


def compose_grids(grid_old: torch.Tensor, grid_residual: torch.Tensor) -> torch.Tensor:
    """Compose two sampling grids: the result maps ``x`` to ``T_old(T_res(x))``.

    ``grid_old`` is sampled as a three-channel image at the coordinates of
    ``grid_residual``, trilinearly. Coordinates outside ``[-1, 1]`` read the
    border value of ``grid_old``.
    """
    old = grid_old.permute(0, 4, 1, 2, 3)
    composed = F.grid_sample(old, grid_residual, mode="bilinear",
                             padding_mode="border", align_corners=True)
    return composed.permute(0, 2, 3, 4, 1)


def warp_by(volume: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    """Resample a moving volume on the fixed grid, without gradients.

    Parameters
    ----------
    volume
        ``(1, C, Z, Y, X)`` moving volume, floating point.
    grid
        ``(1, Z', Y', X', 3)`` sampling grid.
    mode
        ``"nearest"`` for masks and labels, ``"bilinear"`` (trilinear) for
        intensities. Points outside the moving volume read zero.
    """
    with torch.no_grad():
        return F.grid_sample(volume, grid.to(volume.dtype), mode=mode,
                             padding_mode="zeros", align_corners=True)


def jacobian_determinant(disp: torch.Tensor) -> torch.Tensor:
    """``det(I + du/dx)`` on the interior of a normalized displacement.

    Derivatives are central differences converted to normalized units: along
    an axis of size ``S`` one voxel is ``2 / (S - 1)`` normalized units.

    Parameters
    ----------
    disp
        ``(N, D0, D1, D2, 3)`` displacement in normalized coordinates.
        Component 0 (``x``) runs along the last array axis, component 2
        (``z``) along the first.

    Returns
    -------
    torch.Tensor
        ``(N, D0-2, D1-2, D2-2)``, differentiable in ``disp``.
    """
    sizes = disp.shape[1:4]
    interior = [slice(1, -1)] * 3
    g = []
    # Row k of the Jacobian differentiates along x, y, z in turn, that is
    # along array axes 3, 2 and 1 of ``disp``.
    for axis in (3, 2, 1):
        factor = (sizes[axis - 1] - 1) / 2.0
        ahead, behind = list(interior), list(interior)
        ahead[axis - 1] = slice(2, None)
        behind[axis - 1] = slice(0, -2)
        difference = ((disp[(slice(None), *ahead)] - disp[(slice(None), *behind)])
                      * (factor / 2.0))
        g.append([difference[..., i] for i in range(3)])
    a, b, c = g[0][0] + 1.0, g[0][1], g[0][2]
    d, e, f = g[1][0], g[1][1] + 1.0, g[1][2]
    h, i, j = g[2][0], g[2][1], g[2][2] + 1.0
    return a * (e * j - f * i) - b * (d * j - f * h) + c * (d * i - e * h)


def spatial_transform(volume: torch.Tensor, disp: torch.Tensor,
                      mode: str) -> torch.Tensor:
    """Warp a volume by a voxel displacement, as the challenge scorer does.

    This is the scorer's VoxelMorph ``SpatialTransformer`` with
    ``align_corners=True``. The moving location ``i + u(i)`` is normalized by
    the shape of ``volume``, which may differ from the shape of ``disp``.

    Parameters
    ----------
    volume
        ``(X', Y', Z')`` moving volume.
    disp
        ``(X, Y, Z, 3)`` displacement in voxels, on the fixed grid.
    mode
        ``"bilinear"`` (trilinear) or ``"nearest"``. Nearest rounds half to
        even.

    Returns
    -------
    torch.Tensor
        The warped volume ``(X, Y, Z)`` in the dtype of ``disp``.
    """
    shape = tuple(volume.shape)
    locations = _identity_voxels(disp.shape[:3], disp.device, disp.dtype) + disp
    for axis in range(3):
        locations[..., axis] = 2.0 * (locations[..., axis] / (shape[axis] - 1) - 0.5)
    # grid_sample reads component 0 along the last array axis.
    locations = locations[..., [2, 1, 0]]
    warped = F.grid_sample(volume[None, None].to(disp.dtype), locations[None],
                           align_corners=True, mode=mode, padding_mode="zeros")
    return warped[0, 0]

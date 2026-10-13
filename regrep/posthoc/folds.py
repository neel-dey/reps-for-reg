"""Non-diffeomorphic volume (%NDV) of a displacement field.

The challenge measures folding with the method of Liu et al., "On finite
difference Jacobian computation in deformable image registration" (IJCV 2024).
The transform ``phi(x) = x + u(x)`` is differenced with ten one-sided schemes:
the eight combinations of forward and backward differences along the three
axes, and two diagonal ("star") schemes. Each scheme gives a Jacobian
determinant, which is six times the signed volume of a tetrahedron. The
non-diffeomorphic volume of a voxel is the sum over the ten schemes of
``-0.5 * min(det, 0) / 6``.

When the moving grid differs from the fixed grid, every determinant is
first divided by the Jacobian determinant of the identity transform between
the two grids (``regrep.registration.transforms.VoxelMap``), so that the
determinants of an unfolded transform are positive. That value is 1 when the
two grids are the same, which leaves the challenge's definition unchanged.

``ndv_map`` computes the per-voxel volume on the GPU with convolutions, for
the fold repair. ``jacobian_determinants_numpy`` gives the ten determinant
maps with plain array differences, and ``percent_ndv`` sums them as the
scorer does; the evaluator and the tests use these two. All are written for
this package from the published formulas.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from regrep.registration.transforms import VoxelMap

# The ten schemes as (partial set, channel offset) for the three rows of the
# Jacobian. For the line partials, offset 0 is the forward and 1 the backward
# difference; for the star partials, offset 0 is the first and 1 the second
# diagonal kernel.
_SCHEMES = (
    (("x", 0), ("y", 0), ("z", 0)),
    (("x", 0), ("y", 0), ("z", 1)),
    (("x", 0), ("y", 1), ("z", 0)),
    (("x", 0), ("y", 1), ("z", 1)),
    (("x", 1), ("y", 0), ("z", 0)),
    (("x", 1), ("y", 0), ("z", 1)),
    (("x", 1), ("y", 1), ("z", 0)),
    (("x", 1), ("y", 1), ("z", 1)),
    (("*xy", 0), ("*xz", 0), ("*yz", 0)),
    (("*xy", 1), ("*yz", 1), ("*xz", 1)),
)


def _det3(gx, gy, gz):
    """Return the 3x3 determinant whose rows are ``gx``, ``gy`` and ``gz``.

    Each row holds the derivatives of the three components of ``phi`` along
    one direction. The expansion and its order of operations are those of the
    challenge scorer.
    """
    return (
        gx[0] * (gy[1] * gz[2] - gy[2] * gz[1])
        - gy[0] * (gx[1] * gz[2] - gx[2] * gz[1])
        + gz[0] * (gx[1] * gy[2] - gx[2] * gy[1])
    )


def ndv_map(disp: torch.Tensor, voxel_map: VoxelMap) -> torch.Tensor:
    """Return the non-diffeomorphic volume of each voxel, on the field's device.

    Parameters
    ----------
    disp
        Displacement ``(X, Y, Z, 3)`` in voxels; component ``c`` is along
        array axis ``c``.
    voxel_map
        The identity transform between the fixed and the moving grid.

    Returns
    -------
    torch.Tensor
        ``(X, Y, Z)`` float32. The displacement is padded by one voxel with
        its edge values before differencing, so the map has the full size.
        When the grids differ, only the part of the displacement that the
        identity transform does not explain is padded this way. On the
        interior ``[1:-1, 1:-1, 1:-1]`` the map equals the challenge scorer's
        map (divided by the Jacobian determinant of the identity transform).
    """
    device = disp.device
    shape = disp.shape[:3]
    identity = voxel_map.displacement(shape, device)
    field = disp if identity is None else disp - identity
    u = F.pad(field.permute(3, 0, 1, 2)[None].float(), (1, 1, 1, 1, 1, 1),
              mode="replicate")
    del field
    axes = [torch.arange(-1, n + 1, device=device, dtype=torch.float32)
            for n in shape]
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"))
    if identity is not None:
        points = voxel_map.apply(points.permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
    trans = u + points[None]
    del u, points, identity
    jacobian = voxel_map.jacobian

    options = {"dtype": torch.float32, "device": device}
    forward = torch.tensor([0.0, -1.0, 1.0], **options)
    backward = torch.tensor([-1.0, 1.0, 0.0], **options)
    star1 = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]], **options)
    star2 = torch.tensor([[0, 0, 0], [0, -1, 0], [0, 0, 1]], **options)
    line = (forward, backward)
    star = (star1, star2)
    weights = {
        "x": torch.cat([k.view(1, 1, 3, 1, 1) for k in line] * 3, dim=0),
        "y": torch.cat([k.view(1, 1, 1, 3, 1) for k in line] * 3, dim=0),
        "z": torch.cat([k.view(1, 1, 1, 1, 3) for k in line] * 3, dim=0),
        "*xy": torch.cat([k.reshape(1, 1, 3, 3, 1) for k in star] * 3, dim=0),
        "*xz": torch.cat([k.reshape(1, 1, 3, 1, 3) for k in star] * 3, dim=0),
        "*yz": torch.cat([k.reshape(1, 1, 1, 3, 3) for k in star] * 3, dim=0),
    }
    # Each partial set holds, for the three components in turn, the two
    # derivatives of one kind, cropped back to the unpadded grid.
    partials = {
        "x": F.conv3d(trans, weights["x"], groups=3)[:, :, :, 1:-1, 1:-1],
        "y": F.conv3d(trans, weights["y"], groups=3)[:, :, 1:-1, :, 1:-1],
        "z": F.conv3d(trans, weights["z"], groups=3)[:, :, 1:-1, 1:-1, :],
        "*xy": F.conv3d(trans, weights["*xy"], groups=3)[:, :, :, :, 1:-1],
        "*xz": F.conv3d(trans, weights["*xz"], groups=3)[:, :, :, 1:-1, :],
        "*yz": F.conv3d(trans, weights["*yz"], groups=3)[:, :, 1:-1, :, :],
    }
    del trans

    out = None
    for rows in _SCHEMES:
        gx, gy, gz = (partials[name][0, offset::2] for name, offset in rows)
        det = _det3(gx, gy, gz) / jacobian
        term = -0.5 * torch.clamp(det, max=0.0) / 6.0
        out = term if out is None else out + term
        del det, term
    return out


def _interior_jacobian_rows(trans: np.ndarray):
    """Return the Jacobian rows of the ten schemes on the interior voxels.

    ``trans`` is ``phi`` as ``(3, X, Y, Z)`` float32. Every difference is
    taken between voxels of the interior and their direct neighbours, so no
    padding is needed. Each returned row is ``(3, X-2, Y-2, Z-2)``.
    """
    c = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
    lo, mid, hi = slice(None, -2), slice(1, -1), slice(2, None)
    centre = trans[c]

    def shifted(sx, sy, sz):
        return trans[:, sx, sy, sz]

    forward = (
        shifted(hi, mid, mid) - centre,
        shifted(mid, hi, mid) - centre,
        shifted(mid, mid, hi) - centre,
    )
    backward = (
        centre - shifted(lo, mid, mid),
        centre - shifted(mid, lo, mid),
        centre - shifted(mid, mid, lo),
    )
    # First star kernel: the neighbour at -1 on both axes of the plane minus
    # the voxel. Second star kernel: the neighbour at +1 on both axes minus
    # the voxel.
    star1 = {
        "xy": shifted(lo, lo, mid) - centre,
        "xz": shifted(lo, mid, lo) - centre,
        "yz": shifted(mid, lo, lo) - centre,
    }
    star2 = {
        "xy": shifted(hi, hi, mid) - centre,
        "xz": shifted(hi, mid, hi) - centre,
        "yz": shifted(mid, hi, hi) - centre,
    }
    one_sided = {"+": forward, "-": backward}
    for sx in "+-":
        for sy in "+-":
            for sz in "+-":
                yield (one_sided[sx][0], one_sided[sy][1], one_sided[sz][2])
    yield (star1["xy"], star1["xz"], star1["yz"])
    yield (star2["xy"], star2["yz"], star2["xz"])


def jacobian_determinants_numpy(field: np.ndarray) -> list[np.ndarray]:
    """Return the ten Jacobian determinant maps of the interior, with NumPy.

    Parameters
    ----------
    field
        Displacement ``(3, X, Y, Z)`` in voxels.

    Returns
    -------
    list of numpy.ndarray
        Ten float32 maps of shape ``(X-2, Y-2, Z-2)``: the eight one-sided
        schemes ``+++`` to ``---`` (x, y, z) and the two star schemes.
    """
    field = np.asarray(field, dtype=np.float32)
    identity = np.stack(
        np.meshgrid(*[np.arange(n) for n in field.shape[1:]], indexing="ij")
    ).astype(np.float32)
    trans = field + identity
    return [_det3(gx, gy, gz) for gx, gy, gz in _interior_jacobian_rows(trans)]


def scored_voxels(fixed_ct):
    """Return the voxels that %NDV counts, on the interior of the volume.

    These are the interior voxels (one voxel removed from every face) where
    the fixed CT is above its minimum over the interior. This is the same set
    as the scorer's "min-max normalized CT > 0". ``fixed_ct`` is the raw
    ``(X, Y, Z)`` CT as a NumPy array or a tensor; the result is a boolean
    mask of the same kind, of shape ``(X-2, Y-2, Z-2)``.
    """
    inner = fixed_ct[1:-1, 1:-1, 1:-1]
    return inner > inner.min()


def percent_ndv(field: np.ndarray, fixed_ct: np.ndarray, jacobian: float) -> float:
    """Return the challenge's %NDV of a displacement field.

    The voxels counted are those of :func:`scored_voxels`.

    Parameters
    ----------
    field
        Displacement ``(3, X, Y, Z)`` in voxels.
    fixed_ct
        The raw (unclipped) fixed CT, ``(X, Y, Z)``.
    jacobian
        The Jacobian determinant of the identity transform between the fixed
        and the moving grid (``VoxelMap.jacobian``), which divides every
        determinant.
    """
    mask = scored_voxels(np.asarray(fixed_ct))
    dets = jacobian_determinants_numpy(field)
    # The scorer sums the schemes in this order: four forward-x schemes, the
    # two star schemes, then the four backward-x schemes.
    # The sum stays in float32, as in the scorer.
    order = (0, 1, 2, 3, 8, 9, 4, 5, 6, 7)
    volume = np.float32(0)
    for index in order:
        volume += np.sum(-0.5 * np.minimum(dets[index] / jacobian, 0) * mask / 6)
    return float(volume / np.sum(mask) * 100)

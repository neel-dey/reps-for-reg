"""Banded Euclidean distance channels of label maps.

Each channel holds ``min(d, band) / band``, where ``d`` is the distance in mm
from a voxel to the nearest voxel of one structure. The channel is 0 inside
the structure, rises to 1 at ``band`` mm, and is 1 wherever the structure is
absent. The distances are exact (a separable squared-distance transform), but
they are only computed up to ``band``, and only inside a box around each
structure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

# Initial squared distance outside a structure. Adding an offset term of at
# most a few hundred mm^2 to it gives the same float32 value back, so a
# candidate that starts outside the structure can never win the minimum.
_FAR = 1e12


def label_boxes(volume: torch.Tensor, ids: Sequence[int]) -> list:
    """Return the inclusive index bounds of each label.

    Parameters
    ----------
    volume
        Integer label volume ``(D0, D1, D2)``.
    ids
        Label values to look up.

    Returns
    -------
    list
        One entry per id: ``((lo0, lo1, lo2), (hi0, hi1, hi2))``, or ``None``
        when the label has no voxel.
    """
    wanted = [int(i) for i in ids]
    index = torch.nonzero(volume > 0)
    if not len(index):
        return [None] * len(wanted)
    values = volume[index[:, 0], index[:, 1], index[:, 2]].long()
    size = max(int(values.max()), max(wanted)) + 1
    huge = torch.iinfo(torch.int64).max
    low = torch.full((size, 3), huge, dtype=torch.int64, device=volume.device)
    high = torch.full((size, 3), -1, dtype=torch.int64, device=volume.device)
    target = values[:, None].expand(-1, 3)
    low.scatter_reduce_(0, target, index, "amin")
    high.scatter_reduce_(0, target, index, "amax")
    low, high = low.cpu().tolist(), high.cpu().tolist()
    return [None if high[i][0] < 0 else (tuple(low[i]), tuple(high[i]))
            for i in wanted]


def banded_squared_distance(occupancy: torch.Tensor, spacing: Sequence[float],
                            band: float) -> torch.Tensor:
    """Squared distance in mm^2 to the nearest occupied voxel.

    The transform runs one axis at a time. Along axis ``a`` it compares each
    voxel with its neighbours up to ``ceil(band / spacing[a])`` voxels away,
    so the result is exact wherever the distance is at most ``band``. Farther
    voxels keep a large value.

    Parameters
    ----------
    occupancy
        Boolean tensor ``(K, D0, D1, D2)``.
    spacing
        Voxel size in mm along ``(D0, D1, D2)``.
    band
        Largest distance in mm that must be exact.

    Returns
    -------
    torch.Tensor
        float32 tensor with the shape of ``occupancy``.
    """
    field = torch.where(
        occupancy,
        torch.zeros((), device=occupancy.device, dtype=torch.float32),
        torch.full((), _FAR, device=occupancy.device, dtype=torch.float32),
    )
    for axis, step in enumerate(spacing, start=1):
        reach = int(np.ceil(float(band) / float(step)))
        size = field.shape[axis]
        best = field.clone()
        for offset in range(-reach, reach + 1):
            if offset == 0 or abs(offset) >= size:
                continue
            src = [slice(None)] * field.ndim
            dst = [slice(None)] * field.ndim
            if offset > 0:
                dst[axis] = slice(offset, None)
                src[axis] = slice(None, size - offset)
            else:
                dst[axis] = slice(None, offset)
                src[axis] = slice(-offset, None)
            view = best[tuple(dst)]
            torch.minimum(view, field[tuple(src)] + (offset * float(step)) ** 2,
                          out=view)
        field = best
    return field


def distance_channels(labels: torch.Tensor, ids: Sequence[int],
                      spacing: Sequence[float], band: float,
                      out: torch.Tensor | None = None) -> torch.Tensor:
    """Write one banded distance channel per label.

    The transform of each label runs on the label's bounding box grown by
    twice the reach of the band on each side. Outside that box the channel is
    exactly 1, and inside it the values equal those of a transform over the
    whole volume.

    Parameters
    ----------
    labels
        Integer label volume whose shape reduces to ``(D0, D1, D2)``.
    ids
        Label values, in channel order.
    spacing
        Voxel size in mm along ``(D0, D1, D2)``.
    band
        Truncation distance in mm.
    out
        Optional float32 buffer ``(1, len(ids), D0, D1, D2)`` to write into.

    Returns
    -------
    torch.Tensor
        ``(1, len(ids), D0, D1, D2)`` float32 holding ``min(d, band) / band``.
    """
    volume = labels
    while volume.ndim > 3:
        volume = volume[0]
    if out is None:
        out = torch.empty((1, len(ids)) + tuple(volume.shape), device=volume.device,
                          dtype=torch.float32)
    reach = [int(np.ceil(float(band) / float(step))) for step in spacing]
    out.fill_(1.0)
    boxes = label_boxes(volume, ids)
    for channel, (label, box) in enumerate(zip(ids, boxes, strict=True)):
        if box is None:
            continue
        low, high = box
        window = tuple(
            slice(max(0, low[a] - 2 * reach[a]),
                  min(int(volume.shape[a]), high[a] + 2 * reach[a] + 1))
            for a in range(3))
        occupancy = (volume[window] == int(label))[None]
        distance = torch.sqrt(banded_squared_distance(occupancy, spacing, band))
        clamped = torch.clamp(distance[0], max=float(band))
        out[(0, channel) + window] = clamped / float(band)
    return out

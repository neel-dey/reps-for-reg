"""A body mask computed from the CT alone.

The mask is a threshold, the largest connected component after a small
erosion (which cuts thin contacts such as the table), a hole fill in each
slice normal to the axis with the coarsest spacing, and a small closing. It
runs on the CPU with ``scipy.ndimage``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# 6-connected structuring element. A larger neighbourhood connects the body to
# the scanner table through single diagonal contacts.
_CONN6 = ndimage.generate_binary_structure(3, 1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 6-connected component (the lowest label on a tie)."""
    if not mask.any():
        return mask
    labels, count = ndimage.label(mask, structure=_CONN6)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def _open_then_component(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Erode, keep the largest component, dilate back and intersect.

    The erosion cuts thin contacts such as the one between the patient and the
    mattress. Because the result is intersected with the input, this step can
    only remove voxels.
    """
    eroded = ndimage.binary_erosion(
        mask, structure=_CONN6, iterations=iterations, border_value=1
    )
    if not eroded.any():
        return _largest_component(mask)
    kept = _largest_component(eroded)
    grown = ndimage.binary_dilation(
        kept, structure=_CONN6, iterations=iterations, border_value=0
    )
    return np.logical_and(grown, mask)


def _fill_axis(spacing: tuple[float, float, float]) -> int:
    """The axis with the coarsest spacing (the last such axis on a tie)."""
    values = np.asarray(spacing, dtype=float)
    return int(len(values) - 1 - np.argmax(values[::-1]))


def _fill_holes_slicewise(mask: np.ndarray, axis: int) -> np.ndarray:
    """Fill the holes of each 2-D slice along ``axis``, in place."""
    view = np.moveaxis(mask, axis, 0)
    for index in range(view.shape[0]):
        plane = view[index]
        if plane.any():
            view[index] = ndimage.binary_fill_holes(plane)
    return mask


def _close(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary closing that treats the array border as body during erosion.

    ``ndimage.binary_closing`` erodes from the border, which would remove the
    mask where the patient leaves the field of view.
    """
    grown = ndimage.binary_dilation(
        mask, structure=_CONN6, iterations=iterations, border_value=0
    )
    return ndimage.binary_erosion(
        grown, structure=_CONN6, iterations=iterations, border_value=1
    )


def body_mask(
    ct_hu: np.ndarray,
    spacing: tuple[float, float, float],
    threshold: float,
    open_voxels: int,
    close_voxels: int,
) -> np.ndarray:
    """Compute the body mask of a CT.

    Parameters
    ----------
    ct_hu
        The CT in Hounsfield units, a 3-D array.
    spacing
        Voxel spacing in the array's axis order. The hole fill runs on the
        slices normal to the axis with the coarsest spacing.
    threshold
        Voxels strictly above this value (HU) are body candidates.
    open_voxels
        Erosion depth before the largest component is taken.
    close_voxels
        Radius of the final closing.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the shape of ``ct_hu``.
    """
    if open_voxels < 1 or close_voxels < 1:
        # scipy treats iterations=0 as "repeat until nothing changes".
        raise ValueError("open_voxels and close_voxels must be at least 1")
    mask = ct_hu > threshold
    mask = _open_then_component(mask, open_voxels)
    mask = _fill_holes_slicewise(mask, _fill_axis(spacing))
    return _close(mask, close_voxels)

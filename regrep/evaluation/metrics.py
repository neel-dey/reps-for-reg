"""The PSMAReg challenge metrics of one registration pair.

The displacement field ``u`` is ``(3, X, Y, Z)`` in voxels on the fixed grid.
The warped moving image takes, at each fixed voxel ``x``, the moving value at
moving voxel ``x + u(x)``. Every volume (the CT labels, the PET lesion labels
and the PET image) is warped with nearest-neighbour sampling, as in the
challenge scorer. The moving grid may differ from the fixed grid; the volume
and uptake changes are then compared in physical units.

The metrics are:

* ``dsc_ct``, ``hd95_ct``: mean Dice and mean 95th-percentile Hausdorff
  distance (mm) over the CT organ labels 1 to 117 that are present in both the
  fixed and the moving label map. A label that the warp removes counts as Dice
  0, and it is left out of the HD95 mean because its distance is infinite.
* ``dsc_pet_lesion``, ``hd95_pet_lesion``: Dice and HD95 between the fixed and
  the warped moving PET lesion masks (all lesions together). Dice is NaN when
  both masks are empty; HD95 is NaN when either mask is empty.
* ``mtv_percent_err``, ``tlg_percent_err``: absolute percent change of the
  moving lesion volume and of the moving lesion uptake (sum of PET values in
  the lesion, times the voxel volume) caused by the warp.
* ``percent_ndv``: the non-diffeomorphic volume in percent
  (``regrep.posthoc.folds.percent_ndv``).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import torch

from regrep.evaluation import surface_distance
from regrep.posthoc.folds import percent_ndv
from regrep.registration.transforms import spatial_transform
from regrep.segmentation.labels import NUM_CLASSES

LABEL_IDS = range(1, NUM_CLASSES + 1)
METRICS = (
    "dsc_ct",
    "hd95_ct",
    "dsc_pet_lesion",
    "hd95_pet_lesion",
    "percent_ndv",
    "mtv_percent_err",
    "tlg_percent_err",
)


def read_field(path: str) -> np.ndarray:
    """Read a ``(3, X, Y, Z)`` displacement field file as a float32 array.

    This is the layout of the challenge and of ``regrep register``. A field
    of another shape, or with a value that is not finite, raises
    ``ValueError``.
    """
    field = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if field.ndim != 4 or field.shape[0] != 3:
        raise ValueError(f"{path}: expected a (3, X, Y, Z) field, got {field.shape}")
    if not np.isfinite(field).all():
        raise ValueError(f"{path}: the field holds a value that is not finite")
    return field


def _label_dice_and_hd95(fixed_labels, moving_labels, warped_labels, spacing, threads):
    """Return the mean Dice and the mean HD95 over the CT labels.

    The labels are scored in parallel threads and the results are collected
    in label order, so the means do not depend on the thread count.
    """

    def one(label_id):
        fixed_mask = fixed_labels == label_id
        moving_mask = moving_labels == label_id
        if not fixed_mask.any() or not moving_mask.any():
            return None
        warped_mask = warped_labels == label_id
        dice = surface_distance.compute_dice_coefficient(fixed_mask, warped_mask)
        return dice, _binary_hd95(fixed_mask, warped_mask, spacing)

    with ThreadPoolExecutor(threads) as pool:
        scored = list(pool.map(one, LABEL_IDS))

    dice_scores, hd95_scores = [], []
    for result in scored:
        if result is None:
            continue
        dice, hd95 = result
        dice_scores.append(float(dice))
        if np.isfinite(hd95):
            hd95_scores.append(float(hd95))
    return (
        float(np.mean(dice_scores)) if dice_scores else float("nan"),
        float(np.mean(hd95_scores)) if hd95_scores else float("nan"),
    )


def _binary_hd95(fixed_mask, warped_mask, spacing):
    """Return the HD95 of one pair of boolean masks, or NaN if either is empty."""
    if not fixed_mask.any() or not warped_mask.any():
        return float("nan")
    distances = surface_distance.compute_surface_distances(
        fixed_mask, warped_mask, spacing
    )
    return float(surface_distance.compute_robust_hausdorff(distances, 95.0))


def evaluate_pair(
    field: np.ndarray,
    fixed_ct: np.ndarray,
    fixed_ct_labels: np.ndarray,
    moving_ct_labels: np.ndarray,
    fixed_lesions: np.ndarray,
    moving_lesions: np.ndarray,
    moving_pet: np.ndarray,
    spacing: tuple[float, float, float],
    jacobian: float,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Compute the seven challenge metrics of one pair.

    All volumes are ``(X, Y, Z)`` arrays as read from the original
    (unclipped) files: the fixed ones on the fixed grid and the moving ones on
    the moving grid. The CT labels are scored in up to 12 threads.

    Parameters
    ----------
    field
        Displacement ``(3, X, Y, Z)`` in voxels.
    fixed_ct
        Raw fixed CT; it defines the voxels that %NDV counts.
    fixed_ct_labels, moving_ct_labels
        CT organ label maps (TotalSegmentator ``total`` ids).
    fixed_lesions, moving_lesions
        PET lesion label maps; any value above 0 is lesion.
    moving_pet
        Raw moving PET (SUV).
    spacing
        Fixed voxel size in mm along ``(X, Y, Z)``.
    jacobian
        The Jacobian determinant of the identity transform from fixed voxels
        to moving voxels (``regrep.registration.transforms.VoxelMap``): the
        fixed voxel volume divided by the moving one, negative when the grids
        have opposite handedness, and 1 when they are the same grid.
    device
        Device for the warps and the lesion sums. The CPU and the GPU can
        resolve nearest-neighbour ties differently, so compare scores that
        were computed on the same kind of device.

    Returns
    -------
    dict
        One float for each name in ``METRICS``; NaN where a metric is not
        defined for the pair.
    """
    threads = min(12, os.cpu_count() or 1)
    spacing = tuple(float(s) for s in spacing)
    field = np.asarray(field, dtype=np.float32)
    fixed_shape = np.shape(fixed_ct)
    moving_shape = np.shape(moving_ct_labels)
    fixed_shapes = {field.shape[1:], np.shape(fixed_ct_labels), np.shape(fixed_lesions)}
    if fixed_shapes != {fixed_shape}:
        raise ValueError(f"the field {field.shape} and the fixed volumes must share "
                         f"the fixed grid {fixed_shape}")
    if np.shape(moving_lesions) != moving_shape or np.shape(moving_pet) != moving_shape:
        raise ValueError("the moving labels, lesions and PET must share one grid")
    scale = abs(float(jacobian))
    disp = torch.as_tensor(np.moveaxis(field, 0, -1), device=device)

    def on_device(array):
        return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)

    with torch.inference_mode():
        moving_lesions_t = on_device(moving_lesions)
        moving_pet_t = on_device(moving_pet)
        warped_ct = spatial_transform(on_device(moving_ct_labels), disp, "nearest")
        warped_lesions = spatial_transform(moving_lesions_t, disp, "nearest")
        warped_pet = spatial_transform(moving_pet_t, disp, "nearest")
        mtv_before = float((moving_lesions_t > 0).sum())
        mtv_after = float((warped_lesions > 0).sum()) * scale
        tlg_before = float(((moving_lesions_t > 0).float() * moving_pet_t).sum())
        tlg_after = float(((warped_lesions > 0).float() * warped_pet).sum()) * scale
        warped_ct = torch.round(warped_ct).cpu().numpy()
        warped_lesion_mask = (warped_lesions > 0).cpu().numpy()

    dsc_ct, hd95_ct = _label_dice_and_hd95(
        np.asarray(fixed_ct_labels),
        np.asarray(moving_ct_labels),
        warped_ct,
        spacing,
        threads,
    )
    fixed_lesion_mask = np.asarray(fixed_lesions) > 0
    return {
        "dsc_ct": dsc_ct,
        "hd95_ct": hd95_ct,
        "dsc_pet_lesion": float(surface_distance.compute_dice_coefficient(
            fixed_lesion_mask, warped_lesion_mask)),
        "hd95_pet_lesion": _binary_hd95(fixed_lesion_mask, warped_lesion_mask, spacing),
        "percent_ndv": percent_ndv(field, fixed_ct, jacobian),
        "mtv_percent_err": 100.0 * abs(mtv_after - mtv_before) / max(mtv_before, 1e-6),
        "tlg_percent_err": 100.0 * abs(tlg_after - tlg_before) / max(tlg_before, 1e-6),
    }

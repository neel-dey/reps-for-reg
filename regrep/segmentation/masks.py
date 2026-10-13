"""Lesion supports.

The volume penalty and the volume projection protect the lesions of the
moving image. The region they act on, the lesion support, is the union of two
candidates:

* voxels with SUV above a threshold that are not inside or near an organ with
  physiological uptake, and
* the LION mask of the moving PET.

Stage 1 builds the organ part from the 6 mm TotalSegmentator map and stage 2
from the 3 mm map.
"""

from __future__ import annotations

import numpy as np
import torch

from regrep.config import LesionConfig
from regrep.io import Volume
from regrep.registration.transforms import box_radii, dilate_box
from regrep.segmentation.labels import class_ids_by_name


def physiological_ids(names: tuple[str, ...]) -> list[int]:
    """Return the TotalSegmentator class ids of the given class names."""
    by_name = class_ids_by_name()
    return sorted(by_name[name] for name in names)


def lesion_support(organs: torch.Tensor, pet: Volume, lion: torch.Tensor,
                   cfg: LesionConfig) -> torch.Tensor:
    """Return the lesion support of one PET volume.

    The support is ``(PET > suv_threshold) AND NOT halo`` OR ``LION``, where
    the halo is the set of physiological organ voxels dilated by a box of
    half-width ``organ_halo_mm``. LION voxels inside the halo stay in the
    support.

    Parameters
    ----------
    organs
        ``(X, Y, Z)`` TotalSegmentator ``total`` label map on the PET grid.
    pet
        The PET volume in SUV, not clipped.
    lion
        ``(X, Y, Z)`` LION mask on the PET grid, on the device of
        ``organs``; nonzero voxels are lesions.
    cfg
        The lesion settings.

    Returns
    -------
    torch.Tensor
        ``(X, Y, Z)`` boolean mask on the device of ``organs``.
    """
    ids = torch.tensor(physiological_ids(cfg.physiological_organs),
                       dtype=organs.dtype, device=organs.device)
    radii = box_radii(pet.spacing, cfg.organ_halo_mm)
    halo = dilate_box(torch.isin(organs, ids), radii)
    suv = torch.from_numpy(np.asarray(pet.array, np.float32)).to(organs.device)
    return ((suv > cfg.suv_threshold) & ~halo) | (lion > 0)

"""Image features for the registration: body mask, anatomix and MIND-SSC.

:func:`extract_side` turns the CT and PET of one timepoint into a
:class:`SideFeatures`, which holds everything the two registration stages read
from that timepoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK
import torch
import torch.nn.functional as F
from fireants.io.image import Image

from regrep.config import FeatureConfig
from regrep.features.anatomix import (
    anatomix_features,
    grid_shape,
    load_backbone,
    resample_channels,
)
from regrep.features.body_mask import body_mask
from regrep.features.mind import mind_ssc
from regrep.io import Volume

__all__ = ["SideFeatures", "extract_side", "load_backbone"]


@dataclass
class SideFeatures:
    """The features of one timepoint, on its native grid.

    Tensors are in FireANTs order ``(1, C, Z, Y, X)``, fp32, on the
    registration device.

    Attributes
    ----------
    torch2phy, phy2torch
        ``(1, 4, 4)`` maps from normalized grid coordinates to physical ones
        and back, from the FireANTs ``Image`` of the CT.
    spacing
        Voxel spacing (mm) in array order ``(z, y, x)``.
    ct_norm, pet_norm
        CT and PET, each clipped and min-max normalized to [0, 1].
    amix
        32 anatomix channels with unit L2 norm per voxel, not masked.
    mind
        12 MIND-SSC channels, values in (0, 1], not masked.
    body
        Body mask, 0 or 1.

    ``ct_norm``, ``pet_norm`` and ``mind`` are ``None`` after
    :meth:`release_stage1_inputs`.
    """

    torch2phy: torch.Tensor
    phy2torch: torch.Tensor
    spacing: tuple[float, float, float]
    ct_norm: torch.Tensor | None
    pet_norm: torch.Tensor | None
    amix: torch.Tensor
    mind: torch.Tensor | None
    body: torch.Tensor

    def release_stage1_inputs(self) -> None:
        """Free the channels that only stage 1 reads (CT, PET and MIND-SSC)."""
        self.ct_norm = self.pet_norm = self.mind = None


def _clip_zyx(volume: Volume, clip: tuple[float, float]) -> np.ndarray:
    """Clip a volume in float64, cast it to float32 and reorder it to (Z, Y, X)."""
    array = np.clip(np.asarray(volume.array, dtype=np.float64), clip[0], clip[1])
    return np.ascontiguousarray(array.astype(np.float32).transpose(2, 1, 0))


# NIfTI world coordinates are RAS (nibabel); those of ITK are LPS.
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])


def _ct_image(volume: Volume, array_zyx: np.ndarray, device) -> Image:
    """Build the FireANTs ``Image`` of the clipped CT.

    The ITK geometry (origin, voxel size and direction) comes from
    ``volume.affine``, the nibabel affine that all other steps of regrep use.
    When SimpleITK reads the same affine from the file header, as it does on
    the PSMAReg files, the geometry is the one that ``Image.load_file`` gives.
    """
    affine = np.asarray(volume.affine, dtype=np.float64)
    linear = _RAS_TO_LPS @ affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    origin = _RAS_TO_LPS @ affine[:3, 3]
    itk_image = SimpleITK.GetImageFromArray(array_zyx)
    itk_image.SetOrigin(tuple(float(v) for v in origin))
    itk_image.SetSpacing(tuple(float(v) for v in spacing))
    itk_image.SetDirection(tuple(float(v) for v in (linear / spacing).ravel()))
    return Image(itk_image, device=device)


def _minmax(array: torch.Tensor, name: str) -> torch.Tensor:
    """Scale a volume to [0, 1] by its own minimum and maximum."""
    if not bool(torch.isfinite(array).all()):
        raise ValueError(f"the {name} holds NaN or infinite values")
    low, high = array.min(), array.max()
    if not bool(high > low):
        raise ValueError(f"the {name} is constant after clipping")
    return (array - low) / (high - low)


def extract_side(
    ct: Volume,
    pet: Volume,
    model: torch.nn.Module,
    cfg: FeatureConfig,
    device: torch.device,
    sum_mm: float | None,
) -> SideFeatures:
    """Compute the features of one timepoint.

    The caller sets the seed and the cuDNN flags before the first call; this
    function does not change them.

    Parameters
    ----------
    ct, pet
        The raw CT (HU) and PET (SUV) of the timepoint, on the same grid. The
        geometry of the features is that of ``ct.affine``.
    model
        The anatomix network from :func:`load_backbone`, on ``device``.
    cfg
        Feature settings.
    device
        The registration device.
    sum_mm
        The spacing (mm) of the isotropic grid on which the levels of the
        anatomix pyramid are summed, which is also the ``None`` level of
        ``cfg.pyramid_mm``. The registration passes the smallest voxel size
        of the fixed CT for both timepoints, so that both are described at
        the same scales. ``None`` uses the smallest voxel size of this CT.

    Returns
    -------
    SideFeatures
        The features of this timepoint, on ``device``.
    """
    ct_zyx = _clip_zyx(ct, cfg.ct_clip)
    image = _ct_image(ct, ct_zyx, device)
    spacing = tuple(float(s) for s in reversed(image.itk_image.GetSpacing()))

    mask = body_mask(
        ct_zyx,
        spacing,
        threshold=cfg.body_threshold_hu,
        open_voxels=cfg.body_open_voxels,
        close_voxels=cfg.body_close_voxels,
    )
    body = torch.from_numpy(mask).to(device=device, dtype=torch.float32)[None, None]
    del ct_zyx, mask

    ct_norm = _minmax(image.array, "CT")
    # Only the geometry of the image is kept, so that its copy of the CT is
    # freed.
    torch2phy, phy2torch = image.torch2phy, image.phy2torch
    del image
    pet_zyx = torch.from_numpy(_clip_zyx(pet, cfg.pet_clip))
    pet_norm = _minmax(pet_zyx.to(device=device)[None, None], "PET")
    del pet_zyx

    # The CT on the MIND grid. When that spacing is also a pyramid level, the
    # anatomix pyramid uses this volume instead of resampling the CT again.
    native = tuple(int(n) for n in ct_norm.shape[-3:])
    mind_shape = grid_shape(native, spacing, cfg.mind_spacing_mm)
    ct_mind = F.interpolate(
        ct_norm, size=mind_shape, mode="trilinear", align_corners=True
    )
    amix = anatomix_features(
        ct_norm,
        spacing,
        model,
        body,
        cfg,
        ct_mind,
        sum_mm=min(spacing) if sum_mm is None else sum_mm,
    )

    mind_grid = mind_ssc(
        ct_mind, cfg.mind_radius, cfg.mind_dilation, cfg.mind_slab_voxels
    )
    del ct_mind
    torch.cuda.empty_cache()
    mind = resample_channels(mind_grid, native)
    del mind_grid
    torch.cuda.empty_cache()

    return SideFeatures(
        torch2phy=torch2phy,
        phy2torch=phy2torch,
        spacing=spacing,
        ct_norm=ct_norm,
        pet_norm=pet_norm,
        amix=amix,
        mind=mind,
        body=body,
    )

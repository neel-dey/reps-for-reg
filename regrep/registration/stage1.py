"""Stage 1: deformable registration of a 46-channel image stack.

Each timepoint is a stack of the normalized CT, the normalized PET, 32 anatomix
channels and 12 MIND-SSC channels, all set to zero outside the body mask, with
the body mask as a last channel. The stage starts from a translation that
matches the centres of mass of the masked CTs and then optimizes a compositive
warp on a four-level pyramid. The image term is the masked local normalized
cross-correlation summed over the 46 channels. A volume penalty keeps
``det J`` near 1 on the moving lesion support.
"""

from __future__ import annotations

import torch
from fireants.registration.moments import MomentsRegistration

from regrep.config import Stage1Config
from regrep.registration.chunked import (
    ChunkedGreedyRegistration,
    MaskedLNCC,
    StackImages,
    split_channels,
)
from regrep.registration.losses import VolumePenalty
from regrep.registration.transforms import warp_by, xyz_to_zyx


def build_stack(side) -> torch.Tensor:
    """Return the ``(1, 47, Z, Y, X)`` stack of one timepoint.

    Channels 0 to 45 are the CT, the PET, the anatomix and the MIND-SSC
    channels, each multiplied by the body mask; channel 46 is the body mask.

    Parameters
    ----------
    side
        A ``SideFeatures`` with ``ct_norm``, ``pet_norm`` (``(1, 1, Z, Y, X)``),
        ``amix`` (``(1, 32, ...)``), ``mind`` (``(1, 12, ...)``) and ``body``
        (``(1, 1, ...)``, values 0 and 1).
    """
    parts = [side.ct_norm, side.pet_norm, side.amix, side.mind]
    channels = sum(p.shape[1] for p in parts)
    dtype = side.ct_norm.dtype
    inside = (side.body > 0).to(dtype)
    stack = torch.empty((1, channels + 1, *side.ct_norm.shape[2:]), dtype=dtype,
                        device=side.ct_norm.device)
    start = 0
    for part in parts:
        stop = start + part.shape[1]
        torch.mul(part, inside, out=stack[:, start:stop])
        start = stop
    stack[:, -1:] = side.body.to(dtype)
    return stack


def mass_image(side) -> torch.Tensor:
    """The normalized CT inside the body mask, for the centre of mass."""
    return side.ct_norm * (side.body > 0).to(side.ct_norm.dtype)


def side_images(side, array: torch.Tensor) -> StackImages:
    """Wrap an array on the grid of ``side`` with that grid's geometry."""
    return StackImages(array, side.torch2phy, side.phy2torch)


def centre_of_mass_init(fixed_mass: StackImages,
                        moving_mass: StackImages) -> MomentsRegistration:
    """Fit the translation that matches the centres of mass of two images."""
    moments = MomentsRegistration(scale=1, fixed_images=fixed_mass,
                                  moving_images=moving_mass, moments=1,
                                  loss_params={})
    moments.optimize()
    return moments


def initial_linear_map(fixed: StackImages, moving: StackImages,
                       init_affine: torch.Tensor) -> torch.Tensor | None:
    """Return the linear part ``A`` of the initial map, or ``None`` if it is I.

    FireANTs samples the moving image at ``A x + b + u(x)``, where ``x`` and
    the result are normalized coordinates of the fixed and the moving grid
    and ``A x + b`` is the physical initial affine between them. ``A`` is the
    identity when the initial affine has no linear part and both grids have
    the same shape, voxel size and direction; the function then returns
    ``None``.

    Parameters
    ----------
    fixed, moving
        The images of the two grids.
    init_affine
        The physical initial affine ``(1, 3, 4)``.
    """
    linear = init_affine[0, :3, :3]
    fixed_linear = fixed.get_torch2phy()[0, :3, :3]
    moving_linear = moving.get_torch2phy()[0, :3, :3]
    if (torch.equal(linear, torch.eye(3, dtype=linear.dtype, device=linear.device))
            and torch.equal(fixed_linear, moving_linear)):
        return None
    to_moving = moving.get_phy2torch()[0, :3, :3].double()
    return to_moving @ linear.double() @ fixed_linear.double()


def register(fixed: StackImages, moving: StackImages, fixed_mass: StackImages,
             moving_mass: StackImages, support: torch.Tensor, cfg: Stage1Config,
             devices, chunk_channels: int | None) -> torch.Tensor:
    """Run stage 1 on two assembled stacks.

    Parameters
    ----------
    fixed, moving
        The ``(1, 47, Z, Y, X)`` stacks of :func:`build_stack` with their
        geometry. The fixed stack's device is the main device.
    fixed_mass, moving_mass
        The masked normalized CT of each timepoint, ``(1, 1, Z, Y, X)``.
    support
        The lesion support on the moving grid, ``(1, 1, Z, Y, X)``, values 0
        and 1. Stage 1 penalizes the volume change on its image under the
        centre-of-mass translation, measured in physical units when the two
        grids differ.
    cfg
        Stage-1 settings.
    devices
        Devices for the channel chunks; the first must hold the fixed stack.
    chunk_channels
        Largest number of image channels in one chunk. ``None`` gives one
        chunk per device.

    Returns
    -------
    torch.Tensor
        ``grid1``, the ``(1, Z, Y, X, 3)`` sampling grid from the fixed grid
        to the moving grid, on the main device.
    """
    moments = centre_of_mass_init(fixed_mass, moving_mass)
    init_affine = moments.get_affine_init()
    init_grid = moments.get_warped_coordinates(fixed_mass, moving_mass).detach()
    penalty = VolumePenalty(warp_by(support.float(), init_grid, "nearest"),
                            cfg.volume_penalty,
                            initial_linear_map(fixed, moving, init_affine))
    del init_grid

    chunks = split_channels(fixed.shape[1] - 1, devices, chunk_channels)
    reg = ChunkedGreedyRegistration(
        cfg.scales, cfg.iterations, fixed, moving, MaskedLNCC(cfg.cc_kernel_sizes),
        chunks, optimizer_lr=cfg.learning_rate, smooth_grad_sigma=cfg.grad_sigma,
        smooth_warp_sigma=cfg.warp_sigma, displacement_reg=penalty,
        tolerance=cfg.tolerance, max_tolerance_iters=cfg.tolerance_window,
        init_affine=init_affine)
    reg.optimize()
    return reg.get_warped_coordinates(fixed, moving).detach()


def run(fixed, moving, support1: torch.Tensor, stage_plan,
        cfg: Stage1Config) -> torch.Tensor:
    """Build the stacks of two timepoints and run stage 1.

    Parameters
    ----------
    fixed, moving
        ``SideFeatures`` of the fixed and the moving timepoint, on the main
        registration device.
    support1
        The stage-1 lesion support on the moving grid, ``(X, Y, Z)``
        boolean, on the main registration device.
    stage_plan
        The ``StagePlan`` of stage 1: ``devices`` (the first holds the
        features) and ``chunk_channels``.
    cfg
        Stage-1 settings.

    Returns
    -------
    torch.Tensor
        ``grid1``, ``(1, Z, Y, X, 3)`` normalized moving coordinates of the
        fixed voxels.
    """
    return register(side_images(fixed, build_stack(fixed)),
                    side_images(moving, build_stack(moving)),
                    side_images(fixed, mass_image(fixed)),
                    side_images(moving, mass_image(moving)),
                    xyz_to_zyx(support1).float(), cfg, stage_plan.devices,
                    stage_plan.chunk_channels)

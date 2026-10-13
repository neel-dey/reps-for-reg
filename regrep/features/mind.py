# The MIND-SSC computation is adapted from the MINDSSC function of ConvexAdam,
# https://github.com/multimodallearning/convexAdam, by Hanna Siebert, Lasse
# Hansen and Mattias P. Heinrich. The ConvexAdam repository states no
# copyright line. regrep took the function from the copy in anatomix
# (anatomix/registration/registration_infrastructure/mindssc.py,
# https://github.com/neel-dey/anatomix, MIT License, Copyright 2024 Neel Dey),
# which says that it is adapted from ConvexAdam. ConvexAdam uses the Apache
# License, Version 2.0. For the parts adapted from it:
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes made for regrep: the descriptor is the same (the six-neighbourhood,
# the twelve pairs at squared distance 2, the replicate padding, the box
# patch, the subtraction of the minimum, the variance clamp and the channel
# order), but it is computed in slabs along the first axis, each with a halo,
# and the mean variance of the clamp is accumulated over the slabs.
"""MIND-SSC descriptors on an isotropic grid, resampled to the native grid.

MIND-SSC (Heinrich et al., MICCAI 2013) compares 12 pairs of voxels in a
six-neighbourhood around each voxel. This implementation processes the volume
in slabs along the first spatial axis, so that a whole-body 1 mm volume fits on
one GPU. Each slab carries a halo of ``radius + dilation`` rows, so its interior
is computed from the same input as the whole volume would give it. The
variance clamp needs the mean variance of the whole volume; it is accumulated
over the slabs and applied in a second pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Channel order of the MIND-SSC C++ implementation by Heinrich et al.
# (deedsBCV).
_CHANNEL_ORDER = (6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3)

# The six neighbours of the centre of a 3x3x3 cube, as (z, y, x) offsets.
_SIX_NEIGHBOURS = (
    (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 2), (2, 1, 1), (1, 2, 1),
)


def _shift_kernels(device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Two one-hot 3x3x3 kernels that select the two voxels of each pair.

    The 12 pairs are the neighbour pairs ``(i, j)`` with ``i > j`` whose
    offsets are at squared distance 2, in row-major order of ``(i, j)``.
    """
    pairs = []
    for i, first in enumerate(_SIX_NEIGHBOURS):
        for j, second in enumerate(_SIX_NEIGHBOURS):
            distance = sum((a - b) ** 2 for a, b in zip(first, second, strict=True))
            if i > j and distance == 2:
                pairs.append((first, second))
    kernels = []
    for member in (0, 1):
        kernel = torch.zeros(12, 1, 3, 3, 3)
        for channel, pair in enumerate(pairs):
            kernel[(channel, 0) + pair[member]] = 1.0
        kernels.append(kernel.to(device=device, dtype=dtype))
    return kernels[0], kernels[1]


def _distances(img, radius, dilation, kernels):
    """Patch distances of one slab, minus their per-voxel minimum.

    Returns the 12 distances in pair order and their mean over the channels.
    """
    first, second = kernels
    padded = F.pad(img, (dilation,) * 6, mode="replicate")
    conv1 = F.conv3d(padded, first, dilation=dilation)
    conv2 = F.conv3d(padded, second, dilation=dilation)
    del padded
    difference = (conv1 - conv2) ** 2
    del conv1, conv2
    ssd = F.avg_pool3d(
        F.pad(difference, (radius,) * 6, mode="replicate"),
        radius * 2 + 1,
        stride=1,
    )
    del difference
    distances = ssd - torch.min(ssd, 1, keepdim=True)[0]
    del ssd
    return distances, torch.mean(distances, 1, keepdim=True)


def mind_ssc(
    img: torch.Tensor, radius: int, dilation: int, slab_voxels: int
) -> torch.Tensor:
    """Compute MIND-SSC of a volume, in slabs.

    Parameters
    ----------
    img
        Volume of shape ``(1, 1, D, H, W)``.
    radius
        Half-width of the patch over which the squared differences are
        averaged.
    dilation
        Spacing (voxels) of the six neighbours.
    slab_voxels
        Upper bound on the voxels of one slab, per channel.

    Returns
    -------
    torch.Tensor
        Descriptor of shape ``(1, 12, D, H, W)`` with values in ``(0, 1]``.
    """
    depth = int(img.shape[2])
    rows = max(1, slab_voxels // (int(img.shape[3]) * int(img.shape[4])))
    kernels = _shift_kernels(img.device, img.dtype)
    order = torch.tensor(_CHANNEL_ORDER, device=img.device).long()
    halo = radius + dilation
    mind = torch.empty((1, 12) + tuple(img.shape[2:]), device=img.device,
                       dtype=img.dtype)
    variance = torch.empty((1, 1) + tuple(img.shape[2:]), device=img.device,
                           dtype=img.dtype)

    # The volume mean of the variance is accumulated in a Python float, one
    # slab mean at a time.
    total = 0.0
    for start in range(0, depth, rows):
        stop = min(depth, start + rows)
        low, high = max(0, start - halo), min(depth, stop + halo)
        slab, slab_variance = _distances(img[:, :, low:high], radius, dilation,
                                         kernels)
        interior = slice(start - low, stop - low)
        mind[:, :, start:stop] = slab[:, order, interior]
        variance[:, :, start:stop] = slab_variance[:, :, interior]
        total += float(slab_variance[:, :, interior].mean()) * (stop - start)
        del slab, slab_variance
        torch.cuda.empty_cache()

    mean_variance = total / depth
    lower, upper = mean_variance * 0.001, mean_variance * 1000
    for start in range(0, depth, rows):
        stop = min(depth, start + rows)
        block = mind[:, :, start:stop]
        block.div_(variance[:, :, start:stop].clamp_(lower, upper))
        block.neg_().exp_()
    del variance
    torch.cuda.empty_cache()
    return mind

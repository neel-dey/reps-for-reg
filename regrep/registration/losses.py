"""The terms of the stage-2 objective.

The image part of the objective reads two channel stacks on the fixed grid:
one distance channel per structure (see ``distance.py``), the 32 anatomix
channels, and a body mask as the last channel. It has three terms:

* a generalized Dice loss on soft structure memberships,
* a one-sided chamfer distance from the moved structures to the fixed ones,
* a masked local normalized cross-correlation on the anatomix channels.

Each term is written as a set of channel sums, so the stack can be processed
in channel chunks: the sums of all chunks are added, and the term is a
function of the totals.

Two more terms read only the displacement field: the region distance, which
measures how far the fixed structures land from the native moving
structures, and a volume penalty on the lesion support. Stage 1 uses the same
volume penalty on every level of its pyramid.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
from torch import nn

from regrep.registration.chunked import masked_mean
from regrep.registration.transforms import jacobian_determinant

DICE_EPS = 1e-5
CHAMFER_EPS = 1e-6


def shell_membership(distance: torch.Tensor, shell: float) -> torch.Tensor:
    """Soft membership ``max(0, 1 - d / shell)`` from a distance channel.

    ``distance`` and ``shell`` are in band units (mm divided by the band).
    The membership is 1 inside the structure and 0 at ``shell`` and beyond.
    """
    return (1.0 - distance / shell).clamp_min(0.0)


def label_sums(moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor,
               dice_weights: torch.Tensor, chamfer_weights: torch.Tensor,
               shell: float) -> torch.Tensor:
    """Return the weighted Dice and chamfer sums of some structure channels.

    Parameters
    ----------
    moved, fixed
        Distance channels ``(1, k, Z, Y, X)`` of the moved and the fixed side.
    mask
        Combined body mask ``(1, 1, Z, Y, X)``.
    dice_weights, chamfer_weights
        Weights ``(k,)`` of these channels.
    shell
        Width of the soft membership, in band units.

    Returns
    -------
    torch.Tensor
        ``(3,)``: ``sum_c w_c I_c``, ``sum_c w_c V_c`` and ``sum_c v_c t_c``,
        where ``I_c`` is the soft intersection, ``V_c`` the sum of the two
        soft volumes, and ``t_c`` the mean fixed distance over the moved
        membership of structure ``c``.
    """
    spatial = (2, 3, 4)
    moved_points = shell_membership(moved, shell) * mask
    fixed_points = shell_membership(fixed, shell) * mask
    intersection = (moved_points * fixed_points).sum(dim=spatial)
    volumes = moved_points.sum(dim=spatial) + fixed_points.sum(dim=spatial)
    dice_sums = [(dice_weights * intersection).sum(dim=1),
                 (dice_weights * volumes).sum(dim=1)]
    # Sum the moved memberships again instead of reusing ``volumes``: the
    # separate sum gives the gradient its own accumulation path, and the
    # results depend on that order at float rounding.
    points = moved_points.sum(dim=spatial)
    distance = (moved_points * fixed).sum(dim=spatial) / (points + CHAMFER_EPS)
    return torch.cat(dice_sums + [(distance * chamfer_weights).sum(dim=1)])


def dice_and_chamfer(totals: torch.Tensor, chamfer_weight_sum: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Dice loss and the chamfer distance.

    Both come from the totals of :func:`label_sums`. The Dice loss is
    ``1 - (2 A + eps) / (B + eps)`` and the chamfer distance
    ``C / chamfer_weight_sum``, in band units.
    """
    dice = 1.0 - (2.0 * totals[0] + DICE_EPS) / (totals[1] + DICE_EPS)
    return dice, totals[2] / chamfer_weight_sum


def make_lncc() -> LocalNormalizedCrossCorrelationLoss:
    """The local NCC used on the anatomix channels.

    The loss uses a 3x3x3 box window, which is computed as one dense grouped
    convolution. Its ``forward_util`` returns the negative squared NCC of
    each voxel, and it checkpoints the box sums to save memory.
    """
    return LocalNormalizedCrossCorrelationLoss(
        spatial_dims=3, kernel_size=3, kernel_type="rectangular",
        reduction="none", checkpointing=True, use_separable_override=False)


def lncc_sum(lncc: LocalNormalizedCrossCorrelationLoss, moved: torch.Tensor,
             fixed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked sum of the negative squared local NCC over some channels."""
    return (lncc.forward_util(moved, fixed) * mask).sum(dim=(1, 2, 3, 4))


class Stage2Objective:
    """The image term of stage 2, evaluated in channel chunks.

    The image channels are the structure channels ``[0, num_structures)``
    followed by the anatomix channels. A chunk may hold channels of either
    block or of both. The sums ``S`` are the three label sums of
    :func:`label_sums`, the masked LNCC numerator, and the mask sum, which
    only the first chunk adds. The objective is

    ``dice_coeff * Dice + chamfer_coeff * chamfer + lncc_coeff * LNCC``.

    Parameters
    ----------
    num_structures
        Number of structure channels.
    dice_weights, chamfer_weights
        Weights ``(num_structures,)`` of the Dice and chamfer terms.
    shell
        Width of the soft structure membership, in band units.
    dice_coeff, chamfer_coeff, lncc_coeff
        Multipliers of the three terms.
    """

    num_stats = 5

    def __init__(self, num_structures: int, dice_weights: torch.Tensor,
                 chamfer_weights: torch.Tensor, shell: float, dice_coeff: float,
                 chamfer_coeff: float, lncc_coeff: float):
        self.num_structures = int(num_structures)
        self.dice_weights = dice_weights
        self.chamfer_weights = chamfer_weights
        self.chamfer_weight_sum = chamfer_weights.sum()
        self.shell = float(shell)
        self.dice_coeff = float(dice_coeff)
        self.chamfer_coeff = float(chamfer_coeff)
        self.lncc_coeff = float(lncc_coeff)
        self._lncc = {}
        self._weights = {}

    def _on_device(self, device):
        """The LNCC module and the weights on one device, created once."""
        if device not in self._lncc:
            self._lncc[device] = make_lncc().to(device)
            self._weights[device] = (self.dice_weights.to(device),
                                     self.chamfer_weights.to(device))
        return self._lncc[device], self._weights[device]

    def partial_sums(self, moved, fixed, chunk, level):
        """Sums ``(5,)`` of the channels ``[chunk.start, chunk.stop)``.

        The objective is the same on every level, so ``level`` is not read.
        """
        lncc, (dice_w, cham_w) = self._on_device(moved.device)
        zero = torch.zeros(1, device=moved.device, dtype=moved.dtype)
        split = min(max(self.num_structures, chunk.start), chunk.stop) - chunk.start
        width = chunk.stop - chunk.start
        label, numerator, mask_sum = zero.expand(3), zero, zero
        if split > 0:
            lo, hi = chunk.start, chunk.start + split
            mask = moved[:, -1:] * fixed[:, -1:]
            label = label_sums(moved[:, :split], fixed[:, :split], mask,
                               dice_w[lo:hi], cham_w[lo:hi], self.shell)
        if split < width:
            # Compute the mask product again instead of reusing the one of the
            # label sums: the separate product gives the gradient its own
            # accumulation path, and the results depend on that order at
            # float rounding.
            mask = moved[:, -1:] * fixed[:, -1:]
            numerator = lncc_sum(lncc, moved[:, split:width], fixed[:, split:width],
                                 mask)
        if chunk.index == 0:
            mask_sum = mask.sum(dim=(1, 2, 3, 4))
        return torch.cat([label, numerator, mask_sum])

    def combine(self, totals):
        """The objective from the summed ``S``."""
        dice, chamfer = dice_and_chamfer(totals[:3],
                                         self.chamfer_weight_sum.to(totals.device))
        label = self.dice_coeff * dice + self.chamfer_coeff * chamfer
        return label + self.lncc_coeff * masked_mean(totals[3], totals[4])


def structure_sizes(fixed: torch.Tensor, moved: torch.Tensor) -> torch.Tensor:
    """Mean voxel count of each structure on the two sides, at least 1.

    A voxel belongs to a structure where its distance channel is 0.
    """
    def counts(stack):
        return (stack[0] <= 0).sum(dim=(1, 2, 3)).float()

    return ((counts(fixed) + counts(moved)) / 2.0).clamp_min(1.0)


def structure_weights(sizes: torch.Tensor, exponent: float) -> torch.Tensor:
    """Weights ``N_c ** -exponent``, normalized to mean 1.

    ``sizes`` are the structure sizes of :func:`structure_sizes`, at least 1.
    """
    weights = sizes.pow(-exponent)
    return weights / weights.mean()


def sample_at_points(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample ``field (1, C, D0, D1, D2)`` at normalized points ``(n, 3)``.

    Trilinear interpolation, border padding, ``align_corners=True``.
    Returns ``(n, C)``.
    """
    query = points.reshape(1, 1, 1, -1, 3).to(field.dtype)
    sampled = F.grid_sample(field, query, mode="bilinear",
                            padding_mode="border", align_corners=True)
    return sampled[0, :, 0, 0].transpose(0, 1)


def interior_points(labels: torch.Tensor, ids: Sequence[int], mask: torch.Tensor,
                    max_points: int, seed: int) -> list[torch.Tensor]:
    """Voxel centres of each structure, as normalized sampling coordinates.

    Points outside the body mask are removed. A structure with more than
    ``max_points`` points keeps a random subset, drawn with one CPU generator
    shared by all structures in ``ids`` order, so the global random state is
    not used.

    Parameters
    ----------
    labels
        Integer label volume ``(1, 1, D0, D1, D2)``.
    ids
        Label values, in order.
    mask
        Body mask ``(1, 1, D0, D1, D2)``.
    max_points
        Largest number of points per structure.
    seed
        Seed of the generator.

    Returns
    -------
    list of torch.Tensor
        One ``(n_c, 3)`` tensor per id, components ``(x, y, z)``.
    """
    volume = labels[0, 0]
    sizes = tuple(int(s) for s in volume.shape)
    generator = torch.Generator().manual_seed(int(seed))
    points = []
    for label in ids:
        index = (volume == int(label)).nonzero().to(torch.float32)
        normalized = torch.empty_like(index)
        for comp in range(3):
            axis = 2 - comp
            normalized[:, comp] = index[:, axis] / (sizes[axis] - 1) * 2.0 - 1.0
        inside = sample_at_points(mask.float(), normalized)[:, 0] > 0.5
        normalized = normalized[inside]
        if normalized.shape[0] > max_points:
            keep = torch.randperm(normalized.shape[0], generator=generator)
            normalized = normalized[keep[:max_points].to(normalized.device)]
        points.append(normalized)
    return points


class RegionDistance(nn.Module):
    """Mean distance from moved fixed-side points to the native moving structures.

    The points of each fixed structure are moved by the stage-2 displacement
    and then by the stage-1 grid, and the distance channel of the same
    structure on the native moving grid is read at the result. The term is
    the mean over points, then the mean over structures, times ``coeff``.

    Parameters
    ----------
    points
        One ``(n_c, 3)`` tensor of normalized fixed-grid points per structure.
        Structures with no point are left out; at least one structure must
        have a point.
    distances
        Banded distance channels ``(1, K, Zm, Ym, Xm)`` of the native moving
        labels, in the order of ``points``.
    grid1
        Stage-1 sampling grid ``(1, Z, Y, X, 3)``.
    """

    def __init__(self, points: list[torch.Tensor], distances: torch.Tensor,
                 grid1: torch.Tensor):
        super().__init__()
        keep = [k for k in range(len(points)) if points[k].shape[0]]
        if not keep:
            raise ValueError("the region term has no point: no structure has a "
                             "voxel inside the fixed body mask")
        self.channels = keep
        self.slices = []
        start = 0
        for k in keep:
            self.slices.append((start, start + points[k].shape[0]))
            start += points[k].shape[0]
        self.points = torch.cat([points[k] for k in keep])
        self.distances = distances.detach()
        self.grid1_field = grid1.permute(0, 4, 1, 2, 3).detach()
        self.coeff = 1.0

    def value(self, disp: torch.Tensor | None = None) -> torch.Tensor:
        """The mean band-normalized distance; ``disp=None`` is the identity."""
        moved = self.points
        if disp is not None:
            moved = moved + sample_at_points(disp.permute(0, 4, 1, 2, 3), moved)
        moved = sample_at_points(self.grid1_field, moved)
        total = 0.0
        for (start, stop), channel in zip(self.slices, self.channels, strict=True):
            values = sample_at_points(self.distances[:, channel:channel + 1],
                                      moved[start:stop])[:, 0]
            total = total + values.mean()
        return total / len(self.channels)

    def forward(self, disp: torch.Tensor) -> torch.Tensor:
        return self.coeff * self.value(disp)


class VolumePenalty(nn.Module):
    """``lam`` times the weighted mean of ``(det J - 1)^2`` over a support.

    The weight map is given at full resolution on the fixed grid. For a
    displacement of another size (a level of the stage-1 pyramid) it is
    resized trilinearly to that size. The mean is taken over the interior
    voxels, where ``det J`` is defined. Where the weight is zero on all
    interior voxels, the penalty is zero.

    The registration samples the moving image at ``A x + b + u(x)`` in
    normalized coordinates. ``det J`` is the volume ratio of the whole map,
    ``det(A + Du) / det(A) = det(I + A^-1 Du)``. Without ``linear``, ``A`` is
    the identity and ``det J = det(I + Du)``.

    Parameters
    ----------
    weight
        Non-negative map ``(1, 1, Z, Y, X)`` on the fixed grid. It is divided
        by its maximum.
    lam
        Multiplier of the penalty.
    linear
        The ``(3, 3)`` matrix ``A`` in ``(x, y, z)`` order, or ``None`` when
        it is the identity.
    """

    def __init__(self, weight: torch.Tensor, lam: float,
                 linear: torch.Tensor | None = None):
        super().__init__()
        weight = weight.detach().float()
        peak = float(weight.max())
        self.weight_full = weight / peak if peak > 0 else weight
        self.lam = float(lam)
        # The displacement vectors are rows, so A^-1 u is u @ A^-T.
        self.inverse_t = None if linear is None else (
            torch.linalg.inv(linear.double()).T.float().to(weight.device))
        self._levels = {}

    def _at(self, size: tuple[int, ...]):
        """The interior weights at one size and their sum.

        The weights are ``None`` when their sum is zero.
        """
        if size not in self._levels:
            full = self.weight_full
            if tuple(full.shape[2:]) != size:
                full = F.interpolate(full, size=size, mode="trilinear",
                                     align_corners=True).clamp_min(0.0)
            weight = full[0, 0, 1:-1, 1:-1, 1:-1]
            total = weight.sum()
            self._levels[size] = (weight if float(total) > 0 else None, total)
        return self._levels[size]

    def forward(self, disp: torch.Tensor) -> torch.Tensor:
        """Penalty of a normalized displacement ``(1, Z, Y, X, 3)``."""
        weight, total = self._at(tuple(disp.shape[1:4]))
        if weight is None:
            return disp.sum() * 0.0
        if self.inverse_t is not None:
            disp = disp @ self.inverse_t
        det = jacobian_determinant(disp)[0]
        value = (weight * (det - 1.0) ** 2).sum() / total
        return self.lam * value


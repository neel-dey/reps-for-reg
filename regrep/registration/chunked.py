# Adapted from fireants/registration/greedy.py of FireANTs
# (https://github.com/rohitrango/FireANTs; regrep installs the main branch of
# the fork https://github.com/neel-dey/FireANTs).
#
# Copyright (c) 2026 Rohit Jena.
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0).
#
# Changes made for regrep: the optimization loop of
# GreedyRegistration.optimize is copied into ChunkedGreedyRegistration.optimize,
# and its image term is computed in channel chunks that can be on several
# GPUs.
"""Greedy deformable registration with the image term computed in channel chunks.

The image objective of both stages is a function of a few sums over channels
and voxels. ``ChunkedGreedyRegistration`` evaluates those sums one channel
chunk at a time, and the chunks can live on different GPUs. This bounds the
memory of the image term by the chunk size instead of the stack size. The
warp field, the optimizer and the displacement terms stay on the main device
(the device of the fixed stack).

With one chunk on one device the arithmetic is the same as FireANTs
``GreedyRegistration``: one forward pass and one backward pass. With several
chunks the result is the same up to the order of floating-point summation.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from fireants.interpolator import fireants_interpolator
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
from fireants.losses.maskedutils import DEFAULT_MASK_MODE, get_tensors_and_mask
from fireants.registration.greedy import GreedyRegistration
from fireants.utils.globals import MIN_IMG_SIZE
from fireants.utils.imageutils import downsample

from regrep.runtime import channel_ranges

MASK_EPS = 1e-8
"""Added to the mask sum before a masked mean, so that an empty mask gives 0."""


@dataclass(frozen=True)
class ChannelChunk:
    """A contiguous block of image channels and the device it is computed on.

    Attributes
    ----------
    index
        Position of the chunk in the list of chunks.
    start, stop
        The image channels ``[start, stop)`` of the full stack. The mask
        channel is not counted; every chunk carries a copy of it as its
        last channel.
    device
        The device that holds the chunk and computes its sums.
    """

    index: int
    start: int
    stop: int
    device: torch.device


def masked_mean(numerator: torch.Tensor, mask_sum: torch.Tensor) -> torch.Tensor:
    """Divide a masked sum by the sum of the mask (plus ``MASK_EPS``)."""
    return numerator / (mask_sum + MASK_EPS)


class ChunkedObjective(Protocol):
    """An image objective ``L = combine(S)`` with ``S`` a sum over chunks.

    Attributes
    ----------
    num_stats
        Length of the vector ``S``.
    """

    num_stats: int

    def partial_sums(self, moved: torch.Tensor, fixed: torch.Tensor,
                     chunk: ChannelChunk, level: int) -> torch.Tensor:
        """Return the contribution ``(num_stats,)`` of one chunk to ``S``.

        ``moved`` and ``fixed`` are ``(1, k + 1, Z, Y, X)`` on
        ``chunk.device``: the chunk's ``k`` image channels followed by the
        mask channel. ``moved`` is differentiable in the warp field.
        ``level`` is the index of the pyramid level, starting at 0.
        """
        ...

    def combine(self, totals: torch.Tensor) -> torch.Tensor:
        """Return the scalar objective from the summed ``S``."""
        ...


def check_fused_ops(device: torch.device) -> None:
    """Raise if the fused CUDA kernels of FireANTs are not used on ``device``.

    Without them FireANTs falls back to its PyTorch sampler, which gives
    different numbers and needs more memory than the run plan assumes. One
    small warp is run, so that kernels built for another GPU architecture
    fail here with a clear message.
    """
    hint = ("rebuild fireants_fused_ops for the installed torch and this GPU "
            "(see install.sh), and do not set USE_FFO=false")
    if not fireants_interpolator.use_ffo:
        raise RuntimeError(f"the fused FireANTs kernels are not in use; {hint}")
    volume = torch.zeros((1, 1, 2, 2, 2), device=device)
    grid = torch.zeros((1, 2, 2, 2, 3), device=device)
    affine = torch.eye(3, 4, device=device)[None]
    try:
        fireants_interpolator(volume, affine=affine, grid=grid, mode="bilinear",
                              align_corners=True, is_displacement=True)
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        raise RuntimeError(f"the fused FireANTs kernels fail on {device} ({error}); "
                           f"{hint}") from error


def _on(device: torch.device):
    """Make ``device`` the current CUDA device inside the block."""
    if device.type == "cuda":
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def split_channels(num_channels: int, devices: Sequence[torch.device],
                   chunk_channels: int | None) -> list[ChannelChunk]:
    """Split the image channels into contiguous chunks over the devices.

    Each device gets a contiguous block of near-equal size, and each block is
    cut into the fewest near-equal chunks of at most ``chunk_channels``
    channels (``regrep.runtime.channel_ranges``, which the run plan uses
    too). The first device is the main device.

    Parameters
    ----------
    num_channels
        Number of image channels, without the mask channel.
    devices
        The devices to compute on. CUDA devices must have an index.
    chunk_channels
        Largest chunk size. ``None`` gives one chunk per device.
    """
    devices = list(devices)
    chunks = []
    ranges = channel_ranges(num_channels, len(devices), chunk_channels)
    for device, blocks in zip(devices, ranges, strict=True):
        for start, stop in blocks:
            chunks.append(ChannelChunk(len(chunks), start, stop, device))
    return chunks


class StackImages:
    """A channel stack with its image geometry, in the form FireANTs reads.

    This stands in for ``fireants.io.BatchedImages`` with one image. The
    array is ``(1, C, Z, Y, X)``; ``torch2phy`` and ``phy2torch`` are the
    ``(1, 4, 4)`` maps between normalized and physical coordinates.

    :meth:`release` drops the reference to the array. The object keeps its
    shape, device and geometry, which is all FireANTs reads from the stack
    once the registration holds the channel chunks.
    """

    interpolate_mode = "trilinear"

    def __init__(self, array: torch.Tensor, torch2phy: torch.Tensor,
                 phy2torch: torch.Tensor):
        self.array = array
        self._shape = array.shape
        self.torch2phy = torch2phy.to(array.device)
        self.phy2torch = phy2torch.to(array.device)

    def __call__(self) -> torch.Tensor:
        return self.array

    def release(self) -> None:
        """Replace the array by an empty tensor on the same device.

        The memory of the stack is freed when no other reference to it
        exists.
        """
        self.array = self.array.new_empty((self._shape[0], 0))

    def get_torch2phy(self) -> torch.Tensor:
        return self.torch2phy

    def get_phy2torch(self) -> torch.Tensor:
        return self.phy2torch

    def size(self) -> int:
        return self.array.shape[0]

    @property
    def device(self) -> torch.device:
        return self.array.device

    @property
    def dims(self) -> int:
        return len(self._shape) - 2

    @property
    def shape(self) -> torch.Size:
        return self._shape


class MaskedLNCC:
    """Masked local normalized cross-correlation, summed over channels.

    This is the FireANTs ``masked_cc`` loss: the negative squared local NCC of
    each image channel, with a box window, multiplied by the product of the
    moved and fixed masks, summed over channels and voxels, and divided by
    the sum of the mask. ``S`` holds the masked numerator and the mask sum;
    only the first chunk adds the mask sum.

    Parameters
    ----------
    kernel_sizes
        Box width of the window at each pyramid level.
    """

    num_stats = 2

    def __init__(self, kernel_sizes: Sequence[int]):
        self.kernel_sizes = [int(k) for k in kernel_sizes]
        self._losses = {}

    def _loss(self, level: int, device: torch.device):
        key = (self.kernel_sizes[level], device)
        if key not in self._losses:
            loss = LocalNormalizedCrossCorrelationLoss(
                spatial_dims=3, kernel_size=key[0], kernel_type="rectangular",
                reduction="none", use_separable_override=False)
            self._losses[key] = loss.to(device)
        return self._losses[key]

    def partial_sums(self, moved, fixed, chunk, level):
        pred, target, mask = get_tensors_and_mask(moved, fixed, DEFAULT_MASK_MODE)
        negative_ncc = self._loss(level, moved.device).forward_util(pred, target)
        dims = tuple(range(1, negative_ncc.dim()))
        numerator = (negative_ncc * mask).sum(dim=dims)
        if chunk.index == 0:
            mask_sum = mask.sum(dim=dims)
        else:
            mask_sum = torch.zeros_like(numerator)
        return torch.cat([numerator, mask_sum])

    def combine(self, totals):
        return masked_mean(totals[0:1], totals[1:2]).mean()


class ChunkedGreedyRegistration(GreedyRegistration):
    """Compositive greedy registration with a chunked image objective.

    Parameters
    ----------
    scales, iterations
        Downsampling factor and iteration budget of each pyramid level.
    fixed_images, moving_images
        ``StackImages`` whose last channel is the mask, with at least
        ``MIN_IMG_SIZE`` (32) voxels along each axis, the size of the coarsest
        level of the FireANTs pyramid. The fixed stack's device is the main
        device. With more than one chunk, or a chunk on
        another device, the registration copies the chunks and releases the
        stacks (see :meth:`StackImages.release`); a caller that wants the
        memory back must not keep other references to the arrays.
    objective
        The image term, a ``ChunkedObjective``.
    chunks
        The channel chunks, from :func:`split_channels`.
    optimizer_lr
        Adam step size, in half-voxels of the longest axis of each level.
    smooth_grad_sigma, smooth_warp_sigma
        Gaussian sigmas, in voxels of the level, for the gradient and for the
        warp after each step.
    displacement_reg
        Callable of the warp field (normalized displacement,
        ``(1, Z, Y, X, 3)``) added to the loss on the main device.
    tolerance, max_tolerance_iters
        A level stops when the least-squares slope of the last
        ``max_tolerance_iters`` loss values exceeds ``tolerance``.
    init_affine
        Optional ``(1, 3, 4)`` or ``(1, 4, 4)`` physical affine applied
        before the warp.
    """

    def __init__(self, scales: Sequence[float], iterations: Sequence[int],
                 fixed_images, moving_images, objective: ChunkedObjective,
                 chunks: Sequence[ChannelChunk], optimizer_lr: float,
                 smooth_grad_sigma: float, smooth_warp_sigma: float,
                 displacement_reg: Callable[[torch.Tensor], torch.Tensor],
                 tolerance: float, max_tolerance_iters: int,
                 init_affine: torch.Tensor | None = None):
        for images in (fixed_images, moving_images):
            if min(images.shape[2:]) < MIN_IMG_SIZE:
                raise ValueError(f"each image axis needs at least {MIN_IMG_SIZE} "
                                 f"voxels, got {tuple(images.shape[2:])}")
        super().__init__(
            scales=list(scales), iterations=list(iterations),
            fixed_images=fixed_images, moving_images=moving_images,
            loss_type="custom", custom_loss=objective, loss_params={},
            deformation_type="compositive", optimizer="Adam",
            optimizer_lr=optimizer_lr, smooth_grad_sigma=smooth_grad_sigma,
            smooth_warp_sigma=smooth_warp_sigma, displacement_reg=displacement_reg,
            init_affine=init_affine, tolerance=tolerance,
            max_tolerance_iters=max_tolerance_iters, progress_bar=False)
        self.objective = objective
        self.chunks = list(chunks)
        # Copy and release the fixed stack before the moving stack is copied.
        # The peak is then one stack, its chunk copies and the other stack.
        self.fixed_chunks = self._split(fixed_images)
        self.moving_chunks = self._split(moving_images)

    def _split(self, images: StackImages) -> list[torch.Tensor]:
        """Copy each chunk's channels, plus the mask channel, to its device.

        A single chunk on the stack's own device uses the stack itself.
        Otherwise the stack is released after the copies are made, so the
        registration does not hold the channels twice.
        """
        arrays = images()
        if len(self.chunks) == 1 and self.chunks[0].device == arrays.device:
            return [arrays]
        mask = arrays[:, -1:]
        chunks = [torch.cat([arrays[:, c.start:c.stop].to(c.device),
                             mask.to(c.device)], dim=1)
                  for c in self.chunks]
        del arrays, mask
        images.release()
        return chunks

    @staticmethod
    def _image_range(chunks: list[torch.Tensor]) -> tuple[float, float]:
        """Minimum and maximum over the image channels of all chunks."""
        low = min(c[:, :-1].min().item() for c in chunks)
        high = max(c[:, :-1].max().item() for c in chunks)
        return low, high

    def _level_arrays(self, chunks, size, clamp_range):
        """Resize each chunk to one pyramid level.

        The image channels are downsampled with the FFT Gaussian of FireANTs
        and clamped to one range for all chunks, as if the whole stack were
        downsampled at once. The mask channel is resized trilinearly.
        """
        out = []
        for chunk, arrays in zip(self.chunks, chunks, strict=True):
            with _on(chunk.device):
                image = downsample(arrays[:, :-1], size=size, mode="trilinear",
                                   clamp_range=clamp_range)
                mask = F.interpolate(arrays[:, -1:], size=size, mode="trilinear",
                                     align_corners=True)
                out.append(torch.cat([image, mask], dim=1))
        return out

    def optimize(self):
        """Run the pyramid of greedy deformable iterations."""
        fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = self.moving_images.get_phy2torch().to(self.dtype)
        fixed_size = self.fixed_images.shape[2:]
        moving_size = self.moving_images.shape[2:]
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))
        affine_map_init = affine_map_init[:, :-1].contiguous().to(self.dtype)
        affines = {c.device: affine_map_init.to(c.device) for c in self.chunks}
        fixed_range = self._image_range(self.fixed_chunks)
        moving_range = self._image_range(self.moving_chunks)

        levels = zip(self.scales, self.iterations, strict=True)
        for level, (scale, iters) in enumerate(levels):
            self.convergence_monitor.reset()
            size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]
            moving_down = [max(int(s / scale), MIN_IMG_SIZE) for s in moving_size]
            if scale > 1:
                fixed = self._level_arrays(self.fixed_chunks, size_down, fixed_range)
                moving = self._level_arrays(self.moving_chunks, moving_down,
                                            moving_range)
            else:
                fixed, moving = self.fixed_chunks, self.moving_chunks

            self.warp.set_size(size_down)
            for _ in range(iters):
                self.warp.set_zero_grad()
                warp_field = self.warp.get_warp()
                if len(self.chunks) == 1:
                    loss = self._backward_whole(warp_field, moving[0], fixed[0],
                                                affine_map_init, level)
                else:
                    loss = self._backward_chunked(warp_field, moving, fixed, affines,
                                                  level)
                self.warp.step()
                if self.convergence_monitor.converged(loss):
                    break

    def _backward_whole(self, warp_field, moving, fixed, affine, level) -> float:
        """Backpropagate the loss of a single chunk and return its value."""
        moved = fireants_interpolator(moving, affine=affine, grid=warp_field,
                                      mode="bilinear", align_corners=True,
                                      is_displacement=True)
        loss = self.objective.combine(
            self.objective.partial_sums(moved, fixed, self.chunks[0], level))
        loss = loss + self.displacement_reg(warp_field)
        loss.backward()
        return loss.item()

    def _backward_chunked(self, warp_field, moving, fixed, affines, level) -> float:
        """Backpropagate the loss one chunk at a time and return its value."""
        main = warp_field.device
        objective = self.objective
        # A detached copy of the field on each device collects the gradient of
        # the chunks computed there.
        leaves = {}
        for chunk in self.chunks:
            if chunk.device not in leaves:
                leaves[chunk.device] = (warp_field.detach().to(chunk.device)
                                        .requires_grad_())
        work = list(zip(self.chunks, moving, fixed, strict=True))

        def chunk_sums(chunk, moving_chunk, fixed_chunk):
            moved = fireants_interpolator(moving_chunk, affine=affines[chunk.device],
                                          grid=leaves[chunk.device], mode="bilinear",
                                          align_corners=True, is_displacement=True)
            return objective.partial_sums(moved, fixed_chunk, chunk, level)

        # Pass 1 finds the sums S without gradients; the gradient of the
        # objective with respect to S then weights the sums of each chunk in
        # pass 2, which backpropagates one chunk at a time.
        totals = torch.zeros(objective.num_stats, device=main, dtype=warp_field.dtype)
        with torch.no_grad():
            for item in work:
                with _on(item[0].device):
                    totals += chunk_sums(*item).to(main)
        point = totals.clone().requires_grad_()
        (weights,) = torch.autograd.grad(objective.combine(point), point)

        for item in work:
            with _on(item[0].device):
                sums = chunk_sums(*item)
                (sums * weights.to(sums.device)).sum().backward()
            del sums

        grad = None
        for leaf in leaves.values():
            part = leaf.grad.to(main)
            grad = part if grad is None else grad + part
        with torch.no_grad():
            value = objective.combine(totals)
        reg = self.displacement_reg(warp_field)
        torch.autograd.backward([warp_field, reg], [grad, None])
        return (value + reg.detach()).item()

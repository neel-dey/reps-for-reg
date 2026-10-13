# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from MONAI 1.6.0 (https://github.com/Project-MONAI/MONAI), files
# monai/inferers/utils.py and monai/data/utils.py, whose header is the one
# above. Changes made for regrep: the window geometry, the padding and the
# Gaussian importance map of sliding_window_inference (_scan_interval,
# _window_starts, _window_padding and _importance_map here) are rewritten as
# plain functions that do not import MONAI. The rest of this file is covered
# by the license of this repository.
"""anatomix features: a multi-resolution pyramid of the anatomix ViT.

The CT is resampled to three isotropic grids. On each grid the network runs
in overlapping 128-voxel windows with Gaussian blending, and the result is L2
normalized per voxel. The three levels are resampled to the isotropic grid
whose spacing is the smallest voxel size of the fixed CT (the third level),
summed, resampled to the native grid and L2 normalized again.

A level grid that is shorter than the window along an axis is padded with
zeros up to the window size, half on each side, and the output is cropped
back, as MONAI's ``sliding_window_inference`` does.

The network runs under fp16 autocast and the blended sum is accumulated in
fp16. Windows that hold no body voxel are skipped.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from anatomix.model.load_from_hf import load_from_hf

from regrep.config import FeatureConfig

# Side length of the blocks in which the body mask is tested for each window.
_MASK_BLOCK = 8

# Channels per call when feature maps are resampled. Interpolation is per
# channel, so the chunk size changes only the size of the temporary buffers.
_RESAMPLE_CHUNK = 4

# Floor of the blending weight sum before the division. MONAI uses 1e-8,
# which is zero in fp16. Every covered voxel has a weight sum of at least
# 1e-3, so the floor only matters where no window was run.
_COUNT_FLOOR = 1e-4


def load_backbone(cfg: FeatureConfig, device: torch.device) -> torch.nn.Module:
    """Load the anatomix network from the HuggingFace Hub for inference.

    The model is built with ``cfg.backbone_init_threads`` CPU threads, which
    makes the random initialization (overwritten by the weights) faster. The
    previous thread count is restored afterwards.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(cfg.backbone_init_threads)
    try:
        model = load_from_hf(cfg.backbone)
    finally:
        torch.set_num_threads(previous)
    return model.to(device).eval().requires_grad_(False)


def grid_shape(
    shape: tuple[int, ...], spacing: tuple[float, ...], target: float
) -> tuple[int, ...]:
    """Shape of the isotropic grid with spacing ``target`` over the same extent.

    The first and last voxel centres stay in place (the convention of
    ``align_corners=True``).
    """
    return tuple(
        int(round((n - 1) * float(s) / float(target))) + 1
        for n, s in zip(shape, spacing, strict=True)
    )


def resample_channels(
    feats: torch.Tensor, shape: tuple[int, ...], out_dtype=None
) -> torch.Tensor:
    """Trilinearly resample a ``(1, C, ...)`` map to ``shape``, in chunks.

    Each chunk is converted to fp32 before the interpolation. The output has
    ``out_dtype`` (default: the input dtype).
    """
    dtype = out_dtype or feats.dtype
    out = torch.empty(
        (feats.shape[0], feats.shape[1]) + tuple(shape),
        device=feats.device,
        dtype=dtype,
    )
    for start in range(0, feats.shape[1], _RESAMPLE_CHUNK):
        block = feats[:, start:start + _RESAMPLE_CHUNK]
        out[:, start:start + _RESAMPLE_CHUNK] = F.interpolate(
            block.float(), size=shape, mode="trilinear", align_corners=True
        ).to(dtype)
    return out


def l2_normalize_(feats: torch.Tensor) -> torch.Tensor:
    """Divide each voxel's channel vector by its L2 norm, in place.

    The norm is accumulated in fp32. Zero vectors stay zero: the divisor is
    floored at 1e-12, and for fp16 maps also at the smallest normal fp16
    number, because 1e-12 is zero in fp16.
    """
    norm = torch.linalg.vector_norm(
        feats, ord=2, dim=1, keepdim=True, dtype=torch.float32
    )
    divisor = norm.clamp_min_(1e-12).to(feats.dtype)
    if divisor.dtype != torch.float32:
        divisor = divisor.clamp_min_(torch.finfo(divisor.dtype).tiny)
    return feats.div_(divisor)


def _scan_interval(
    image_size: tuple[int, ...], roi: tuple[int, ...], overlap: float
) -> tuple[int, ...]:
    """Stride between windows along each axis (MONAI ``_get_scan_interval``)."""
    interval = []
    for size, window in zip(image_size, roi, strict=True):
        if window == size:
            interval.append(int(window))
        else:
            interval.append(int(window * (1 - overlap)))
    return tuple(interval)


def _window_starts(
    image_size: tuple[int, ...], roi: tuple[int, ...], interval: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """Start corners of all windows (MONAI ``dense_patch_slices``).

    Each axis is covered by the fewest windows at the given stride; the last
    window is shifted back to end at the border. The windows are listed with
    the first axis varying slowest, which is the order of accumulation.
    """
    starts = []
    for size, window, step in zip(image_size, roi, interval, strict=True):
        count = int(math.ceil(float(size) / step))
        reaches = [i for i in range(count) if i * step + window >= size]
        number = reaches[0] + 1
        axis_starts = []
        for i in range(number):
            start = i * step
            axis_starts.append(start - max(start + window - size, 0))
        starts.append(axis_starts)
    grid = np.asarray([g.flatten() for g in np.meshgrid(*starts, indexing="ij")]).T
    return [tuple(int(s) for s in row) for row in grid]


def _importance_map(
    roi: tuple[int, ...], sigma_scale: float, device: torch.device
) -> torch.Tensor:
    """Gaussian blending weight of one window (MONAI ``compute_importance_map``).

    The weight is the outer product of one 1-D Gaussian per axis with sigma
    ``sigma_scale * window``, clamped below at ``max(min, 1e-3)``. It is not
    normalized. Returns a ``(1, 1, *roi)`` fp32 tensor.
    """
    importance = None
    for axis, size in enumerate(roi):
        sigma = size * sigma_scale
        grid = torch.arange(
            start=-(size - 1) / 2.0,
            end=(size - 1) / 2.0 + 1,
            dtype=torch.float,
            device=device,
        )
        grid = torch.exp(grid**2 / (-2 * sigma**2))
        importance = (
            grid if axis == 0 else importance.unsqueeze(-1) * grid[(None,) * axis]
        )
    smallest = max(torch.min(importance).item(), 1e-3)
    return torch.clamp_(importance, min=smallest)[None, None]


def _kept_windows(
    starts: list[tuple[int, ...]],
    window: int,
    mask: torch.Tensor,
    margin: int,
) -> list[tuple[int, ...]]:
    """The windows that have a body voxel within ``margin`` voxels.

    The mask is reduced to one bit per block of 8 x 8 x 8 voxels by a max
    pool, which can only keep more windows than an exact test would.
    """
    coarse = F.max_pool3d(mask, _MASK_BLOCK, stride=_MASK_BLOCK, ceil_mode=True)
    coarse = (coarse[0, 0] > 0).cpu()
    kept = []
    for start in starts:
        index = tuple(
            slice(
                max(0, (s - margin) // _MASK_BLOCK),
                min(n, -(-(s + window + margin) // _MASK_BLOCK)),
            )
            for s, n in zip(start, coarse.shape, strict=True)
        )
        if bool(coarse[index].any()):
            kept.append(start)
    return kept


def _window_padding(shape: tuple[int, ...], window: int) -> tuple[int, ...]:
    """Zero padding that makes every axis at least ``window`` voxels long.

    The padding of an axis is split as MONAI's ``sliding_window_inference``
    splits it: half (rounded down) before and the rest after. The result is
    in the order of ``F.pad``, last axis first.
    """
    pad = []
    for size in reversed(shape):
        missing = max(window - int(size), 0)
        pad += [missing // 2, missing - missing // 2]
    return tuple(pad)


def body_windows(
    image_size: tuple[int, ...], mask: torch.Tensor, cfg: FeatureConfig
) -> list[tuple[int, ...]]:
    """Start corners of the windows to run on a grid of ``image_size``.

    Every axis must be at least ``cfg.window`` voxels long. Windows with no
    body voxel within ``cfg.window_margin`` voxels of them are left out. They
    cover only voxels outside the body, which every later step masks out.
    ``mask`` is the body mask on that grid, fp32 in {0, 1}.
    """
    roi = (cfg.window,) * 3
    interval = _scan_interval(image_size, roi, cfg.window_overlap)
    starts = _window_starts(image_size, roi, interval)
    return _kept_windows(starts, roi[0], mask, cfg.window_margin)


def sliding_window(
    volume: torch.Tensor,
    model: torch.nn.Module,
    starts: list[tuple[int, ...]],
    cfg: FeatureConfig,
) -> torch.Tensor:
    """Run the network over a volume in blended windows.

    Parameters
    ----------
    volume
        Input of shape ``(1, 1, D, H, W)``, fp32.
    model
        The anatomix network.
    starts
        Start corners of the windows to run, from :func:`body_windows`.
    cfg
        Feature settings.

    Returns
    -------
    torch.Tensor
        The Gaussian-weighted mean of the window outputs, fp16, of shape
        ``(1, C, D, H, W)``. Voxels that no kept window covers are zero.
    """
    image_size = tuple(int(n) for n in volume.shape[2:])
    roi = (cfg.window,) * 3
    weight = _importance_map(roi, cfg.window_sigma, volume.device)
    weight = weight.to(torch.float16)

    out = count = None
    for position in range(0, len(starts), cfg.window_batch):
        chunk = [
            tuple(slice(s, s + w) for s, w in zip(start, roi, strict=True))
            for start in starts[position:position + cfg.window_batch]
        ]
        data = torch.cat([volume[(slice(None), slice(None)) + s] for s in chunk])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            prediction = model(data)
        del data
        if out is None:
            out = torch.zeros(
                (1, prediction.shape[1]) + image_size,
                dtype=torch.float16,
                device=volume.device,
            )
            count = torch.zeros(
                (1, 1) + image_size, dtype=torch.float16, device=volume.device
            )
        prediction = prediction.to(torch.float16).mul_(weight)
        # One window at a time, in order, so the fp16 sums are reproducible.
        for index, s in enumerate(chunk):
            key = (slice(0, 1), slice(None)) + s
            out[key] += prediction[index:index + 1]
            count[key] += weight
        del prediction
    if out is None:
        raise ValueError("the body mask selects no window")
    out.div_(count.clamp_min_(_COUNT_FLOOR))
    del count
    return out


def run_level(
    volume: torch.Tensor,
    mask: torch.Tensor,
    model: torch.nn.Module,
    cfg: FeatureConfig,
) -> torch.Tensor:
    """Run the network over one pyramid level in blended windows.

    An axis shorter than the window is padded with zeros up to the window
    size, as MONAI's ``sliding_window_inference`` does, and the output is
    cropped back to the size of ``volume``.

    Parameters
    ----------
    volume
        The level, ``(1, 1, D, H, W)`` fp32.
    mask
        The body mask on the level grid, ``(1, 1, D, H, W)`` fp32 in {0, 1}.
    model
        The anatomix network.
    cfg
        Feature settings.

    Returns
    -------
    torch.Tensor
        The output of :func:`sliding_window`, ``(1, C, D, H, W)`` fp16.
    """
    shape = tuple(int(n) for n in volume.shape[2:])
    pad = _window_padding(shape, cfg.window)
    if any(pad):
        volume = F.pad(volume, pad)
        mask = F.pad(mask, pad)
    starts = body_windows(tuple(int(n) for n in volume.shape[2:]), mask, cfg)
    feats = sliding_window(volume, model, starts, cfg)
    if any(pad):
        # F.pad lists the last axis first.
        crop = tuple(slice(pad[4 - 2 * axis], pad[4 - 2 * axis] + shape[axis])
                     for axis in range(3))
        feats = feats[(slice(None), slice(None)) + crop].contiguous()
    return feats


def anatomix_features(
    ct_norm: torch.Tensor,
    spacing: tuple[float, float, float],
    model: torch.nn.Module,
    body: torch.Tensor,
    cfg: FeatureConfig,
    ct_mind: torch.Tensor,
    sum_mm: float,
) -> torch.Tensor:
    """Compute the 32 anatomix channels of a CT on its native grid.

    Parameters
    ----------
    ct_norm
        Normalized CT of shape ``(1, 1, D, H, W)``, fp32.
    spacing
        Voxel spacing (mm) in the array's axis order.
    model
        The anatomix network.
    body
        Body mask on the native grid, ``(1, 1, D, H, W)`` fp32 in {0, 1}.
    cfg
        Feature settings.
    ct_mind
        ``ct_norm`` resampled to the MIND-SSC spacing
        (``cfg.mind_spacing_mm``). When that spacing is a pyramid level, this
        volume is used instead of resampling ``ct_norm`` again.
    sum_mm
        The spacing (mm) of the isotropic grid on which the levels are
        summed. The ``None`` entry of ``cfg.pyramid_mm`` stands for it.

    Returns
    -------
    torch.Tensor
        Features of shape ``(1, 32, D, H, W)``, fp32, with unit L2 norm per
        voxel (zero where no window was run). They are not masked.
    """
    native = tuple(int(n) for n in ct_norm.shape[-3:])
    iso_shape = grid_shape(native, spacing, sum_mm)

    total = None
    for target in cfg.pyramid_mm:
        target = sum_mm if target is None else target
        shape = grid_shape(native, spacing, target)
        if target == cfg.mind_spacing_mm:
            volume = ct_mind
        else:
            volume = F.interpolate(
                ct_norm, size=shape, mode="trilinear", align_corners=True
            )
        level_mask = F.interpolate(body, size=shape, mode="nearest")
        feats = run_level(volume, level_mask, model, cfg)
        del volume, level_mask
        torch.cuda.empty_cache()

        l2_normalize_(feats)
        if shape != iso_shape:
            feats = resample_channels(feats, iso_shape, out_dtype=torch.float32)
        else:
            feats = feats.float()
        total = feats if total is None else total.add_(feats)
        del feats
        torch.cuda.empty_cache()

    primary = resample_channels(total, native)
    del total
    return l2_normalize_(primary)

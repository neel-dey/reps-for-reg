# This file adapts code from two projects:
#
# TotalSegmentator, https://github.com/wasserth/TotalSegmentator, by Jakob
# Wasserthal. The TotalSegmentator repository states no copyright line.
#
# nnU-Net, https://github.com/MIC-DKFZ/nnUNet. The LICENSE file of nnU-Net
# states:
#
#     Copyright [2019] [Division of Medical Image Computing, German Cancer
#     Research Center (DKFZ), Heidelberg, Germany]
#
# The nnU-Net NIfTI reader (nnunetv2/imageio/nibabel_reader_writer.py)
# states:
#
#     Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical
#     Image Computing, German Cancer Research Center (DKFZ), Heidelberg,
#     Germany
#
# Both projects use the Apache License, Version 2.0. For the parts adapted
# from them:
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
# Changes made for regrep: the resampling of cropped logits to the native
# grid, the input and output resampling of the CT, the crop to a coarse
# segmentation, the merge of the part models and an in-memory version of
# nnU-Net's NIfTI reader are adapted from TotalSegmentator and nnU-Net. They
# run on the GPU here instead of on the CPU, with the same coordinate
# conventions.
"""Organ segmentation of CT volumes with the TotalSegmentator models.

``OrganSegmenter`` loads seven nnU-Net models of the TotalSegmentator
``total`` task on one GPU and makes these label maps for a pair of CTs:

* ``organs6_moving``: the 6 mm model (task 298) on the moving CT, with the
  logits resampled to the CT grid (TotalSegmentator's ``-ho`` option).
* ``organs3_moving``: the same with the 3 mm model (task 297).
* ``labels_fixed`` and ``labels_moving``: the 117-class map at 1.5 mm. For
  each group of part models the CT is cropped to a box around voxels of a
  6 mm map (all labelled voxels for tasks 291, 293 and 294; only the rib and
  vertebra voxels for tasks 292 and 295), widened by ``crop_margin_mm``, and
  resampled to 1.5 mm. Tasks 291, 293 and 294 give 65 classes; tasks 292
  and 295 give the 52 rib and vertebra classes, which are painted over them.
  The rib and vertebra map comes back to the CT grid through a linear
  resample of each class indicator followed by an argmax; the other map
  comes back by nearest-neighbour resampling.

Every map is a uint8 array in ``(X, Y, Z)`` order on the grid of its CT.
"""

from __future__ import annotations

import math
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.preprocessing.resampling.default_resampling import (
    determine_do_sep_z_and_axis,
)
from totalsegmentator.cropping import crop_to_mask
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.map_to_binary import class_map_5_parts, map_taskid_to_partname_ct

from regrep.config import OrganSegmentationConfig
from regrep.io import Volume
from regrep.segmentation import cudnn_flags
from regrep.segmentation.labels import (
    ORGAN_MAPS,
    RIB_VERTEBRA_IDS,
    class_ids_by_name,
    load_class_map,
    merge_label_maps,
    rib_vertebra_lut,
)
from regrep.weights import TOTALSEG_TASKS, totalseg_model_folder, totalseg_task_config

ORGANS_6MM_TASK = 298
ORGANS_3MM_TASK = 297
RIB_VERTEBRA_TASKS = (292, 295)
FULL_TASKS = (291, 293, 294)

# Margin, in low-resolution voxels, around the bounding box of each class when
# the logits are resampled to the CT grid (TotalSegmentator's value).
LOGIT_CROP_MARGIN = 3


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def download_weights() -> None:
    """Download the seven models to TotalSegmentator's weight folder.

    Models that are already present are not downloaded again.
    """
    for task_id in TOTALSEG_TASKS:
        download_pretrained_weights(task_id)


# ---------------------------------------------------------------------------
# Separable resampling on the GPU
# ---------------------------------------------------------------------------


def _interp_axis(
    volume: torch.Tensor, coords: torch.Tensor, axis: int, order: int
) -> torch.Tensor:
    """Sample ``volume`` along one axis at the positions ``coords``.

    Out-of-range positions read the edge value (SciPy's ``mode='nearest'``).
    Order 0 takes the nearest voxel; order 1 interpolates linearly as
    ``(1 - f) * a + f * b`` with float64 weights.
    """
    size = volume.shape[axis]
    if order == 0:
        index = torch.floor(coords + 0.5).long().clamp_(0, size - 1)
        return volume.index_select(axis, index)
    low = torch.floor(coords)
    frac = coords - low
    # Clamp both taps, so that a position outside the volume reads the edge
    # value through each of them.
    below = low.long().clamp(0, size - 1)
    above = (low.long() + 1).clamp(0, size - 1)
    shape = [1] * volume.dim()
    shape[axis] = -1
    frac = frac.reshape(shape)
    return (1.0 - frac) * volume.index_select(axis, below) + frac * volume.index_select(
        axis, above
    )


def resample_separable(
    volume: torch.Tensor, coords: list[torch.Tensor], order: int | tuple[int, ...]
) -> torch.Tensor:
    """Resample a 3-D volume at the outer product of three coordinate vectors.

    The axes are processed one after another, in the order 0, 1, 2. ``order``
    is the interpolation order of every axis, or a sequence of one order per
    axis.
    """
    orders = (order,) * 3 if isinstance(order, int) else tuple(order)
    for axis in range(3):
        volume = _interp_axis(volume, coords[axis], axis, orders[axis])
    return volume


def corner_aligned_coords(n_in: int, n_out: int, device: torch.device) -> torch.Tensor:
    """Return the sample positions of ``scipy.ndimage.zoom`` (``grid_mode=False``).

    The first and last output voxels fall on the first and last input voxels.
    """
    step = (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
    return torch.arange(n_out, dtype=torch.float64, device=device) * step


def half_pixel_coords(n_in: int, n_out: int, device: torch.device) -> torch.Tensor:
    """Return the sample positions of ``skimage.transform.resize``."""
    return (torch.arange(n_out, dtype=torch.float64, device=device) + 0.5) * (
        n_in / n_out
    ) - 0.5


def zoom_linear(volume: torch.Tensor, zoom: np.ndarray) -> torch.Tensor:
    """Linear ``scipy.ndimage.zoom(volume, zoom, order=1, mode='nearest')``.

    ``volume`` is float64. The output shape is ``round(n * zoom)`` per axis.
    """
    shape = [int(round(n * z)) for n, z in zip(volume.shape, zoom, strict=True)]
    coords = [
        corner_aligned_coords(n_in, n_out, volume.device)
        for n_in, n_out in zip(volume.shape, shape, strict=True)
    ]
    return resample_separable(volume, coords, order=1)


def resample_probabilities(
    channel: torch.Tensor, shape: tuple[int, int, int], lowres_axis: int | None
) -> torch.Tensor:
    """Resample one channel of logits as nnU-Net's ``resample_data_or_seg`` does.

    The sample positions are those of ``skimage.transform.resize``. Without
    ``lowres_axis`` every axis is interpolated linearly. With it (nnU-Net's
    separate-z resampling of anisotropic data) the other two axes are
    interpolated linearly and ``lowres_axis`` takes the nearest voxel.
    ``channel`` is float64.
    """
    coords = [
        half_pixel_coords(n_in, n_out, channel.device)
        for n_in, n_out in zip(channel.shape, shape, strict=True)
    ]
    orders = tuple(0 if axis == lowres_axis else 1 for axis in range(3))
    return resample_separable(channel, coords, orders)


def resample_logits_to_labels(
    logits: torch.Tensor, shape: tuple[int, int, int], lowres_axis: int | None
) -> torch.Tensor:
    """Resample logits to a finer grid linearly and take the argmax.

    This is TotalSegmentator's memory-saving export: each foreground class is
    resampled linearly, only inside the bounding box of its low-resolution
    argmax widened by ``LOGIT_CROP_MARGIN`` voxels, and it wins a voxel when
    its value is larger than the best value so far. The background channel
    seeds the best value; it is resampled with nnU-Net's resampling of
    probabilities (:func:`resample_probabilities`). The best values are
    stored in the dtype of ``logits`` (float16), while each comparison is made
    in float64.

    Parameters
    ----------
    logits
        ``(C, n0, n1, n2)`` tensor, channel 0 is background.
    shape
        The output grid ``(m0, m1, m2)``.
    lowres_axis
        The axis that nnU-Net resamples separately for anisotropic data, or
        ``None`` (see ``determine_do_sep_z_and_axis``).

    Returns
    -------
    torch.Tensor
        uint8 labels of shape ``shape`` on the device of ``logits``.
    """
    old_shape = tuple(logits.shape[1:])
    shape = tuple(int(n) for n in shape)
    device = logits.device
    best = resample_probabilities(logits[0].to(torch.float64), shape, lowres_axis)
    best = best.to(logits.dtype)
    labels = torch.zeros(shape, dtype=torch.uint8, device=device)
    lowres = torch.argmax(logits, dim=0)
    scale = [n_out / n_in for n_in, n_out in zip(old_shape, shape, strict=True)]
    for class_id in torch.unique(lowres).tolist():
        if class_id == 0:
            continue
        found = (lowres == class_id).nonzero()
        low = [max(v - LOGIT_CROP_MARGIN, 0) for v in found.min(dim=0).values.tolist()]
        high = [
            min(v + 1 + LOGIT_CROP_MARGIN, n)
            for v, n in zip(found.max(dim=0).values.tolist(), old_shape, strict=True)
        ]
        start = [max(math.floor(v * s), 0) for v, s in zip(low, scale, strict=True)]
        stop = [
            min(math.ceil(v * s), n) for v, s, n in zip(high, scale, shape, strict=True)
        ]
        coords = [
            (torch.arange(a, b, dtype=torch.float64, device=device) + 0.5)
            * (n_in / n_out)
            - 0.5
            - offset
            for a, b, n_in, n_out, offset in zip(
                start, stop, old_shape, shape, low, strict=True
            )
        ]
        crop = logits[class_id][tuple(map(slice, low, high))].to(torch.float64)
        score = resample_separable(crop, coords, order=1)
        window = tuple(map(slice, start, stop))
        best_window = best[window]
        better = score > best_window.to(torch.float64)
        labels[window][better] = class_id
        best_window[better] = score[better].to(best.dtype)
    return labels


def soft_label_resample(
    labels: torch.Tensor, coords: list[torch.Tensor]
) -> torch.Tensor:
    """Resample a label map by linear interpolation of each class indicator.

    Each label present in ``labels`` becomes a float32 indicator volume, which
    is resampled linearly at ``coords``; every output voxel takes the label
    with the largest value. Labels are visited in increasing order and a later
    label must be strictly larger to win, so the lower label (background
    first) wins an exact tie.
    """
    best = None
    out = None
    for value in torch.unique(labels).tolist():
        score = resample_separable((labels == value).to(torch.float32), coords, 1)
        if best is None:
            best = score
            out = torch.full_like(score, value, dtype=torch.uint8)
        else:
            better = score > best
            best = torch.where(better, score, best)
            out[better] = value
    return out


# ---------------------------------------------------------------------------
# Images and orientation
# ---------------------------------------------------------------------------


def _float_image(ct: Volume) -> nib.Nifti1Image:
    """Return the CT as a float64 NIfTI image with a header built from its affine."""
    return nib.Nifti1Image(np.asarray(ct.array, dtype=np.float64), ct.affine)


def _from_canonical(array: np.ndarray, original_affine: np.ndarray) -> np.ndarray:
    """Reorient an array from the closest RAS orientation to that of an affine."""
    ornt = ornt_transform(axcodes2ornt("RAS"), io_orientation(original_affine))
    return apply_orientation(array, ornt)


def _spacing(image: nib.Nifti1Image) -> np.ndarray:
    """Return the voxel size stored in the header (float32 values) as float64."""
    return np.asarray(image.header.get_zooms()[:3], dtype=np.float64)


# ---------------------------------------------------------------------------
# The segmenter
# ---------------------------------------------------------------------------


def _load_model(predictor: nnUNetPredictor, folder: Path) -> nnUNetPredictor:
    """Read the weights of one model and move its network to the GPU."""
    if not folder.is_dir():
        raise FileNotFoundError(
            f"TotalSegmentator model not found: {folder}. "
            "Run `regrep download-weights` first."
        )
    predictor.initialize_from_trained_model_folder(
        str(folder), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    predictor.network.to(predictor.device).eval()
    labels = predictor.label_manager
    if labels.has_regions or list(labels.all_labels) != list(
        range(len(labels.all_labels))
    ):
        raise RuntimeError(f"{folder}: expected plain labels 0..C-1")
    return predictor


class _CancelledError(Exception):
    """Raised inside a worker thread after the pair has been abandoned."""


class OrganSegmenter:
    """The TotalSegmentator models on one GPU, loaded once.

    The models load in a background thread, in the order in which they are
    first needed, so that the first job can start before all of them are
    ready. The weights are read from TotalSegmentator's weights folder (see
    ``regrep.weights.totalseg_model_folder``).

    Parameters
    ----------
    device
        The CUDA device.
    cfg
        The segmentation settings.
    """

    def __init__(self, device: torch.device | str, cfg: OrganSegmentationConfig):
        self.device = torch.device(device)
        self.cfg = cfg
        self._gpu_lock = threading.Lock()
        self._part_luts = {
            task: _part_lut(task) for task in (*RIB_VERTEBRA_TASKS, *FULL_TASKS)
        }
        _check_part_classes(self._part_luts)
        folders = {task: totalseg_model_folder(task) for task in TOTALSEG_TASKS}
        # nnUNetPredictor turns on cudnn.benchmark when it is created.
        with cudnn_flags(benchmark=True, deterministic=False):
            predictors = {
                task: nnUNetPredictor(
                    tile_step_size=self.cfg.tile_step,
                    use_gaussian=True,
                    use_mirroring=False,
                    perform_everything_on_device=True,
                    device=self.device,
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=False,
                )
                for task in TOTALSEG_TASKS
            }
        loader = ThreadPoolExecutor(max_workers=1)
        self._models: dict[int, Future] = {
            task: loader.submit(_load_model, predictors[task], folders[task])
            for task in TOTALSEG_TASKS
        }
        loader.shutdown(wait=False)

    def release(self) -> None:
        """Free the models and their GPU memory.

        The background loading is waited for first, so that no model is
        moved to the GPU after this call. A loading error is not raised
        here; the job that needed the model has raised it already.
        """
        for future in self._models.values():
            future.exception()
        self._models = {}
        torch.cuda.empty_cache()

    def segment_pair(
        self,
        fixed_ct: Volume,
        moving_ct: Volume,
        emit: Callable[[str, np.ndarray], None],
    ) -> None:
        """Segment both CTs of a pair and hand each map to ``emit`` when it is ready.

        The moving CT and the fixed CT are processed in two threads. One
        thread prepares its next input on the CPU while the other uses the
        GPU; the GPU runs one job at a time. ``emit(name, array)`` is called
        from the calling thread for each map of ``ORGAN_MAPS`` as soon as it
        is ready. Each array is uint8 in ``(X, Y, Z)`` order on the grid of
        its CT.

        The cuDNN flags of the caller are restored on return. Any failure,
        in a thread or in ``emit``, raises here after both threads have
        stopped.
        """
        results: queue.Queue = queue.Queue()
        abandoned = threading.Event()

        def deliver(name: str, array: np.ndarray) -> None:
            if abandoned.is_set():
                raise _CancelledError
            results.put((name, np.ascontiguousarray(array, dtype=np.uint8)))

        def moving_side() -> None:
            image = _float_image(moving_ct)
            deliver(
                "organs6_moving", self._resampled_logits_map(image, ORGANS_6MM_TASK)
            )
            deliver(
                "organs3_moving", self._resampled_logits_map(image, ORGANS_3MM_TASK)
            )
            deliver("labels_moving", self._label_map(image))

        def fixed_side() -> None:
            image = _float_image(fixed_ct)
            deliver("labels_fixed", self._label_map(image))

        def run(side: Callable[[], None]) -> None:
            try:
                side()
            except _CancelledError:
                pass
            except BaseException as error:
                results.put((None, error))

        threads = [
            threading.Thread(target=run, args=(side,), daemon=True)
            for side in (moving_side, fixed_side)
        ]
        with cudnn_flags(benchmark=True, deterministic=False):
            for thread in threads:
                thread.start()
            try:
                for _ in ORGAN_MAPS:
                    name, value = results.get()
                    if name is None:
                        raise value
                    emit(name, value)
            finally:
                abandoned.set()
                for thread in threads:
                    thread.join()

    # -- jobs ---------------------------------------------------------------

    def _resampled_logits_map(self, original: nib.Nifti1Image,
                              task_id: int) -> np.ndarray:
        """Run a low-resolution model and resample its logits to the CT grid.

        ``original`` is the CT from :func:`_float_image`. nnU-Net's
        preprocessing resamples it to the model grid; the logits come back
        with ``resample_logits_to_labels``.
        """
        image = nib.as_closest_canonical(original)
        labels = self._predict(task_id, image, resample_logits=True)
        return _from_canonical(labels, original.affine)

    def _coarse_map(self, original: nib.Nifti1Image) -> np.ndarray:
        """Return the 6 mm map that sets the crop box of the 1.5 mm models.

        The CT is resampled linearly to 6 mm, segmented at 6 mm, and the labels
        come back to the CT grid by nearest-neighbour resampling, as in
        TotalSegmentator's ``roi_subset`` crop.
        """
        image = nib.as_closest_canonical(original)
        spacing = totalseg_task_config(ORGANS_6MM_TASK)["resample"]
        resampled = self._zoom_image(image, spacing)
        labels = self._predict(ORGANS_6MM_TASK, resampled, resample_logits=False)
        with self._gpu_lock, torch.inference_mode():
            volume = torch.from_numpy(labels).to(self.device)
            coords = [
                corner_aligned_coords(n_in, n_out, self.device)
                for n_in, n_out in zip(labels.shape, image.shape, strict=True)
            ]
            back = resample_separable(volume, coords, order=0).cpu().numpy()
            del volume
            torch.cuda.empty_cache()
        return _from_canonical(back, original.affine)

    def _label_map(self, original: nib.Nifti1Image) -> np.ndarray:
        """Return the 117-class map of one CT at the CT grid.

        ``original`` is the CT from :func:`_float_image`. When the 6 mm map
        holds no rib or vertebra (a scan without spine or ribs), the rib and
        vertebra classes stay empty.
        """
        coarse = self._coarse_map(original)
        if not coarse.any():
            raise RuntimeError("the 6 mm organ map of the CT is empty")
        ribs_and_spine = rib_vertebra_lut()[coarse]
        if ribs_and_spine.any():
            rib_vertebra = self._part_map(
                original, ribs_and_spine, RIB_VERTEBRA_TASKS, soft=True
            )
        else:
            rib_vertebra = np.zeros(original.shape, dtype=np.uint8)
        full = self._part_map(original, coarse > 0, FULL_TASKS, soft=False)
        return merge_label_maps(full, rib_vertebra)

    def _part_map(
        self,
        original: nib.Nifti1Image,
        crop_mask: np.ndarray,
        tasks: tuple[int, ...],
        soft: bool,
    ) -> np.ndarray:
        """Run part models at 1.5 mm on the CT cropped to a mask.

        The crop box is the bounding box of ``crop_mask`` widened by
        ``cfg.crop_margin_mm`` on each side. The CT inside it is truncated
        to integers, as TotalSegmentator does, and resampled linearly to the
        1.5 mm grid. The part maps are converted to ``total`` class ids and
        combined, a later task overwriting an earlier one. The result returns
        to the CT grid through ``soft_label_resample`` when ``soft`` is True,
        and through nearest-neighbour resampling otherwise. ``crop_mask``
        must hold at least one voxel.
        """
        mask_image = nib.Nifti1Image(crop_mask.astype(np.uint8), original.affine)
        margin = [self.cfg.crop_margin_mm] * 3
        cropped, bbox = crop_to_mask(original, mask_image, addon=margin, dtype=np.int32)
        image = nib.as_closest_canonical(cropped)
        resampled = self._zoom_image(image, totalseg_task_config(tasks[0])["resample"])

        combined = np.zeros(resampled.shape, dtype=np.uint8)
        for task_id in tasks:
            part = self._predict(task_id, resampled, resample_logits=False)
            mapped = self._part_luts[task_id][part]
            np.copyto(combined, mapped, where=mapped != 0)

        with self._gpu_lock, torch.inference_mode():
            volume = torch.from_numpy(combined).to(self.device)
            coords = [
                corner_aligned_coords(n_in, n_out, self.device)
                for n_in, n_out in zip(combined.shape, image.shape, strict=True)
            ]
            if soft:
                back = soft_label_resample(volume, coords)
            else:
                back = resample_separable(volume, coords, order=0)
            back = back.cpu().numpy()
            del volume
            torch.cuda.empty_cache()

        out = np.zeros(original.shape, dtype=np.uint8)
        box = tuple(slice(lo, hi) for lo, hi in bbox)
        out[box] = _from_canonical(back, cropped.affine)
        return out

    # -- shared steps -------------------------------------------------------

    def _zoom_image(self, image: nib.Nifti1Image, spacing: float) -> nib.Nifti1Image:
        """Resample an image linearly to an isotropic spacing and truncate to int32.

        The affine is scaled column by column, as TotalSegmentator's
        ``change_spacing`` does.
        """
        zoom = _spacing(image) / np.array([spacing] * 3)
        with self._gpu_lock, torch.inference_mode():
            # A reoriented image can be a view with negative strides.
            volume = torch.from_numpy(np.ascontiguousarray(image.get_fdata()))
            volume = volume.to(self.device)
            data = zoom_linear(volume, zoom).cpu().numpy().astype(np.int32)
            del volume
            torch.cuda.empty_cache()
        affine = image.affine.copy()
        affine[:3, :3] = affine[:3, :3] / zoom
        return nib.Nifti1Image(data, affine)

    def _predict(
        self, task_id: int, image: nib.Nifti1Image, resample_logits: bool
    ) -> np.ndarray:
        """Run one model on an image in RAS orientation.

        nnU-Net's preprocessing (crop to the non-zero region, CT
        normalization, resampling to the model grid) runs on the CPU in the
        calling thread. The sliding-window inference and the export to labels
        run on the GPU while the thread holds the GPU lock.

        Returns
        -------
        np.ndarray
            uint8 labels on the grid of ``image``, in its array order. With
            ``resample_logits`` the logits are resampled to the image grid
            with ``resample_logits_to_labels``; otherwise the model grid must
            equal the image grid.
        """
        predictor = self._models[task_id].result()
        plans = predictor.plans_manager
        configuration = predictor.configuration_manager
        # nnU-Net's NIfTI reader returns (1, z, y, x) float32 data and the
        # spacing in the same reversed order.
        data = np.ascontiguousarray(
            image.get_fdata().transpose((2, 1, 0))[None], dtype=np.float32
        )
        properties = {"spacing": [float(s) for s in image.header.get_zooms()[:3][::-1]]}
        data = DefaultPreprocessor(verbose=False).run_case_npy(
            data, None, properties, plans, configuration, predictor.dataset_json
        )[0]
        target_shape = tuple(properties["shape_after_cropping_and_before_resampling"])

        with self._gpu_lock, torch.inference_mode():
            logits = predictor.predict_sliding_window_return_logits(
                torch.from_numpy(data).to(self.device)
            )
            if resample_logits:
                image_spacing = [
                    properties["spacing"][i] for i in plans.transpose_forward
                ]
                separate_z, axis = determine_do_sep_z_and_axis(
                    None, configuration.spacing, image_spacing
                )
                labels = resample_logits_to_labels(
                    logits, target_shape, axis if separate_z else None
                )
            else:
                if tuple(logits.shape[1:]) != target_shape:
                    raise RuntimeError(
                        f"task {task_id}: model grid {tuple(logits.shape[1:])} "
                        f"differs from image grid {target_shape}"
                    )
                labels = torch.argmax(logits, dim=0).to(torch.uint8)
            del logits
            labels = labels.cpu().numpy()
            torch.cuda.empty_cache()

        canvas = np.zeros(properties["shape_before_cropping"], dtype=np.uint8)
        canvas = insert_crop_into_image(
            canvas, labels, properties["bbox_used_for_cropping"]
        )
        return canvas.transpose(plans.transpose_backward).transpose((2, 1, 0))


def _part_lut(task_id: int) -> np.ndarray:
    """Return the lookup table of a part model.

    It maps the class index of the part model to its ``total`` class id.
    """
    names = class_map_5_parts[map_taskid_to_partname_ct[task_id]]
    ids = class_ids_by_name()
    lut = np.zeros(max(names) + 1, dtype=np.uint8)
    for index, name in names.items():
        lut[index] = ids[name]
    return lut


def _check_part_classes(luts: dict[int, np.ndarray]) -> None:
    """Check how the part models split the 117 classes.

    The rib and vertebra parts and the other parts must split them as
    ``RIB_VERTEBRA_IDS`` says.
    """
    rib_vertebra = {int(i) for t in RIB_VERTEBRA_TASKS for i in luts[t] if i}
    full = {int(i) for t in FULL_TASKS for i in luts[t] if i}
    if rib_vertebra != set(RIB_VERTEBRA_IDS) or full != set(load_class_map()) - set(
        RIB_VERTEBRA_IDS
    ):
        raise RuntimeError("the part models do not split the total classes as expected")

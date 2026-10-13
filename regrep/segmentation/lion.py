# This file adapts code from two projects:
#
# LION (lionz), https://github.com/ENHANCE-PET/LION, by Lalith Kumar Shiyam
# Sundar, Sebastian Gutschmayer and Manuel Pires. The LION repository states
# no copyright line.
#
# nnU-Net, https://github.com/MIC-DKFZ/nnUNet. The LICENSE file of nnU-Net
# states:
#
#     Copyright [2019] [Division of Medical Image Computing, German Cancer
#     Research Center (DKFZ), Heidelberg, Germany]
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
# Changes made for regrep: the steps of LION's lion_segment are called
# directly for one PET volume at a time, the PET is resampled in one piece
# instead of in chunks, the nnU-Net export runs in this process, and the
# mirror loop of nnU-Net's nnUNetPredictor runs a fixed subset of its passes.
"""PET lesion segmentation with LION.

``LionSegmenter`` loads the LION PSMA model once and segments any number of
PET volumes. Each volume is resampled to the model's 3 mm grid with a cubic
B-spline, segmented by nnU-Net with a subset of its mirror passes, and the
lesion class is resampled back to the PET grid with nearest-neighbour
interpolation.
"""

from __future__ import annotations

import json
import os

import numpy as np
import SimpleITK
import torch
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from regrep.config import LesionConfig
from regrep.io import Volume
from regrep.segmentation import cudnn_flags
from regrep.weights import lion_model_folder, lion_models_directory


class MirrorSubsetPredictor(nnUNetPredictor):
    """nnU-Net predictor that averages a fixed list of mirror passes.

    nnU-Net averages the network output over the unflipped patch and every
    combination of flips over the allowed axes. This predictor runs the
    unflipped patch and then the flips in ``flips``, in that order, and
    divides the sum by the number of passes.

    Parameters
    ----------
    flips
        Flips over the axes of the 5-D input patch, for example
        ``((2,), (3,), (2, 3))``.
    """

    def __init__(self, flips: tuple[tuple[int, ...], ...], **kwargs):
        super().__init__(**kwargs)
        self.flips = flips

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.network(x)
        for axes in self.flips:
            prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
        prediction /= len(self.flips) + 1
        return prediction


def resample_bspline(image: SimpleITK.Image, spacing) -> np.ndarray:
    """Resample an image to another voxel size with a cubic B-spline.

    The new grid has the origin and the direction of the image, and
    ``round(n * s / t)`` voxels along an axis of ``n`` voxels of size ``s``,
    where ``t`` is the new voxel size. Points outside the image take the
    value 0. This is the resampling of lionz's
    ``ImageResampler.resample_image_SimpleITK_DASK_array`` on an image that it
    does not cut into chunks. lionz cuts long axes into chunks and gives each
    chunk its own grid, which shifts the samples after each seam; here the
    whole image is resampled at once.

    Parameters
    ----------
    image
        The image.
    spacing
        The new voxel size, in the axis order of SimpleITK.

    Returns
    -------
    numpy.ndarray
        The float32 array of the resampled image, in the axis order of NumPy.
    """
    size = [round(n * (s / t))
            for n, s, t in zip(image.GetSize(), image.GetSpacing(), spacing,
                               strict=True)]
    resampled = SimpleITK.Resample(
        image, size, SimpleITK.Transform(), SimpleITK.sitkBSpline, image.GetOrigin(),
        tuple(float(t) for t in spacing), image.GetDirection(), 0.0,
        image.GetPixelIDValue())
    return SimpleITK.GetArrayFromImage(resampled)


def _check_model_folder(model_name: str) -> None:
    """Raise if lionz would delete an existing model folder.

    lionz removes a model folder and downloads it again when its
    ``model_version.json`` names a different URL than the installed lionz.
    This check stops before that happens, so a shared or read-only weights
    folder is never changed.
    """
    from lionz import constants, models

    metadata = models.MODEL_METADATA[model_name]
    directory = lion_model_folder(model_name)
    if not os.path.exists(directory):
        return
    version_file = os.path.join(directory, "model_version.json")
    try:
        with open(version_file) as handle:
            url = json.load(handle).get("url")
    except (OSError, ValueError):
        url = None
    if url != metadata[constants.KEY_URL]:
        raise RuntimeError(
            f"{directory} does not hold the LION weights that this lionz expects "
            f"({metadata[constants.KEY_URL]}); remove the folder or set "
            "LIONZ_MODELS_DIRECTORY to a different one"
        )


def download_weights(cfg: LesionConfig) -> str:
    """Download the LION model into the models folder if it is missing.

    Returns the folder. An existing folder whose weights do not match the
    installed lionz raises instead of being replaced (see
    ``_check_model_folder``).
    """
    from lionz import models, system

    directory = lion_models_directory()
    _check_model_folder(cfg.lion_model)
    models.Model(cfg.lion_model, system.OutputManager(False, False),
                 base_directory=directory)
    return directory


class LionSegmenter:
    """LION lesion segmentation of PET volumes with one loaded model.

    The weights are read from the LION models folder
    (``regrep.weights.lion_models_directory``); lionz downloads them into it
    if they are missing.

    Parameters
    ----------
    cfg
        The lesion settings (model id, mirror passes, CPU threads).
    device
        The CUDA device the network runs on.
    """

    def __init__(self, cfg: LesionConfig, device: str | torch.device):
        from lionz import models, system

        _check_model_folder(cfg.lion_model)
        self.cfg = cfg
        self.model = models.Model(cfg.lion_model,
                                  system.OutputManager(False, False),
                                  base_directory=lion_models_directory())
        # nnU-Net turns on cudnn.benchmark when it builds a predictor.
        with cudnn_flags(benchmark=True, deterministic=False):
            self.predictor = MirrorSubsetPredictor(
                cfg.lion_mirror_flips, allow_tqdm=False,
                device=torch.device(device))
            self.predictor.initialize_from_trained_model_folder(
                self.model.configuration_directory, use_folds=("all",))

    def segment(self, pet: Volume) -> np.ndarray:
        """Return the LION lesion mask of one PET volume.

        Parameters
        ----------
        pet
            A PET volume in SUV, as stored on disk (not clipped).

        Returns
        -------
        numpy.ndarray
            ``(X, Y, Z)`` uint8 mask with values 0 and 1 on the PET grid.
        """
        from lionz import image_processing, predict

        spacing = self.model.voxel_spacing
        # Both resamplings below put voxel 0 of the 3 mm grid on voxel 0 of the
        # PET grid, so only the PET spacing matters, not its origin or direction.
        image = SimpleITK.GetImageFromArray(
            np.ascontiguousarray(np.asarray(pet.array, np.float32).transpose(2, 1, 0)))
        image.SetSpacing(pet.spacing)
        array = resample_bspline(image, spacing)
        case = next(predict.preprocessing_iterator_from_array(
            array[None, ...], {"spacing": spacing}, self.predictor))
        # The export sets the CPU threads of torch and restores them only when
        # it succeeds; leaving this block restores them in every case.
        with cudnn_flags(benchmark=True, deterministic=False,
                         threads=self.cfg.lion_threads):
            logits = self.predictor.predict_logits_from_preprocessed_data(
                case["data"]).cpu().detach().numpy()
            labels = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits, self.predictor.plans_manager,
                self.predictor.configuration_manager, self.predictor.label_manager,
                case["data_properties"], False)
        del logits

        lesion = (labels == self.model.tumor_label).astype(np.uint8)
        coarse = SimpleITK.GetImageFromArray(lesion)
        coarse.SetSpacing(spacing)
        coarse.SetOrigin(image.GetOrigin())
        coarse.SetDirection(image.GetDirection())
        mask = SimpleITK.GetArrayFromImage(
            image_processing.ImageResampler.resample_segmentation(image, coarse))
        return np.ascontiguousarray(mask.transpose(2, 1, 0))

    def release(self) -> None:
        """Free the network and its GPU memory."""
        self.predictor = None
        torch.cuda.empty_cache()

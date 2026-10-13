"""Find and download the network weights that the chain needs.

There are three sets of weights:

* anatomix (the variant in ``FeatureConfig.backbone``), downloaded from the
  HuggingFace Hub into the HuggingFace cache (``HF_HOME``). It is loaded with
  anatomix's ``load_from_hf``, which uses the cached file when the Hub cannot
  be reached.
* The seven TotalSegmentator models of the ``total`` task, in
  TotalSegmentator's weights folder (``TOTALSEG_WEIGHTS_PATH``, else
  ``TOTALSEG_HOME_DIR/nnunet/results``, else
  ``~/.totalsegmentator/nnunet/results``).
* The LION PSMA model, in lionz's models folder (``LIONZ_MODELS_DIRECTORY``,
  else the folder inside lionz's Python environment).

:func:`download_all` fetches whatever is missing; it is meant to run once,
before the machine goes offline. :func:`missing` lists what is not there.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from regrep.config import ChainConfig

log = logging.getLogger(__name__)


def anatomix_file(config: ChainConfig) -> str | None:
    """The cached anatomix checkpoint, or ``None`` if it is not cached."""
    from anatomix.model.load_from_hf import DEFAULT_REPO
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(DEFAULT_REPO, f"{config.features.backbone}.pth")
    return path if isinstance(path, str) else None


def download_anatomix(config: ChainConfig) -> str:
    """Download the anatomix checkpoint into the HuggingFace cache."""
    from anatomix.model.load_from_hf import DEFAULT_REPO
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DEFAULT_REPO, f"{config.features.backbone}.pth")


TOTALSEG_TASKS = (298, 297, 292, 295, 291, 293, 294)
"""The seven models of the TotalSegmentator ``total`` task that the chain runs."""


def totalseg_task_config(task_id: int) -> dict:
    """Return TotalSegmentator's settings for one model of the ``total`` task.

    The result holds ``trainer``, ``plans``, ``model``, ``folds`` and
    ``resample`` (the grid spacing of the model in mm).
    """
    from totalsegmentator.map_tasks_config import DEFAULT_CONFIG, TASK_CONFIGS

    for mode in TASK_CONFIGS["total"]["sub_modes"].values():
        ids = mode["task_id"]
        if task_id == ids or (isinstance(ids, list) and task_id in ids):
            return {**DEFAULT_CONFIG, **mode}
    raise KeyError(f"task {task_id} is not a model of the TotalSegmentator total task")


def totalseg_model_folder(task_id: int) -> Path:
    """Return the nnU-Net output folder of one TotalSegmentator model.

    ``task_id`` is a task id of the ``total`` task (291 to 298). The folder
    is in TotalSegmentator's weights folder: ``$TOTALSEG_WEIGHTS_PATH``, else
    ``$TOTALSEG_HOME_DIR/nnunet/results``, else
    ``~/.totalsegmentator/nnunet/results``.
    """
    from totalsegmentator.config import get_weights_dir
    from totalsegmentator.map_tasks_config import TASK_ID_WEIGHTS_CONFIGS

    config = totalseg_task_config(task_id)
    root = get_weights_dir()
    plans = config["plans"] or "nnUNetPlans"
    name = f"{config['trainer']}__{plans}__{config['model']}"
    return root / TASK_ID_WEIGHTS_CONFIGS[task_id]["foldername"] / name


def missing_totalsegmentator() -> list[str]:
    """The TotalSegmentator model folders that do not hold a checkpoint."""
    missing = []
    for task in TOTALSEG_TASKS:
        folder = totalseg_model_folder(task)
        if not (folder / "fold_0" / "checkpoint_final.pth").exists():
            missing.append(str(folder))
    return missing


def lion_models_directory() -> str:
    """Return the LION models folder.

    This is ``LIONZ_MODELS_DIRECTORY`` if it is set, else lionz's default
    folder inside its Python environment.
    """
    if os.environ.get("LIONZ_MODELS_DIRECTORY"):
        return os.environ["LIONZ_MODELS_DIRECTORY"]
    from lionz import system

    return system.MODELS_DIRECTORY_PATH


def lion_model_folder(model_name: str) -> str:
    """Return the folder of one LION model inside the LION models folder."""
    from lionz import constants, models

    metadata = models.MODEL_METADATA[model_name]
    return os.path.join(lion_models_directory(), metadata[constants.KEY_FOLDER_NAME])


def lion_model_present(config: ChainConfig) -> bool:
    """Whether the LION model of ``config`` is in the models folder."""
    folder = lion_model_folder(config.lesions.lion_model)
    return os.path.exists(os.path.join(folder, "model_version.json"))


def missing(config: ChainConfig) -> list[str]:
    """Describe each set of weights that is not available locally.

    This imports only the configuration modules of the segmentors, not the
    networks, so it is fast.
    """
    problems = []
    if anatomix_file(config) is None:
        problems.append(f"anatomix {config.features.backbone} is not in the "
                        "HuggingFace cache")
    for folder in missing_totalsegmentator():
        problems.append(f"TotalSegmentator model missing: {folder}")
    if not lion_model_present(config):
        problems.append(f"LION model {config.lesions.lion_model!r} missing in "
                        f"{lion_models_directory()}")
    return problems


def download_all(config: ChainConfig) -> None:
    """Download every set of weights that is missing."""
    from regrep.segmentation import lion, totalseg

    log.info("anatomix: %s", download_anatomix(config))
    totalseg.download_weights()
    log.info("TotalSegmentator: %s",
             totalseg_model_folder(TOTALSEG_TASKS[0]).parent.parent)
    log.info("LION: %s", lion.download_weights(config.lesions))

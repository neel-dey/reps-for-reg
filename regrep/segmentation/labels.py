"""Class ids of the TotalSegmentator ``total`` task and the merge of its maps.

The ``total`` task labels 117 structures with ids 1 to 117; 0 is background.
regrep reads the id-to-name table from ``regrep/data/totalseg_class_map.json``,
which is a copy of ``totalsegmentator.map_to_binary.class_map["total"]``.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

CLASS_MAP_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "totalseg_class_map.json"
)

NUM_CLASSES = 117
"""The number of structures of the ``total`` task (ids 1 to 117)."""

ORGAN_MAPS = ("organs6_moving", "organs3_moving", "labels_fixed", "labels_moving")
"""The maps of the organ segmentation.

``organs6_moving`` and ``organs3_moving`` are the 6 mm and 3 mm maps of the
moving CT; ``labels_fixed`` and ``labels_moving`` are the 117-class maps of
both CTs.
"""

RIB_VERTEBRA_IDS: tuple[int, ...] = tuple(range(25, 51)) + tuple(range(92, 118))
"""The 52 classes of the vertebra and rib models (tasks 292 and 295).

They are the sacrum, the vertebrae S1 to C1, the 24 ribs, the sternum and the
costal cartilages. The other 65 classes come from tasks 291, 293 and 294.
"""


@functools.cache
def load_class_map() -> dict[int, str]:
    """Return the ``total`` class map as ``{id: name}`` for ids 1 to 117."""
    with open(CLASS_MAP_FILE) as handle:
        return {int(key): name for key, name in json.load(handle).items()}


def class_ids_by_name() -> dict[str, int]:
    """Return the inverse of ``load_class_map``, ``{name: id}``."""
    return {name: key for key, name in load_class_map().items()}


def rib_vertebra_lut() -> np.ndarray:
    """Return a boolean lookup table over uint8 values.

    The table is True on the rib and vertebra ids.
    """
    lut = np.zeros(256, dtype=bool)
    lut[list(RIB_VERTEBRA_IDS)] = True
    return lut


def merge_label_maps(full: np.ndarray, rib_vertebra: np.ndarray) -> np.ndarray:
    """Paint the rib and vertebra map over the map of the other classes.

    Parameters
    ----------
    full
        uint8 map of the 65 classes of tasks 291, 293 and 294.
    rib_vertebra
        uint8 map of the 52 classes of tasks 292 and 295, on the same grid.

    Returns
    -------
    np.ndarray
        uint8 map with all 117 classes. A rib or vertebra label wins every
        voxel where it is present; elsewhere the voxel keeps its ``full``
        label, unless that label is a rib or vertebra id, which becomes 0.
    """
    if full.shape != rib_vertebra.shape:
        raise ValueError(f"shape mismatch: {full.shape} and {rib_vertebra.shape}")
    merged = np.asarray(full, dtype=np.uint8).copy()
    painted = np.asarray(rib_vertebra, dtype=np.uint8)
    merged[rib_vertebra_lut()[merged]] = 0
    present = painted > 0
    merged[present] = painted[present]
    return merged

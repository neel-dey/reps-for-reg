"""Hyperparameters of the registration chain.

The settings that a user can tune live here. The defaults are the settings
of the method. ``load_config`` can override any of them from a YAML or JSON
file. A ``ChainConfig`` checks all its values when it is created and raises
a ``ValueError`` that names the first invalid setting.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


# --- working grid
@dataclass(frozen=True)
class GridConfig:
    """The voxel size to which much finer scans are resampled.

    Many settings below are in voxels and were tuned on scans with this voxel
    size. On much finer grids the method is less accurate and needs much
    more memory, so the CT and PET of a timepoint are resampled before all
    other steps along every axis that is much finer than this voxel size
    (``regrep.io.working_grid`` gives the grid rule). The field is returned
    on the grid of the fixed CT file.
    """

    # Working voxel size in mm along the left-right, anterior-posterior and
    # superior-inferior directions. Each array axis takes the value of the
    # direction it points along most closely.
    spacing_mm: tuple[float, float, float] = (2.7344, 2.7344, 3.27)
    # An array axis is resampled to its working voxel size only when its voxel
    # size is below this fraction of it. All other axes keep their voxels, and
    # a timepoint without such an axis is used as it is.
    resample_below: float = 0.5


# --- organ segmentation
@dataclass(frozen=True)
class OrganSegmentationConfig:
    """Settings of the TotalSegmentator models."""

    # Step between sliding windows, as a fraction of the patch size. It is
    # used for all seven models.
    tile_step: float = 0.8
    # Margin in mm added on each side of the bounding box of the 6 mm map when
    # the CT is cropped for the 1.5 mm part models.
    crop_margin_mm: float = 20.0


# --- lesion segmentation and lesion supports
@dataclass(frozen=True)
class LesionConfig:
    """Settings of LION and of the lesion supports."""

    # LION model id; "psma" is the PSMA PET lesion model (Dataset711_PSMA).
    lion_model: str = "psma"
    # Mirror passes of LION's test-time augmentation, as flips over the tensor
    # axes (2, 3, 4) = (z, y, x) of the 3 mm patch. The unflipped pass always
    # runs first; these five follow it in this order.
    lion_mirror_flips: tuple[tuple[int, ...], ...] = (
        (2,), (3,), (4,), (2, 3), (2, 4),
    )
    # CPU threads of torch while LION runs.
    lion_threads: int = 3
    # A voxel can be a lesion when its PET value (SUV) is above this threshold.
    suv_threshold: float = 3.0
    # Half-width in mm of the box by which the physiological organs are dilated
    # before they are removed from the thresholded PET.
    organ_halo_mm: float = 10.0
    # TotalSegmentator classes with physiological PSMA uptake.
    physiological_organs: tuple[str, ...] = (
        "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
        "stomach", "small_bowel", "duodenum", "colon", "urinary_bladder",
        "kidney_cyst_left", "kidney_cyst_right", "heart", "brain", "pancreas",
        "esophagus",
    )


# --- features
@dataclass(frozen=True)
class FeatureConfig:
    """Settings of the image features."""

    # Intensity window of the CT in HU and of the PET in SUV. Each volume is
    # clipped to its window and then min-max normalized over its own range.
    ct_clip: tuple[float, float] = (-1024.0, 1024.0)
    pet_clip: tuple[float, float] = (0.0, 15.0)
    # Body mask: CT threshold in HU, erosion depth (voxels) before the largest
    # component is taken, and closing radius (voxels) at the end.
    body_threshold_hu: float = -500.0
    body_open_voxels: int = 1
    body_close_voxels: int = 2
    # anatomix backbone on the HuggingFace Hub, and the CPU thread count used
    # while the model is built (the previous count is restored afterwards).
    backbone: str = "anatomix-dev-vit"
    backbone_init_threads: int = 1
    # Isotropic spacings (mm) of the feature pyramid. None stands for the
    # smallest voxel size of the fixed CT, which both timepoints use.
    pyramid_mm: tuple[float | None, ...] = (1.0, 1.75, None)
    # Sliding window: cube size, overlap fraction, windows per forward pass,
    # and the Gaussian blending sigma as a fraction of the window size.
    window: int = 128
    window_overlap: float = 0.25
    window_batch: int = 16
    window_sigma: float = 0.25
    # Windows with no body voxel within this many voxels are skipped.
    window_margin: int = 16
    # MIND-SSC: isotropic spacing (mm) it is computed on, patch radius and
    # neighbour dilation (voxels), and the voxel budget of one slab.
    mind_spacing_mm: float = 1.0
    mind_radius: int = 1
    mind_dilation: int = 2
    mind_slab_voxels: int = 40_000_000


# --- stage 1
@dataclass(frozen=True)
class Stage1Config:
    """Settings of the first deformable stage."""

    # Downsampling factor of each pyramid level, coarse to fine.
    scales: tuple[int, ...] = (6, 4, 2, 1)
    # Largest number of iterations at each level.
    iterations: tuple[int, ...] = (100, 100, 100, 30)
    # Box width of the local cross-correlation window at each level.
    cc_kernel_sizes: tuple[int, ...] = (9, 7, 5, 3)
    # Adam step size, in half-voxels of the longest axis of the level.
    learning_rate: float = 2.0
    # Gaussian sigma, in voxels of the level, applied to the gradient.
    grad_sigma: float = 2.0
    # Gaussian sigma, in voxels of the level, applied to the warp after a step.
    warp_sigma: float = 0.5
    # Multiplier of the volume penalty on the lesion support.
    volume_penalty: float = 10.0
    # Early stop: a level stops when the slope of its last tolerance_window
    # losses is above tolerance.
    tolerance: float = 1e-6
    tolerance_window: int = 10


# --- stage 2
@dataclass(frozen=True)
class Stage2Config:
    """Settings of the second deformable stage."""

    # Distance at which the structure distance channels saturate, in mm.
    band_mm: float = 20.0
    # Width of the soft structure membership, in multiples of the smallest
    # voxel size of the fixed CT.
    shell_voxels: float = 2.0
    # Dice weight of a structure is its mean voxel count to this negative power.
    dice_size_exponent: float = 2.0
    # Dice and chamfer weight of the lesion channel. The organ weights are
    # normalized to a mean of 1, so 0.25 is a quarter of an average organ.
    lesion_channel_weight: float = 0.25
    # Values of the terms at the identity. The region term starts at
    # region_share, the chamfer term at chamfer_share and the Dice term at
    # 1 - region_share - chamfer_share. The anatomix LNCC term, which is
    # negative, starts at -amix_share, in addition to these.
    region_share: float = 0.25
    chamfer_share: float = 0.15
    amix_share: float = 0.5
    # Largest number of fixed-side points per structure in the region term.
    region_points: int = 4000
    # Seed of the private generator that subsamples those points.
    region_seed: int = 12345
    # Multiplier of the volume penalty on the lesion support.
    volume_penalty: float = 10.0
    # Largest number of iterations.
    iterations: int = 35
    # Adam step size, in half-voxels of the longest axis.
    learning_rate: float = 2.0
    # Gaussian sigmas, in voxels, applied to the gradient and to the warp
    # after a step.
    grad_sigma: float = 1.0
    warp_sigma: float = 0.5
    # Early stop: the stage stops when the slope of its last
    # tolerance_window losses is above tolerance.
    tolerance: float = 1e-6
    tolerance_window: int = 10


# --- post-hoc steps
@dataclass(frozen=True)
class PosthocConfig:
    """Settings of the volume projection, the low pass and the fold repair."""

    # Volume projection (used after stage 1 and on the composed transform).
    # Largest number of pointwise passes that drive log det J to 0 on the
    # warped lesion support.
    passes: int = 4
    # Dilation of the warped lesion support, in voxels, before det J is read.
    margin: int = 1
    # Dilation of the support, in voxels, over which a correction fades out.
    envelope: int = 3
    # Bound on each component of a pointwise correction, in voxels.
    cap: float = 0.5
    # Box-filter radius, in voxels, applied twice to each correction.
    smooth: int = 1
    # Lower clamp of det J before the logarithm.
    det_floor: float = 0.05
    # A lesion component gets pointwise passes only if its warped voxel count
    # changes by more than this, as |log(count ratio)|.
    gate: float = 0.03
    # Largest number of passes that drive the warped voxel count of each
    # lesion component to its count before the warp.
    count_passes: int = 8
    # Bound on each component of a count correction, in voxels.
    count_cap: float = 0.5
    # Export: sigma, in voxels, of the Gaussian low pass of the final field.
    export_sigma: float = 1.0
    # Exterior fold repair: maximum number of blur passes.
    repair_passes: int = 6
    # Sigma, in voxels, of the Gaussian blur the repair blends in.
    repair_sigma: float = 2.0
    # Growth, in voxels, of the folded region before it is blurred.
    repair_reach: int = 2
    # Collar around the body, in mm, that the repair never changes.
    repair_margin_mm: float = 5.0


# --- the whole chain
@dataclass(frozen=True)
class ChainConfig:
    """All settings of the chain."""

    grid: GridConfig = field(default_factory=GridConfig)
    organs: OrganSegmentationConfig = field(default_factory=OrganSegmentationConfig)
    lesions: LesionConfig = field(default_factory=LesionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    posthoc: PosthocConfig = field(default_factory=PosthocConfig)
    # Seed of the Python, NumPy and torch generators, set before the features
    # of each pair.
    seed: int = 12345

    def __post_init__(self):
        _validate(self)


# --- checks of the values
def _number(value) -> bool:
    """Whether a value is a finite int or float (not a bool)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _at_least(low: float) -> Callable[[object], bool]:
    return lambda value: _number(value) and value >= low


def _above(low: float) -> Callable[[object], bool]:
    return lambda value: _number(value) and value > low


def _within(low: float, high: float) -> Callable[[object], bool]:
    """Whether a number is in ``(low, high]``."""
    return lambda value: _number(value) and low < value <= high


def _whole_at_least(low: int) -> Callable[[object], bool]:
    return lambda value: _whole(value) and value >= low


def _text(value) -> bool:
    return isinstance(value, str) and bool(value)


def _tuple(item: Callable[[object], bool], length: int | None = None,
           empty: bool = False) -> Callable[[object], bool]:
    """Whether a value is a tuple whose items pass ``item``.

    ``length`` fixes the number of items; otherwise any number of items, at
    least one unless ``empty``, is accepted.
    """
    def check(value) -> bool:
        if not isinstance(value, tuple) or not all(item(v) for v in value):
            return False
        return len(value) == length if length is not None else bool(value) or empty
    return check


def _window(value) -> bool:
    """Whether a value is a pair of numbers ``(low, high)`` with ``low < high``."""
    return _tuple(_number, 2)(value) and value[0] < value[1]


def _flip(value) -> bool:
    """Whether a value is a nonempty tuple of distinct axes among 2, 3 and 4."""
    return (_tuple(lambda axis: axis in (2, 3, 4) and _whole(axis))(value)
            and len(set(value)) == len(value))


_POSITIVE = (_above(0.0), "a positive number")
_NOT_NEGATIVE = (_at_least(0.0), "a number of at least 0")
_COUNT = (_whole_at_least(1), "a whole number of at least 1")
_COUNT_OR_ZERO = (_whole_at_least(0), "a whole number of at least 0")
_ANY_NUMBER = (_number, "a finite number")

# For each section of ChainConfig and each of its settings: a test of the
# value and the text that describes a valid value.
_RULES: dict[str, dict[str, tuple[Callable[[object], bool], str]]] = {
    "grid": {
        "spacing_mm": (_tuple(_above(0.0), 3), "three positive numbers"),
        "resample_below": (_within(0.0, 1.0), "a number in (0, 1]"),
    },
    "organs": {
        "tile_step": (_within(0.0, 1.0), "a number in (0, 1]"),
        "crop_margin_mm": _NOT_NEGATIVE,
    },
    "lesions": {
        "lion_model": (_text, "a model name"),
        "lion_mirror_flips": (_tuple(_flip, empty=True),
                              "a list of flips, each a list of axes among 2, 3, 4"),
        "lion_threads": _COUNT,
        "suv_threshold": _ANY_NUMBER,
        "organ_halo_mm": _NOT_NEGATIVE,
        "physiological_organs": (_tuple(_text, empty=True),
                                 "a list of TotalSegmentator class names"),
    },
    "features": {
        "ct_clip": (_window, "two numbers, the lower one first"),
        "pet_clip": (_window, "two numbers, the lower one first"),
        "body_threshold_hu": _ANY_NUMBER,
        "body_open_voxels": _COUNT,
        "body_close_voxels": _COUNT,
        "backbone": (_text, "a model name"),
        "backbone_init_threads": _COUNT,
        "pyramid_mm": (_tuple(lambda v: v is None or _above(0.0)(v)),
                       "a list of positive numbers or null"),
        "window": _COUNT,
        "window_overlap": (lambda v: _number(v) and 0.0 <= v < 1.0,
                           "a number in [0, 1)"),
        "window_batch": _COUNT,
        "window_sigma": _POSITIVE,
        "window_margin": _COUNT_OR_ZERO,
        "mind_spacing_mm": _POSITIVE,
        "mind_radius": _COUNT_OR_ZERO,
        "mind_dilation": _COUNT,
        "mind_slab_voxels": _COUNT,
    },
    "stage1": {
        "scales": (_tuple(_whole_at_least(1)), "a list of whole numbers of at least 1"),
        "iterations": (_tuple(_whole_at_least(1)),
                       "a list of whole numbers of at least 1"),
        "cc_kernel_sizes": (_tuple(lambda v: _whole_at_least(1)(v) and v % 2 == 1),
                            "a list of odd whole numbers"),
        "learning_rate": _POSITIVE,
        "grad_sigma": _NOT_NEGATIVE,
        "warp_sigma": _NOT_NEGATIVE,
        "volume_penalty": _NOT_NEGATIVE,
        "tolerance": _ANY_NUMBER,
        "tolerance_window": _COUNT,
    },
    "stage2": {
        "band_mm": _POSITIVE,
        "shell_voxels": _POSITIVE,
        "dice_size_exponent": _ANY_NUMBER,
        "lesion_channel_weight": _NOT_NEGATIVE,
        "region_share": (lambda v: _number(v) and 0.0 <= v < 1.0, "a number in [0, 1)"),
        "chamfer_share": (lambda v: _number(v) and 0.0 <= v < 1.0,
                          "a number in [0, 1)"),
        "amix_share": _NOT_NEGATIVE,
        "region_points": _COUNT,
        "region_seed": (_whole, "a whole number"),
        "volume_penalty": _NOT_NEGATIVE,
        "iterations": _COUNT,
        "learning_rate": _POSITIVE,
        "grad_sigma": _NOT_NEGATIVE,
        "warp_sigma": _NOT_NEGATIVE,
        "tolerance": _ANY_NUMBER,
        "tolerance_window": _COUNT,
    },
    "posthoc": {
        "passes": _COUNT_OR_ZERO,
        "margin": _COUNT_OR_ZERO,
        "envelope": _COUNT_OR_ZERO,
        "cap": _POSITIVE,
        "smooth": _COUNT_OR_ZERO,
        "det_floor": _POSITIVE,
        "gate": _NOT_NEGATIVE,
        "count_passes": _COUNT_OR_ZERO,
        "count_cap": _POSITIVE,
        "export_sigma": _POSITIVE,
        "repair_passes": _COUNT_OR_ZERO,
        "repair_sigma": _POSITIVE,
        "repair_reach": _COUNT_OR_ZERO,
        "repair_margin_mm": _NOT_NEGATIVE,
    },
}


def _validate(config: ChainConfig) -> None:
    """Raise a ``ValueError`` that names the first invalid setting of ``config``."""
    for section, rules in _RULES.items():
        values = getattr(config, section)
        for name, (valid, text) in rules.items():
            value = getattr(values, name)
            if not valid(value):
                raise ValueError(f"{section}.{name} must be {text}, got {value!r}")
    if not _whole(config.seed):
        raise ValueError(f"seed must be a whole number, got {config.seed!r}")
    stage1 = config.stage1
    if not len(stage1.scales) == len(stage1.iterations) == len(stage1.cc_kernel_sizes):
        raise ValueError("stage1.scales, stage1.iterations and stage1.cc_kernel_sizes "
                         "must have one entry for each pyramid level")
    if config.stage2.region_share + config.stage2.chamfer_share >= 1.0:
        raise ValueError("stage2.region_share + stage2.chamfer_share must be below 1")
    from regrep.segmentation.labels import class_ids_by_name

    unknown = set(config.lesions.physiological_organs) - set(class_ids_by_name())
    if unknown:
        raise ValueError("lesions.physiological_organs holds names that are not "
                         f"TotalSegmentator classes: {sorted(unknown)}")


def _as_tuple(value):
    """Turn a list (as YAML and JSON give it) into a tuple, at every depth."""
    if isinstance(value, (list, tuple)):
        return tuple(_as_tuple(v) for v in value)
    return value


def _converted(name: str, current, value):
    """Return ``value`` with the type of the default ``current`` of a field.

    A file can give an integer for a float setting; an integer setting must
    get a whole number.
    """
    if isinstance(current, tuple):
        return _as_tuple(value)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return value
    if not _number(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if isinstance(current, int):
        if value != int(value):
            raise ValueError(f"{name} must be a whole number, got {value!r}")
        return int(value)
    return float(value)


def _replace(instance, values, prefix: str = ""):
    """Return a copy of a dataclass with the fields in ``values`` replaced.

    ``prefix`` is the path of ``instance`` in the file, for the messages.
    """
    if not isinstance(values, dict):
        raise ValueError(f"{prefix.rstrip('.') or 'the file'} must be a mapping, "
                         f"got {values!r}")
    changes = {}
    for name, value in values.items():
        if name not in {f.name for f in dataclasses.fields(instance)}:
            raise ValueError(f"{prefix}{name} is not a setting")
        current = getattr(instance, name)
        if dataclasses.is_dataclass(current):
            changes[name] = _replace(current, value, f"{prefix}{name}.")
        else:
            changes[name] = _converted(f"{prefix}{name}", current, value)
    return dataclasses.replace(instance, **changes)


def load_config(path: str | Path | None = None) -> ChainConfig:
    """Return the chain settings, with the overrides of a YAML or JSON file.

    Parameters
    ----------
    path
        A file that holds a nested mapping with the same structure as
        ``ChainConfig``. ``None`` returns the defaults.
    """
    config = ChainConfig()
    if path is None:
        return config
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        values = yaml.safe_load(text) or {}
    else:
        values = json.loads(text)
    return _replace(config, values)

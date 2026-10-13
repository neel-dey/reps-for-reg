"""Reading and writing NIfTI volumes, and the workspace of one pair.

Volume arrays and every file are in nibabel axis order ``(X, Y, Z)``.
"""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import scipy.ndimage
from nibabel.orientations import inv_ornt_aff, io_orientation, ornt_transform

from regrep.config import GridConfig

# Allowance, in working voxels, in the voxel count of a working grid axis: an
# extent that is a whole number of working voxels up to rounding does not gain
# a voxel.
_COUNT_ALLOWANCE = 1e-3

# Largest difference between two affine entries (mm, or direction cosines)
# for which two files count as one grid.
GRID_TOLERANCE = 1e-3

# Fewest voxels along each axis of a working grid. The coarse levels of the
# FireANTs pyramid also have at least this many voxels along each axis.
MIN_AXIS_VOXELS = 32


@dataclass(frozen=True)
class Volume:
    """A volume and the geometry of the file it came from.

    Attributes
    ----------
    array
        Voxel values in ``(X, Y, Z)`` order, with ``scl_slope`` and
        ``scl_inter`` applied.
    affine
        The voxel-to-world matrix of the array.
    header
        The NIfTI header of the file, with the shape and the affine of the
        array.
    path
        The source file, or ``None`` for a volume made in memory.
    file_index
        For a volume whose array was reordered (:func:`orient_like`) or
        resampled (:func:`resample_linear`) after reading, the 4 x 4 map from
        its voxel indices to the voxel indices of the source file. ``None``
        when the array is the array of the file.
    """

    array: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header
    path: str | None = None
    file_index: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.array.shape[:3])

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel size in mm along ``(X, Y, Z)``, from the affine."""
        return voxel_size(self.affine)


@dataclass(frozen=True)
class PairInputs:
    """The four input files of one registration pair."""

    fixed_ct: str
    fixed_pet: str
    moving_ct: str
    moving_pet: str


@dataclass(frozen=True)
class PairOutputs:
    """The files that one registration writes, all in one folder.

    The folder is made when the files are written.
    """

    folder: str

    @property
    def field(self) -> str:
        """The displacement field."""
        return os.path.join(self.folder, "field.nii.gz")

    @property
    def moved_ct(self) -> str:
        """The moving CT on the grid of the fixed CT."""
        return os.path.join(self.folder, "moved_ct.nii.gz")

    @property
    def moved_pet(self) -> str:
        """The moving PET on the grid of the fixed CT."""
        return os.path.join(self.folder, "moved_pet.nii.gz")

    def files(self) -> tuple[str, str, str]:
        """All output files, in the order in which they are written."""
        return (self.moved_ct, self.moved_pet, self.field)

    def exist(self) -> bool:
        """Whether all output files exist."""
        return all(os.path.exists(path) for path in self.files())

    def check(self, inputs) -> None:
        """Raise a ``ValueError`` if the outputs cannot be written safely.

        This is the case when the folder is a file, or when an output file is
        one of the input files (:func:`check_output`).
        """
        if os.path.exists(self.folder) and not os.path.isdir(self.folder):
            raise ValueError(f"the output folder {self.folder} is a file")
        for path in self.files():
            check_output(path, inputs)


def voxel_size(affine: np.ndarray) -> tuple[float, float, float]:
    """Return the voxel size in mm along each array axis of an affine.

    These are the lengths of the first three columns of the affine. All
    steps of regrep take the voxel size from the affine, not from the
    ``pixdim`` field of the header.
    """
    columns = np.asarray(affine, dtype=np.float64)[:3, :3]
    return tuple(float(s) for s in np.linalg.norm(columns, axis=0))


def same_grid(shape_a, affine_a: np.ndarray, shape_b, affine_b: np.ndarray) -> bool:
    """Whether two volumes share a grid.

    They do when they have the same shape and, up to ``GRID_TOLERANCE``, the
    same affine.
    """
    return (tuple(shape_a) == tuple(shape_b)
            and np.allclose(affine_a, affine_b, rtol=0.0, atol=GRID_TOLERANCE))


def read_array(image: nib.spatialimages.SpatialImage, path: str) -> np.ndarray:
    """Return the float64 ``(X, Y, Z)`` array of a loaded NIfTI image.

    Trailing axes of length 1 after the third (as in an ``(X, Y, Z, 1)``
    file) are removed. Any other array that is not 3-D raises a
    ``ValueError``.
    """
    array = image.get_fdata(dtype=np.float64)
    while array.ndim > 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D volume, but the array has shape "
                         f"{image.shape}")
    return array


def load_volume(path: str) -> Volume:
    """Read a NIfTI file into a ``Volume`` with a float32 array.

    The array is that of :func:`read_array`.
    """
    image = nib.load(path)
    array = np.asarray(read_array(image, path), dtype=np.float32)
    return Volume(array=array, affine=image.affine.copy(),
                  header=image.header.copy(), path=os.path.abspath(path))


def orient_like(volume: Volume, affine: np.ndarray) -> Volume:
    """Reorder the array axes of a volume to the axis orientation of ``affine``.

    The axes are permuted and flipped (no resampling) so that each array axis
    points along the same anatomical direction as the matching axis of the
    grid of ``affine``, as ``nibabel.orientations.io_orientation`` gives the
    directions. A volume that has that orientation already is returned
    unchanged.
    """
    change = ornt_transform(io_orientation(volume.affine), io_orientation(affine))
    if np.array_equal(change, [[0, 1], [1, 1], [2, 1]]):
        return volume
    image = nib.Nifti1Image(volume.array, volume.affine, volume.header)
    turned = image.as_reoriented(change)
    file_index = inv_ornt_aff(change, volume.shape)
    if volume.file_index is not None:
        file_index = volume.file_index @ file_index
    return Volume(array=np.ascontiguousarray(np.asanyarray(turned.dataobj)),
                  affine=turned.affine, header=turned.header, path=volume.path,
                  file_index=file_index)


def read_grid(path: str) -> tuple[tuple[int, int, int], np.ndarray]:
    """Return the shape and the voxel-to-world affine of a NIfTI file.

    Only the header is read. The values are those of :func:`load_volume`.
    """
    image = nib.load(path)
    return tuple(int(n) for n in image.shape[:3]), image.affine.copy()


def working_grid(shape, affine: np.ndarray,
                 cfg: GridConfig) -> tuple[tuple[int, int, int], np.ndarray] | None:
    """Return the working grid of a CT grid, or ``None`` if it is one already.

    Each array axis takes the working voxel size ``s'`` of the anatomical
    direction it points along most closely
    (``nibabel.orientations.io_orientation``). An axis whose voxel size ``s``
    is below ``cfg.resample_below * s'`` gets voxel size ``s'`` and
    ``ceil((n - 1) * s / s') + 1`` voxels: the fewest whose centres span the
    centres of its ``n`` input voxels. Every other axis keeps its voxel size
    and its voxels. The working grid has the same axis directions and the
    same centre as the input grid. Its outermost voxel centres are thus at
    most half a working voxel beyond those of the input, and every input
    voxel centre lies inside it, up to ``_COUNT_ALLOWANCE`` (1e-3) of a
    working voxel: an extent within that allowance of a whole number of
    working voxels is rounded down.

    Parameters
    ----------
    shape
        The shape of the CT.
    affine
        The voxel-to-world affine of the CT.
    cfg
        The working voxel size and the fraction below which an axis is
        resampled.

    Returns
    -------
    tuple or None
        The shape and the voxel-to-world affine of the working grid, or
        ``None`` when every axis keeps its voxel size.
    """
    affine = np.asarray(affine, dtype=np.float64)
    columns = affine[:3, :3]
    spacing = np.linalg.norm(columns, axis=0)
    target = np.asarray(cfg.spacing_mm, dtype=np.float64)[
        io_orientation(affine)[:, 0].astype(int)]
    keep = spacing >= cfg.resample_below * target
    if keep.all():
        return None
    counts = np.asarray(shape, dtype=np.float64)
    resampled = np.ceil((counts - 1) * spacing / target - _COUNT_ALLOWANCE) + 1
    new_counts = np.where(keep, counts, resampled)
    linear = np.where(keep, columns, columns / spacing * target)
    centre = columns @ ((counts - 1) / 2) + affine[:3, 3]
    new_affine = np.eye(4)
    new_affine[:3, :3] = linear
    new_affine[:3, 3] = centre - linear @ ((new_counts - 1) / 2)
    return tuple(int(n) for n in new_counts), new_affine


def check_axis_lengths(shape, path: str) -> None:
    """Raise a ``ValueError`` if a working grid has an axis that is too short.

    Each axis needs at least ``MIN_AXIS_VOXELS`` voxels. ``path`` names the
    CT in the message.
    """
    if min(shape) < MIN_AXIS_VOXELS:
        raise ValueError(
            f"{path}: regrep needs at least {MIN_AXIS_VOXELS} voxels along each "
            f"axis, but the working grid of this CT has {_shape_text(shape)} voxels")


def _shape_text(shape) -> str:
    return " x ".join(str(int(n)) for n in shape)


def working_grid_size(path: str,
                      cfg: GridConfig) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return the shape and the voxel size (mm) of the working grid of a CT file.

    Only the header is read. For a CT that is not resampled
    (:func:`working_grid` gives ``None``), these are the shape and the voxel
    size of the file. A working grid with a short axis raises
    (:func:`check_axis_lengths`).
    """
    image = nib.load(path)
    shape = tuple(int(n) for n in image.shape[:3])
    affine = image.affine
    grid = working_grid(shape, affine, cfg)
    if grid is not None:
        shape, affine = grid
    check_axis_lengths(shape, path)
    return shape, voxel_size(affine)


def resample_linear(volume: Volume, shape, affine: np.ndarray) -> Volume:
    """Resample a volume onto another grid by linear interpolation.

    Grid points beyond the outermost voxel centres of the volume take the
    value of the nearest voxel. The result keeps the source file of the
    volume, and its ``file_index`` maps its voxels to the voxels of that file.

    Parameters
    ----------
    volume
        The volume to resample.
    shape, affine
        The shape and the voxel-to-world affine of the new grid.
    """
    affine = np.asarray(affine, dtype=np.float64)
    # Voxel indices of the new grid to voxel indices of the volume.
    index = np.linalg.solve(np.asarray(volume.affine, dtype=np.float64), affine)
    array = scipy.ndimage.affine_transform(
        volume.array, index[:3, :3], index[:3, 3], output_shape=tuple(shape),
        order=1, mode="nearest")
    header = nib.Nifti1Image(array, affine, volume.header).header
    file_index = index if volume.file_index is None else volume.file_index @ index
    return Volume(array=array, affine=affine, header=header, path=volume.path,
                  file_index=file_index)


def to_working_grid(volume: Volume, grid) -> Volume:
    """Resample a volume onto a working grid from :func:`working_grid`.

    ``grid`` is ``None`` for a timepoint that is not resampled; the volume is
    then returned unchanged.
    """
    return volume if grid is None else resample_linear(volume, *grid)


def file_key(path: str) -> tuple:
    """Return a key that two paths share exactly when they name the same file.

    An existing file is identified by its device and inode, so that symbolic
    and hard links to it have its key. A file that does not exist yet is
    identified by its absolute path after symbolic links are followed.
    """
    try:
        status = os.stat(path)
    except FileNotFoundError:
        return ("path", os.path.realpath(path))
    return ("inode", status.st_dev, status.st_ino)


def check_output(output: str, inputs) -> None:
    """Raise a ``ValueError`` if an output file is one of the input files.

    Parameters
    ----------
    output
        The file that will be written.
    inputs
        The files that are read.
    """
    key = file_key(output)
    for path in inputs:
        if file_key(path) == key:
            raise ValueError(f"the output {output} is the input file {path}")


def save_nifti(path: str, array: np.ndarray, affine: np.ndarray,
               header: nib.Nifti1Header | None = None) -> str:
    """Write a NIfTI file atomically.

    The data go to a temporary file next to ``path``, which is then renamed,
    so a reader never sees a partial file. A ``.nii.gz`` file is written at
    gzip level 0, which stores the data without compression and is fast.
    """
    image = nib.Nifti1Image(array, affine, header)
    if header is not None:
        image.set_data_dtype(array.dtype)
    data = image.to_bytes()
    directory, name = os.path.split(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    partial = os.path.join(directory, f".{name}.{os.getpid()}.part")
    try:
        if name.endswith(".gz"):
            with gzip.open(partial, "wb", compresslevel=0) as handle:
                handle.write(data)
        else:
            with open(partial, "wb") as handle:
                handle.write(data)
        os.replace(partial, path)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return path


class Workspace:
    """A temporary directory for the files of one pair.

    In the concurrent mode the segmentation workers write their maps and
    logs here, and read the chain settings from here; the main process
    waits for the maps. Every map is an uncompressed
    NIfTI file that appears in one atomic rename, so a file that exists is
    complete. A new directory is created in the system default location
    (``TMPDIR``).

    Parameters
    ----------
    keep
        Keep the directory when :meth:`close` is called.
    path
        Use this existing directory instead of creating one. A worker
        process opens the workspace of the main process this way.
    """

    def __init__(self, keep: bool = False, path: str | None = None):
        if path is None:
            path = tempfile.mkdtemp(prefix="regrep-")
        self.root = os.path.abspath(path)
        self.keep = keep

    def path(self, name: str) -> str:
        """The file of the map called ``name``."""
        return os.path.join(self.root, f"{name}.nii")

    def write_labels(self, name: str, array: np.ndarray, like: Volume) -> str:
        """Write a uint8 ``(X, Y, Z)`` map with the affine of ``like``."""
        array = np.ascontiguousarray(array, dtype=np.uint8)
        if array.shape != like.shape:
            raise ValueError(f"{name}: shape {array.shape} differs from the image "
                             f"shape {like.shape}")
        return save_nifti(self.path(name), array, like.affine)

    def read_labels(self, name: str) -> np.ndarray:
        """Read a map written by :meth:`write_labels`."""
        return np.asarray(nib.load(self.path(name)).dataobj, dtype=np.uint8)

    def exists(self, name: str) -> bool:
        """Whether the map called ``name`` has been written."""
        return os.path.exists(self.path(name))

    def close(self) -> None:
        """Remove the directory, unless it is kept."""
        if not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)

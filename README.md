# reps-for-reg

Longitudinal whole-body PSMA PET/CT registration. This is the method that won
the test phase of Learn2Reg 2026 Task 1 (PSMAReg). It registers a follow-up
PET/CT scan to a baseline PET/CT scan of the same patient and returns a
displacement field. The [Method](#method) section describes the steps. The
Python package and its command are called `regrep`.

Project page: <https://www.neeldey.com/reps-for-reg/>

## Installation

You need Linux, an NVIDIA GPU, Python 3.10 to 3.13 and a CUDA toolkit
(`nvcc`) whose version matches the CUDA version of the PyTorch build. In a
fresh environment, run:

```bash
conda create -n regrep python=3.11 && conda activate regrep
git clone https://github.com/neel-dey/reps-for-reg.git
cd reps-for-reg
bash install.sh
```

`install.sh` checks Python and `nvcc` first, then installs PyTorch (CUDA 12.9
build by default; set `TORCH_INDEX` to another PyTorch wheel index if your
driver does not support CUDA 12.9), FireANTs with its CUDA extension
`fireants_fused_ops`, LION (`lionz`), and regrep with anatomix,
TotalSegmentator and nnU-Net. To build the CUDA extension, set `CUDA_HOME` to
the toolkit if `nvcc` is not on the `PATH`, and `TORCH_CUDA_ARCH_LIST` (for
example `"8.9;9.0"`) if the build machine has no GPU or a different GPU. The
comments at the top of the script list all its settings. In step 4, pip
reports that lionz requires other versions of torch, torchvision and
nnunetv2; the script prints a note before this message, and it is expected.
Any other error stops the script.

regrep needs the `main` branch of the FireANTs fork at
<https://github.com/neel-dey/FireANTs>, which `install.sh` installs. The fork
adds functions that regrep calls, such as the `clamp_range` option of the
pyramid downsampling, which the upstream repository
(<https://github.com/rohitrango/FireANTs>) does not have.

Then download the network weights (anatomix, TotalSegmentator and LION):

```bash
regrep download-weights
```

The anatomix weights go to the HuggingFace cache (`HF_HOME`), the
TotalSegmentator weights to `TOTALSEG_WEIGHTS_PATH` if it is set, else to
`TOTALSEG_HOME_DIR` (default `~/.totalsegmentator`), and the LION weights to
`LIONZ_MODELS_DIRECTORY` (default: a folder inside the lionz installation).
Set the same variables when you run regrep. To run without network access
after the download, also set `HF_HUB_OFFLINE=1`.

## Register a pair

The inputs are four 3-D NIfTI files:

* the CT and the PET of the baseline scan (the fixed images), and the CT and
  the PET of the follow-up scan (the moving images);
* CT values in Hounsfield units;
* PET values in SUV. regrep does not convert activity concentration (Bq/mL)
  to SUV, and its lesion support uses an SUV threshold;
* the CT and the PET of one timepoint on one grid (the same array shape and
  affine). The two timepoints may have different grids;
* at least 32 voxels along each axis (on the working grid; see
  [Input geometry](#input-geometry)).

Register the pair and write the field:

```bash
regrep register baseline_ct.nii.gz baseline_pet.nii.gz \
    followup_ct.nii.gz followup_pet.nii.gz field.nii.gz --gpus 0
```

`--gpus` selects the GPUs (default: all visible GPUs; see
[Run modes](#run-modes)). A registration that fails writes no output file and
exits with status 1. The output must not be one of the input files. In the
concurrent mode, `--keep-workspace` keeps the temporary folder with the worker
logs and label maps of a pair.

The field is a NIfTI file with an array of shape `(3, X, Y, Z)`, float32,
where `(X, Y, Z)` is the shape of the fixed CT. It holds displacements in
voxels: fixed voxel `x` corresponds to voxel `x + u(x)` of the moving files,
so the warped moving image takes, at fixed voxel `x`, the moving value at
`x + u(x)`. Component `a` of `u` is along array axis `a`. As in the PSMAReg
challenge, the file has an identity affine; the physical geometry is that of
the fixed CT file. Tools that expect a displacement in mm in the file's own
geometry cannot read it directly.

### Apply the field

`regrep.registration.transforms.spatial_transform` warps a moving volume
with the field. Use `"bilinear"` for images and `"nearest"` for label maps:

```python
import nibabel as nib
import numpy as np
import torch

from regrep.evaluation.metrics import read_field
from regrep.io import load_volume
from regrep.registration.transforms import spatial_transform

field = read_field("field.nii.gz")
disp = torch.from_numpy(np.moveaxis(field, 0, -1).copy())
moving = load_volume("followup_pet.nii.gz")
warped = spatial_transform(torch.from_numpy(moving.array), disp, "bilinear")
fixed = load_volume("baseline_ct.nii.gz")
nib.save(nib.Nifti1Image(warped.numpy(), fixed.affine), "followup_pet_warped.nii.gz")
```

The warped volume is on the grid of the fixed CT, so it is saved with the
affine of the fixed CT.

## Register many pairs

`regrep register-batch` reads a CSV file with one pair per row:

```text
fixed_ct,fixed_pet,moving_ct,moving_pet,output
p01/ct_base.nii.gz,p01/pet_base.nii.gz,p01/ct_fu.nii.gz,p01/pet_fu.nii.gz,fields/p01.nii.gz
p02/ct_base.nii.gz,p02/pet_base.nii.gz,p02/ct_fu.nii.gz,p02/pet_fu.nii.gz,fields/p02.nii.gz
```

Relative paths are relative to the directory in which you run the command.
Pairs run in parallel, one worker per group of `--gpus-per-pair` GPUs:

```bash
regrep register-batch pairs.csv --gpus 0,1,2,3 --gpus-per-pair 2
```

Existing outputs are skipped unless you pass `--overwrite`. Two rows cannot
write the same file, and no output can be an input file of any row. A pair
that fails is reported and the other pairs continue; the command then exits
with status 1.

## Evaluate

`regrep evaluate` scores a field with the challenge metrics: CT organ DSC and
HD95, PET lesion DSC and HD95, MTV and TLG error, and %NDV. It needs
annotations of both timepoints:

* CT organ label maps with the TotalSegmentator `total` class ids 1 to 117,
  for example the `labelsTr` maps of the PSMAReg data;
* PET lesion masks, where any value above 0 is lesion;
* each map on the grid of its image: the fixed maps on the grid of the fixed
  CT and the moving maps on the grid of the moving PET. regrep stops with an
  error when a map is on another grid.

```bash
regrep evaluate --field field.nii.gz --fixed-ct baseline_ct.nii.gz \
    --fixed-labels baseline_organs.nii.gz --moving-labels followup_organs.nii.gz \
    --fixed-lesions baseline_lesions.nii.gz --moving-lesions followup_lesions.nii.gz \
    --moving-pet followup_pet.nii.gz --output scores.csv
```

`--csv` scores many fields; its columns are listed in `regrep evaluate
--help`. MTV and TLG error are the percent change of the moving lesion volume
and of the moving lesion uptake that the warp causes. A metric that is not
defined for a pair, such as the lesion DSC when both lesion masks are empty,
is NaN in a CSV file and null in a JSON file. The label maps that
`--keep-workspace` keeps are the method's own segmentations on its working
grids, not annotations.

`regrep <command> --help` lists all options of each command.

## Configuration

The settings that a user can tune are fields of the dataclasses in
`regrep/config.py`, with a comment for each. `--config` takes a YAML or JSON
file with a nested mapping that overrides some of them:

```yaml
# settings.yaml
features:
  window_batch: 8
stage2:
  iterations: 50
```

```bash
regrep register baseline_ct.nii.gz baseline_pet.nii.gz \
    followup_ct.nii.gz followup_pet.nii.gz field.nii.gz --config settings.yaml
```

A setting with an invalid value stops the command before any work, with a
message that names the setting. The memory estimates of the run plan were
measured with the default feature settings; settings that need more memory,
such as a larger `features.window_batch`, can make a plan that does not fit.

## Input geometry

The registration works in physical space, from the affines of the files.
The fixed and the moving timepoint may have different array shapes, voxel
sizes, origins and orientations. When the array axes of the moving files
point in other anatomical directions than those of the fixed CT (for example
LPS against RAS), regrep reorders them to the order of the fixed CT before
all other steps, so that both timepoints are processed alike, and returns
the field in the voxels of the moving files as they are stored. The voxel
size of a file is taken from its affine.

Many settings of the method are in voxels and were tuned on the voxel size of
the PSMAReg data, 2.7344 x 2.7344 x 3.27 mm (left-right, anterior-posterior
and superior-inferior). On much finer grids the method is less accurate and
needs much more memory. Therefore, along each axis of a CT whose voxel size is
below half of this working size, regrep resamples the CT and the PET of that
timepoint by linear interpolation to the working size, before all other
steps. The new grid has the same axis directions and the same centre, and the
fewest voxels whose centres span those of the input. All other axes keep
their voxels, so a timepoint without such an axis, for example a PSMAReg scan
or a scan of 2 x 2 x 6.54 mm, is used as it is. The field is still returned
on the grid of the fixed CT file and in the voxels of the moving files. The
`grid` section of the configuration sets the working voxel size and the
fraction (0.5) below which an axis is resampled.

When the two grids differ, `x` in the field counts fixed voxels and
`x + u(x)` counts moving voxels, so a transform that moves no point has a
nonzero field. `spatial_transform` normalizes `x + u(x)` by the shape of the
moving image. When the two grids are the same, as in the PSMAReg data, this is
the convention of the challenge. `regrep evaluate` accepts such pairs too: it
compares the lesion volume and uptake before and after the warp in physical
units, and its %NDV divides each Jacobian determinant by that of the
transform that moves no point, so that a pure change of voxel size is not
counted as folding.

## Run modes

`--mode` sets how the steps share the GPUs, and `--gpus` selects the GPUs
(default: all visible GPUs). The GPU indices are those that `nvidia-smi`
shows, unless `CUDA_DEVICE_ORDER` is set to another order. When
`CUDA_VISIBLE_DEVICES` is set, the indices count the GPUs in that list, from
0. The first GPU listed is the main GPU.

* `serial`: all steps run in one process, one after the other.
* `concurrent`: TotalSegmentator and LION run in two worker processes while
  the main process computes the features and runs stage 1. With two or more
  GPUs the segmentors run on the GPU other than the main GPU with the most
  free memory.
* `auto` (default): GPUs other than the main GPU that cannot hold the
  segmentors are left out. Then `concurrent` with two or more GPUs; with one
  GPU, `concurrent` if the segmentors and the feature extraction fit on it
  together, and `serial` otherwise.

In every mode, each stage computes its image term in channel chunks. When
one GPU cannot hold a stage in one chunk, the concurrent mode spreads the
chunks over several GPUs if that gives fewer chunks; the serial mode does so
only when no chunk size fits on one GPU. `--chunk-stage1` and
`--chunk-stage2` set the chunk size.

Measured on 46 GB NVIDIA L40S GPUs, with pairs of 192 x 192 x 288 voxels:

| mode | GPUs | time per pair | peak memory per GPU |
|---|--:|--:|---|
| serial | 1 | 163 s | 42 GB |
| concurrent | 1 | 143 s | 42 GB |
| concurrent | 2 | 127 s | 42 + 8 GB |
| auto (stage 2 on three GPUs) | 4 | 110 s | 38 / 22 / 31 / 9 GB |

All modes optimize the same objective, but their fields are not
bit-identical. The chunks change the order of floating-point sums, and the
segmentation networks and some GPU kernels are not deterministic. In our
tests on L40S GPUs, repeated runs and runs in different modes differed by
about 0.05 voxels on average.

The memory grows with the physical extent of the scans and with the number of
voxels after the resampling above; because the much finer axes are resampled,
a scan with voxels of 0.98 mm in-plane needs about as much as a PSMAReg scan of
the same extent. The plan estimates the memory on these grids before the
registration starts and stops with an error when it does not fit on the GPUs
given.

## Method

The method combines organ and lesion segmentations with learned and handcrafted
image features in two deformable stages, built on FireANTs. Because the
challenge also scores the lesion volume and uptake after the warp, both
stages penalize volume change on the lesions, and a post-hoc projection
corrects the remaining change. The steps are:

1. **Organ segmentation.** Seven TotalSegmentator models of the `total` task
   label 117 structures in both CTs. The 6 mm and 3 mm maps of the moving CT
   feed the lesion supports; the 1.5 mm part models give the label maps of
   stage 2.
2. **Lesion segmentation.** LION (PSMA model) segments the lesions in both
   PETs.
3. **Features.** For each timepoint: a body mask from a CT threshold, 32
   anatomix channels from a three-level feature pyramid, and 12 MIND-SSC
   channels, all computed from the CT. Both timepoints use the same pyramid
   levels: 1 mm, 1.75 mm and the smallest voxel size of the working grid of
   the fixed CT (see [Input geometry](#input-geometry)).
4. **Stage 1.** A deformable registration of a 46-channel stack (CT, PET,
   anatomix, MIND-SSC) with a masked local normalized cross-correlation on a
   four-level pyramid. A volume penalty keeps `det J` near 1 on the moving
   lesion support (PET above an SUV threshold outside the physiological
   organs, together with the LION mask).
5. **Stage 2.** A label-driven deformable registration that starts from
   stage 1. Its objective is a generalized Dice loss and a chamfer distance
   on banded distance channels of the organs and the lesions, a
   region-distance term, a masked local NCC on the anatomix channels, and the
   volume penalty on the lesion support.
6. **Volume projection.** After stage 1 and again on the composed transform,
   the field is corrected so that each lesion component keeps its volume
   under the warp. On the composed transform, the projection measures the
   field through a mild low pass (a downsampling by 2 and an upsampling
   back), and composes the corrections into the field itself. A component
   that reaches the edge of the fixed field of view after the warp is left
   out, because only part of its volume is visible.
7. **Low pass.** A Gaussian filter (sigma 1 voxel) smooths the displacement
   field.
8. **Exterior fold repair.** Where the field folds outside the body, it is
   blurred locally until the folds go away. Voxels in the body and in a 5 mm
   collar around it do not change.

## Repository layout

| path | content |
|---|---|
| `regrep/cli.py` | the `regrep` command |
| `regrep/pipeline.py`, `regrep/runtime.py`, `regrep/batch.py` | one registration, the run plan and GPU use, many registrations |
| `regrep/config.py` | the settings of the method |
| `regrep/io.py` | NIfTI reading and writing, the workspace of a pair |
| `regrep/weights.py` | locating and downloading the network weights |
| `regrep/segmentation/` | TotalSegmentator and LION runners, the lesion supports |
| `regrep/features/` | body mask, anatomix and MIND-SSC features |
| `regrep/registration/` | the chunked FireANTs optimizer, stage 1, stage 2 and their losses |
| `regrep/posthoc/` | volume projection, fold measure and repair, field export |
| `regrep/evaluation/` | the challenge metrics |
| `tests/` | unit tests that need no data (`pytest`) |

## Citation

A paper that describes regrep is in preparation; until it appears, please
cite this repository.

regrep builds on these works; please cite them too. TotalSegmentator and LION
are nnU-Net models, and TotalSegmentator asks you to cite nnU-Net as well:

```bibtex
@misc{dey2024learninggeneralpurposebiomedicalvolume,
  title         = {Learning General-Purpose Biomedical Volume Representations using Randomized Synthesis},
  author        = {Neel Dey and Benjamin Billot and Hallee E. Wong and Clinton J. Wang and Mengwei Ren and P. Ellen Grant and Adrian V. Dalca and Polina Golland},
  year          = {2024},
  eprint        = {2411.02372},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2411.02372}
}

@article{jena2024fireants,
  title   = {FireANTs: Adaptive Riemannian Optimization for Multi-Scale Diffeomorphic Registration},
  author  = {Jena, Rohit and Chaudhari, Pratik and Gee, James C},
  journal = {Nature Communications},
  year    = {2024}
}

@article{wasserthal2023totalsegmentator,
  title   = {TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images},
  author  = {Wasserthal, Jakob and Breit, Hanns-Christian and Meyer, Manfred T. and Pradella, Maurice and Hinck, Daniel and Sauter, Alexander W. and Heye, Tobias and Boll, Daniel T. and Cyriac, Joshy and Yang, Shan and Bach, Michael and Segeroth, Martin},
  journal = {Radiology: Artificial Intelligence},
  volume  = {5},
  number  = {5},
  pages   = {e230024},
  year    = {2023},
  doi     = {10.1148/ryai.230024}
}

@software{lion,
  title     = {LION: fully automated tumor segmentation for FDG and PSMA PET scans},
  author    = {Shiyam Sundar, Lalith Kumar and Gutschmayer, Sebastian and Pires, Manuel},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.12626789},
  url       = {https://github.com/ENHANCE-PET/LION}
}

@article{isensee2021nnunet,
  title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021},
  doi     = {10.1038/s41592-020-01008-z}
}

@inproceedings{heinrich2013mindssc,
  title     = {Towards Realtime Multimodal Fusion for Image-Guided Interventions Using Self-similarities},
  author    = {Heinrich, Mattias P. and Jenkinson, Mark and Papie{\.z}, Bart{\l}omiej W. and Brady, Sir Michael and Schnabel, Julia A.},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2013},
  series    = {Lecture Notes in Computer Science},
  volume    = {8149},
  pages     = {187--194},
  year      = {2013},
  publisher = {Springer},
  doi       = {10.1007/978-3-642-40811-3_24}
}
```

## License

regrep is released under the MIT License (see `LICENSE`). Some files adapt
code from other projects under the Apache License 2.0 (see `LICENSE-APACHE`);
`NOTICE` lists them, with their copyright lines.

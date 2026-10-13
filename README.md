[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# reps-for-reg

### [Project page](https://www.neeldey.com/reps-for-reg/) | [Workshop talk](https://youtu.be/H51kyUb5Jog) | [Learn2Reg challenge](https://learn2reg.grand-challenge.org/) | [anatomix registration](https://github.com/neel-dey/anatomix/tree/main/anatomix/registration)

![CT and PET of training pair 0028 before and after registration, and the deformation](https://www.neeldey.com/reps-for-reg/media/readme_highlight.jpg)

*From left to right: the CT scans before and after registration, the PET
scans before and after registration, and the deformation. In each overlay,
one scan is green and the other scan is magenta. Aligned structures are
grey.*

This repository registers a follow-up PET/CT scan to a baseline PET/CT scan
of the same patient. It returns the displacement field and the moved CT and
PET scans. Many of the choices below are specific to the PSMAReg challenge
(e.g., producing displacements in voxels instead of physical coordinates).
For a general and better functional registration interface, use
[this repo](https://github.com/neel-dey/anatomix/tree/main/anatomix/registration)
instead.

## Installation

Make and activate a new environment with Python 3.10 to 3.13, with the tool
of your choice (for example conda, micromamba, venv or uv). Then run:

```bash
git clone https://github.com/neel-dey/reps-for-reg.git
cd reps-for-reg
bash install.sh
```

* `install.sh` installs PyTorch (CUDA 12.9 build), our
  [FireANTs fork](https://github.com/neel-dey/FireANTs) with its CUDA
  extension, LION, TotalSegmentator, anatomix and regrep.
* The CUDA extension needs Linux, an NVIDIA GPU and a CUDA toolkit (`nvcc`)
  with the CUDA version of PyTorch.
* The comments at the top of `install.sh` list its settings, for example
  `TORCH_INDEX` for another CUDA version.

Then download the network weights (anatomix, TotalSegmentator and LION):

```bash
regrep download-weights
```

The anatomix weights will go to the HuggingFace cache (`HF_HOME`), the
TotalSegmentator weights to `TOTALSEG_WEIGHTS_PATH` or `TOTALSEG_HOME_DIR`
(default `~/.totalsegmentator`), and the LION weights to
`LIONZ_MODELS_DIRECTORY` (default: a folder in the lionz installation). Set
the same variables when you run regrep. After the download, regrep does not
need network access.

## Register a pair

The inputs are four 3-D NIfTI files: the CT and the PET of the baseline scan
(the fixed images), and the CT and the PET of the follow-up scan (the moving
images).

```bash
regrep register baseline_ct.nii.gz baseline_pet.nii.gz \
    followup_ct.nii.gz followup_pet.nii.gz out/ --gpus 0
```

This writes three files to `out/`:

* `field.nii.gz`: the displacement field, in the format of the PSMAReg
  challenge (see below).
* `moved_ct.nii.gz` and `moved_pet.nii.gz`: the follow-up CT and PET, warped
  onto the grid of the baseline CT.

`--gpus` selects the GPUs (default: all visible GPUs, see
[Single/multi-GPU execution modes](#singlemulti-gpu-execution-modes)).

The field is a float32 array of shape `(3, X, Y, Z)` on the grid of the
baseline CT, with an identity affine. It holds displacements in voxels:
baseline voxel `x` corresponds to voxel `x + u(x)` of the follow-up files.
Tools that expect a displacement in mm cannot read it directly.

## Register many pairs

`regrep register-batch` reads a CSV file with one pair per row. The `output`
column names the output folder of each pair:

```text
fixed_ct,fixed_pet,moving_ct,moving_pet,output
p01/ct_base.nii.gz,p01/pet_base.nii.gz,p01/ct_fu.nii.gz,p01/pet_fu.nii.gz,out/p01
p02/ct_base.nii.gz,p02/pet_base.nii.gz,p02/ct_fu.nii.gz,p02/pet_fu.nii.gz,out/p02
```

Each folder gets the field and the moved CT and PET, as with
`regrep register`. Pairs run in parallel, one pair on each group of
`--gpus-per-pair` GPUs:

```bash
regrep register-batch pairs.csv --gpus 0,1,2,3 --gpus-per-pair 2
```

Pairs whose outputs exist are skipped, unless you add `--overwrite`. When a
pair fails, the other pairs continue.

## Evaluate

`regrep evaluate` scores a field with the challenge metrics: CT organ DSC and
HD95, PET lesion DSC and HD95, MTV and TLG error, and %NDV. It needs the CT
organ labels (TotalSegmentator `total` classes, as in the PSMAReg
`labelsTr`) and the PET lesion masks of both scans, each on the grid of its
image.

```bash
regrep evaluate --field out/field.nii.gz --fixed-ct baseline_ct.nii.gz \
    --fixed-labels baseline_organs.nii.gz --moving-labels followup_organs.nii.gz \
    --fixed-lesions baseline_lesions.nii.gz --moving-lesions followup_lesions.nii.gz \
    --moving-pet followup_pet.nii.gz --output scores.csv
```

`--csv` scores many fields. `regrep <command> --help` lists all options of
each command.

## Configuration

`regrep/config.py` holds the settings of the method, with a comment for each.
`--config` takes a YAML or JSON file that changes some of them:

```yaml
# settings.yaml
features:
  window_batch: 8
stage2:
  iterations: 50
```

```bash
regrep register baseline_ct.nii.gz baseline_pet.nii.gz \
    followup_ct.nii.gz followup_pet.nii.gz out/ --config settings.yaml
```

regrep checks all settings before it starts. Settings that use more memory,
such as a larger `features.window_batch`, can make the memory estimate too
low.

## Input geometry

* regrep uses the affines of the files. The two scans can have different
  shapes, voxel sizes, origins and orientations.
* The CT and the PET of one scan must be on the same grid.
* The method was tuned on the voxel size of the PSMAReg data,
  2.7344 x 2.7344 x 3.27 mm. An axis with voxels smaller than half of this
  size is resampled to this size first. The `grid` settings change these
  values.
* Each axis needs at least 32 voxels after this step.
* The outputs are on the grid of the baseline CT file. The field points to
  the voxels of the follow-up files as they are stored.
* When the two scans have different grids, `x` counts baseline voxels and
  `x + u(x)` counts follow-up voxels. Thus a transform that moves no point
  has a field that is not zero. `regrep evaluate` corrects for this.

## Single/multi-GPU execution modes

`--gpus` selects the GPUs (default: all visible GPUs). The indices are those
that `nvidia-smi` shows. When `CUDA_VISIBLE_DEVICES` is set, they count the
GPUs in that list. The first GPU is the main GPU. `--mode` sets how the
steps use the GPUs:

* `serial`: all steps run in one process, one after the other.
* `concurrent`: TotalSegmentator and LION run in worker processes while the
  main process computes the features and runs stage 1. With two or more GPUs,
  the segmentation runs on a GPU other than the main GPU.
* `auto` (default): `concurrent` if the GPUs have sufficient memory for it,
  else `serial`.

When a registration stage does not fit on one GPU, regrep divides its image
loss into channel chunks. The chunks can go to different GPUs.
`--chunk-stage1` and `--chunk-stage2` set the chunk size.

Measured on 46 GB NVIDIA L40S GPUs, with pairs of 192 x 192 x 288 voxels:

| mode | GPUs | time per pair | peak memory per GPU |
|---|--:|--:|---|
| serial | 1 | 163 s | 42 GB |
| concurrent | 1 | 143 s | 42 GB |
| concurrent | 2 | 127 s | 42 + 8 GB |
| auto (stage 2 on three GPUs) | 4 | 110 s | 38 / 22 / 31 / 9 GB |

The fields of two runs differ by approximately 0.05 voxels on average, also
in the same mode. The chunks change the order of floating-point sums, and
some GPU kernels are not deterministic.

The memory increases with the physical size of the scans. regrep estimates
the memory before it starts, and stops with an error if the pair does not
fit on the GPUs.

## Method

The [project page](https://www.neeldey.com/reps-for-reg/) describes the
method.

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

regrep uses these works. Please cite them:

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

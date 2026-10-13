#!/usr/bin/env bash
# Install regrep and its dependencies into the active Python environment.
#
# Run it from the root of the reps-for-reg repository, in a fresh environment
# with Python 3.10 to 3.13 (a venv, conda or micromamba environment), for
# example:
#
#     conda create -n regrep python=3.11 && conda activate regrep
#     bash install.sh
#
# FireANTs is installed from the main branch of the fork at
# https://github.com/neel-dey/FireANTs, which has functions that regrep calls
# (for example the clamp_range option of the pyramid downsampling) that the
# upstream repository https://github.com/rohitrango/FireANTs does not have.
#
# Environment variables:
#   PYTHON          the Python interpreter to install into (default: python)
#   TORCH_INDEX     the PyTorch wheel index (default: the CUDA 12.9 index).
#                   The NVIDIA driver must support this CUDA version; see
#                   https://pytorch.org/get-started/locally/ for other builds.
#   FIREANTS_DIR    where FireANTs is cloned
#                   (default: ~/.cache/regrep/FireANTs)
#   CUDA_HOME       the CUDA toolkit used to build the FireANTs CUDA extension.
#                   Its nvcc must match the CUDA version of the torch build.
#   TORCH_CUDA_ARCH_LIST
#                   the GPU architectures to build the extension for, for
#                   example "8.9" for an L40S or "9.0" for an H100 (default:
#                   the architectures of the visible GPUs).

set -euo pipefail

PYTHON="${PYTHON:-python}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"
# anatomix pins these versions; a different torch would be replaced by the
# anatomix requirement in step 5.
TORCH_VERSION="2.13.0"
TORCHVISION_VERSION="0.28.0"
# regrep uses the LION model URL and some internal functions of lionz, and was
# tested with this version.
LIONZ_VERSION="1.0.6"
FIREANTS_URL="https://github.com/neel-dey/FireANTs.git"
FIREANTS_DIR="${FIREANTS_DIR:-${HOME}/.cache/regrep/FireANTs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP=("${PYTHON}" -m pip)

# Check the interpreter and the CUDA toolkit before anything is installed.
# lionz supports Python 3.10 to 3.13.
"${PYTHON}" -c 'import sys; v = sys.version_info[:2]; sys.exit(0 if (3, 10) <= v <= (3, 13) else "error: regrep needs Python 3.10 to 3.13, not %d.%d" % v)'
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if ! command -v nvcc > /dev/null; then
    echo "error: nvcc was not found. Set CUDA_HOME to a CUDA toolkit whose" \
         "version matches the CUDA version of the torch build (12.9 by default;" \
         "see TORCH_INDEX)." >&2
    exit 1
fi

echo "==> [1/5] Installing torch ${TORCH_VERSION} and torchvision ${TORCHVISION_VERSION} from ${TORCH_INDEX}"
"${PIP[@]}" install --upgrade pip
"${PIP[@]}" install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX}"
"${PYTHON}" -c 'import torch; print("    torch", torch.__version__, "CUDA", torch.version.cuda)'

echo "==> [2/5] Installing FireANTs (main branch of ${FIREANTS_URL}) from ${FIREANTS_DIR}"
# The clone does not fetch the submodules: the cookbook submodule uses an SSH
# URL, and FireANTs does not need it.
if [ -d "${FIREANTS_DIR}/.git" ]; then
    git -C "${FIREANTS_DIR}" pull --ff-only
else
    mkdir -p "$(dirname "${FIREANTS_DIR}")"
    git clone --depth 1 "${FIREANTS_URL}" "${FIREANTS_DIR}"
fi
"${PIP[@]}" install "${FIREANTS_DIR}"

echo "==> [3/5] Building the FireANTs CUDA extension (fireants_fused_ops)"
echo "    nvcc: $(nvcc --version | tail -n 1)"
# With ninja, torch compiles the CUDA source files in parallel; without it,
# one file after another. torch looks for the ninja program on PATH, so the
# scripts directory of the interpreter, where pip puts ninja, goes first on
# PATH for the build. It is not on PATH when the environment is not activated.
"${PIP[@]}" install ninja
SCRIPTS_DIR="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
PATH="${SCRIPTS_DIR}:${PATH}" "${PIP[@]}" install --no-build-isolation "${FIREANTS_DIR}/fused_ops"
"${PYTHON}" -c 'from fireants.interpolator import FFO_AVAILABLE; assert FFO_AVAILABLE, "fireants_fused_ops does not load"'

echo "==> [4/5] Installing LION (lionz ${LIONZ_VERSION}) without its torch pin"
echo "    pip may report that lionz requires other versions of torch, torchvision"
echo "    and nnunetv2. This is expected: regrep uses the versions installed here."
# lionz pins an old torch, so it is installed without its dependencies, and
# its other dependencies are installed here. The list below is the dependency
# list of LIONZ_VERSION without torch, torchvision and nnunetv2; nnunetv2
# comes with regrep in step 5.
"${PIP[@]}" install --no-deps "lionz==${LIONZ_VERSION}"
"${PIP[@]}" install \
    nibabel halo pandas SimpleITK "pydicom>=2.4.5" imageio numpy matplotlib \
    pyfiglet cfonts rich-click natsort "pillow>=12.2.0" colorama \
    "dask[distributed]>=2026.1.0" rich "dicom2nifti>=2.6.2" emoji \
    "requests>=2.33.0" opencv-python

echo "==> [5/5] Installing regrep (with anatomix main, TotalSegmentator and nnU-Net)"
"${PIP[@]}" install -e "${REPO_DIR}"

"${PYTHON}" -c 'import torch, fireants, lionz, totalsegmentator, nnunetv2, anatomix, regrep; print("    regrep", regrep.__version__, "with torch", torch.__version__)'
# Step 5 can replace torch; the CUDA extension must still load with the torch
# that is installed now.
"${PYTHON}" -c 'from fireants.interpolator import FFO_AVAILABLE; assert FFO_AVAILABLE, "fireants_fused_ops does not load with the installed torch; rebuild it with pip install --no-build-isolation FIREANTS_DIR/fused_ops"'
echo "==> Done. Next, download the network weights: regrep download-weights"

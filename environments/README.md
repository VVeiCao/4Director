# Environment

Every stage and the app run in one Conda environment, `4director`
(Python 3.11, Torch 2.7.1 + CUDA 12.8):

```bash
conda env create -f environments/4director.yml
conda activate 4director
pip install -e '.[ui]'
CUDA_HOME=/usr/local/cuda-12.8 bash environments/install_extensions.sh
```

`requirements/4director.txt` holds the pip packages. `install_extensions.sh`
adds what pip cannot resolve, building each piece in a scratch directory or
inside the environment, so no checkout is modified:

- SAM 2 for the app's mask, at a pinned commit.
- PyTorch3D for the stage-3 depth renderer, at a pinned commit.
- The CUDA extensions Pixal3D inherits from TRELLIS.2 (o-voxel, FlexGEMM,
  CuMesh, nvdiffrast, nvdiffrec), plus its utils3d and natten wheels.
- MegaSAM from the `third_party/TAPIP3D` submodule, patched with
  `third_party/patches/tapip3d.tracked.patch` into `<env>/opt/megasam`, with
  its `droid_backends` and `lietorch` compiled, RAFT's `raft-things.pth`
  downloaded, and a matching `torch-scatter` wheel.

It needs a GPU and an nvcc for CUDA 12.x, ends by importing every extension,
and skips what is already done when rerun.

Stage 4 imports `diffsynth` from the pinned `third_party/DiffSynth-Studio`
submodule, and Pixal3D runs from a patched copy of the pinned
`third_party/Pixal3D` submodule, so neither is installed separately.

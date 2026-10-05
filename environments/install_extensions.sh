#!/usr/bin/env bash
# Add the compiled pieces, and the cuDNN video generation runs with, to the
# environment from 4director.yml. Run it once,
# inside that activated environment, on a machine with a GPU and a CUDA 12.8
# toolkit, the version Torch was built with:
#
#   conda activate 4director
#   bash environments/install_extensions.sh
#
# The toolkit is $CUDA_HOME if set, else /usr/local/cuda-12.8 if it exists,
# else the nvcc on PATH.
#
# Everything is built in a scratch directory or inside the environment, so no
# checkout, submodule included, is modified. A rerun skips what is already done.
set -euo pipefail

REPOSITORY=$(cd "$(dirname "$0")/.." && pwd)
WORK=${EXTENSIONS_DIR:-$(mktemp -d -t 4director-extensions-XXXXXX)}
PREFIX=$(python -c 'import sys; print(sys.prefix)')
TRELLIS2_REPOSITORY=${TRELLIS2_REPOSITORY:-https://github.com/microsoft/TRELLIS.2.git}
PYTORCH3D_COMMIT=d34e87ce5266634ffd2b155ffd08ba02542fa009
SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
# Pixal3D's CUDA extensions, at the commits the release was validated with.
NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae  # tag v0.4.0
NVDIFFREC_COMMIT=b296927cc7fd01c2ac1087c8065c4d7248f72da4   # branch renderutils
CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
FLEXGEMM_COMMIT=6dd94a859c26ee8246888502eada3dd8ad85532e
TRELLIS2_COMMIT=${TRELLIS2_COMMIT:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}
# MegaSAM as TAPIP3D ships it (the third_party/TAPIP3D submodule).
MEGASAM_SOURCE=${MEGASAM_SOURCE:-$REPOSITORY/third_party/TAPIP3D/third_party/megasam}
MEGASAM_HOME="$PREFIX/opt/megasam"
RAFT_SHA256=fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1
mkdir -p "$WORK"
echo "Building under $WORK"

python -c "import torch; assert torch.cuda.is_available(), 'a CUDA GPU is required'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Torch refuses to build an extension with a CUDA toolkit of another major
# version, and a machine's default /usr/local/cuda often is one.
TORCH_CUDA=$(python -c 'import torch; print(torch.version.cuda)')
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -x "/usr/local/cuda-$TORCH_CUDA/bin/nvcc" ]; then
    CUDA_HOME="/usr/local/cuda-$TORCH_CUDA"
  elif command -v nvcc > /dev/null; then
    CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
  else
    echo "No CUDA toolkit found: set CUDA_HOME to a CUDA $TORCH_CUDA install" >&2
    exit 1
  fi
fi
NVCC_CUDA=$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
if [ "${NVCC_CUDA%%.*}" != "${TORCH_CUDA%%.*}" ]; then
  echo "$CUDA_HOME has CUDA $NVCC_CUDA, but Torch was built with CUDA $TORCH_CUDA;" \
    "set CUDA_HOME to a CUDA $TORCH_CUDA toolkit" >&2
  exit 1
fi
if [ "$NVCC_CUDA" != "$TORCH_CUDA" ]; then
  echo "note: compiling with CUDA $NVCC_CUDA; 4Director was validated with CUDA $TORCH_CUDA" >&2
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
echo "Compiling with CUDA $NVCC_CUDA from $CUDA_HOME"
pip install ninja

installed() { python -c "import $1" 2>/dev/null; }

clone() {  # url destination commit [git clone options...]
  local url=$1 destination=$2 commit=$3
  shift 3
  [ -d "$destination" ] || git clone "$@" "$url" "$destination"
  git -C "$destination" checkout -q "$commit"
  git -C "$destination" submodule update --init --recursive -q
}

# SAM2 for the app's mask, at a pinned commit. The image predictor does not
# need SAM2's optional CUDA extension, so it is not built.
installed sam2.build_sam || SAM2_BUILD_CUDA=0 pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/sam2.git@$SAM2_COMMIT"

# Pixal3D was written against utils3d 0.0.2, while MoGe-2 needs the newer
# utils3d the environment already has. Keep Pixal3D's copy beside it; only the
# Pixal3D process puts this directory first on its path.
PIXAL3D_UTILS3D="$PREFIX/opt/pixal3d-utils3d"
pip install --no-deps --upgrade --target "$PIXAL3D_UTILS3D" \
  "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl#sha256=ff63440827d6933807dd06c8a5a2db7e51fd5f33c7f3dddcc766a80e0f419252"

# cuDNN 9.13.1, the version the released videos were generated with; Torch
# 2.7.1 ships 9.7.1. It sits beside the environment's packages rather than
# replacing Torch's, and the Wan runner loads it from there. cuDNN looks each
# sublibrary up by its full-version name first, so give them those names too.
CUDNN_LIB="$PREFIX/opt/cudnn/nvidia/cudnn/lib"
[ -f "$CUDNN_LIB/libcudnn.so.9" ] || pip install --no-deps --target "$PREFIX/opt/cudnn" \
  nvidia-cudnn-cu12==9.13.1.26
for library in "$CUDNN_LIB"/libcudnn_*.so.9; do
  ln -sfn "$(basename "$library")" "$library.13.1"
done

# Prebuilt wheels: natten for Pixal3D's NAF upsampler, torch-scatter for MegaSAM.
installed natten || pip install \
  "https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.0/natten-0.21.0%2Btorch270cu128-cp311-cp311-linux_x86_64.whl#sha256=b3d946655dde77c616611b4a9b2f75b9cb8fa2653e6176718c5d9ec001cd6bcc"
installed torch_scatter || pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Stage-3 renderer: PyTorch3D at the commit the teaser controls were verified with.
if ! installed pytorch3d.renderer; then
  clone https://github.com/facebookresearch/pytorch3d.git "$WORK/pytorch3d" "$PYTORCH3D_COMMIT"
  FORCE_CUDA=1 pip install "$WORK/pytorch3d" --no-build-isolation
fi

# Pixal3D: the CUDA extensions it inherits from TRELLIS.2.
build() {  # module url destination commit [git clone options...]
  local module=$1 url=$2 destination=$3 commit=$4
  shift 4
  installed "$module" && return
  clone "$url" "$destination" "$commit" "$@"
  pip install "$destination" --no-build-isolation
}
build nvdiffrast.torch https://github.com/NVlabs/nvdiffrast.git "$WORK/nvdiffrast" "$NVDIFFRAST_COMMIT" -b v0.4.0
build nvdiffrec_render https://github.com/JeffreyXiang/nvdiffrec.git "$WORK/nvdiffrec" "$NVDIFFREC_COMMIT" -b renderutils
build cumesh https://github.com/JeffreyXiang/CuMesh.git "$WORK/CuMesh" "$CUMESH_COMMIT" --recursive
build flex_gemm https://github.com/JeffreyXiang/FlexGEMM.git "$WORK/FlexGEMM" "$FLEXGEMM_COMMIT" --recursive
if ! installed o_voxel; then
  clone "$TRELLIS2_REPOSITORY" "$WORK/TRELLIS.2" "$TRELLIS2_COMMIT" --recursive
  pip install "$WORK/TRELLIS.2/o-voxel" --no-build-isolation
fi

# MegaSAM: a patched copy inside the environment, where stage 1 and the app
# look for it. third_party/patches/tapip3d.tracked.patch makes its depth
# alignment skip invalid pixels. The marker records the patch and the TAPIP3D
# commit, so changing either rebuilds the copy and its backends.
MEGASAM_PATCH="$REPOSITORY/third_party/patches/tapip3d.tracked.patch"
MEGASAM_STAMP="$(sha256sum "$MEGASAM_PATCH" | cut -d' ' -f1) $(git -C "$MEGASAM_SOURCE" rev-parse HEAD 2>/dev/null || echo "$MEGASAM_SOURCE")"
MEGASAM_REBUILT=0
if [ "$(cat "$MEGASAM_HOME/.4director-ready" 2>/dev/null)" != "$MEGASAM_STAMP" ]; then
  rm -rf "$MEGASAM_HOME"
  mkdir -p "$(dirname "$MEGASAM_HOME")"
  cp -r "$MEGASAM_SOURCE" "$MEGASAM_HOME"
  git -C "$MEGASAM_HOME" init -q
  git -C "$MEGASAM_HOME" apply -p3 "$MEGASAM_PATCH"
  rm -rf "$MEGASAM_HOME/.git"
  # RAFT's released weights, the raft-things.pth MegaSAM's README points to.
  curl -sSL -o "$WORK/raft-models.zip" https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
  unzip -o -q -j "$WORK/raft-models.zip" models/raft-things.pth -d "$MEGASAM_HOME/cvd_opt"
  echo "$RAFT_SHA256  $MEGASAM_HOME/cvd_opt/raft-things.pth" | sha256sum -c -
  echo "$MEGASAM_STAMP" > "$MEGASAM_HOME/.4director-ready"
  MEGASAM_REBUILT=1
fi

# MegaSAM camera tracking: droid_backends and lietorch from its base/.
if [ "$MEGASAM_REBUILT" = 1 ] || ! (cd "$WORK" && installed "torch, droid_backends, lietorch"); then
  rm -rf "$WORK/megasam-base"
  cp -r "$MEGASAM_HOME/base" "$WORK/megasam-base"
  rm -rf "$WORK/megasam-base/build" "$WORK/megasam-base"/*.egg-info
  # MegaSAM and its lietorch predate Torch 2's dispatch API: their dispatch
  # macros must get tensor.scalar_type(), not the removed tensor.type().
  sed -i -E 's/((AT_)?DISPATCH_[A-Z_]+\(([^,()]+,[[:space:]]*)?[A-Za-z0-9_]+)\.type\(\)/\1.scalar_type()/g' \
    "$WORK/megasam-base"/src/*.cu \
    "$WORK/megasam-base"/thirdparty/lietorch/lietorch/src/*.cu \
    "$WORK/megasam-base"/thirdparty/lietorch/lietorch/src/*.cpp
  (cd "$WORK/megasam-base" && python setup.py install)
fi

cd "$WORK"
python - "$PIXAL3D_UTILS3D" <<'PY'
import importlib, subprocess, sys
for name in ("pytorch3d.renderer", "nvdiffrast.torch", "cumesh", "flex_gemm", "o_voxel",
             "natten", "torch_scatter", "droid_backends", "lietorch", "moge.model.v2",
             "xformers.ops", "sam2.build_sam"):
    importlib.import_module(name)
import utils3d
assert hasattr(utils3d, "pt"), "MoGe-2 needs the newer utils3d"
# Pixal3D's copy must still offer the old API from its side directory.
subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, sys.argv[1]); import utils3d.torch as t; t.intrinsics_from_fov_xy",
     sys.argv[1]],
    check=True,
)
print("every compiled extension imports cleanly")
PY

"""Locate the checkout, the helper scripts the stages shell out to, the
weights, and the optional ``components.env`` that moves some of them."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Paths that live somewhere else on this machine go in this untracked file.
COMPONENT_ENV_FILE = "components.env"
# What it may set. These come from that file alone, never from the shell, so an
# export left over from another project cannot point this checkout elsewhere.
COMPONENT_SETTINGS = (
    "FOURDIRECTOR_CHECKPOINT",
    "WAN_ROOT",
    "FOURDIRECTOR_SAM2_CHECKPOINT",
    "FOURDIRECTOR_QWEN3_VL_SEED",
)

_SCRIPTS: dict[str, str] = {
    "aligner": "third_party/alignment/align_mesh.py",
    "vace_runner": "third_party/inference/wan_vace/run_inference.py",
    "renderer": (
        "third_party/production_renderer/src/baseline_eval/adapters/v5_rotation/controls.py"
    ),
    "pixal3d_generator": (
        "third_party/scripts/evaluation/vendor/pixal3d/generate_with_correspondence.py"
    ),
    "pixal3d_adapter": "third_party/pixal3d_adapter/run_patched.py",
    "pixal3d_patch": "third_party/patches/pixal3d.tracked.patch",
    "moge2_worker": "third_party/geometry/megasam/moge2_infer_video.py",
    "megasam_orchestrator": "third_party/geometry/megasam/inference_moge2.py",
}

# Where download_models.py puts weights, and where every stage looks for them.
MODELS = ROOT / "models"
SAM2_CHECKPOINT = MODELS / "sam2" / "sam2_hiera_large.pt"
MOTION_ADAPTER_CHECKPOINT = MODELS / "4Director" / "step-2610.safetensors"
# The Hugging Face and Torch caches, which the models the stages fetch on first
# use (MoGe-2, UniDepth, Qwen3-VL, Pixal3D) fill.
MODEL_CACHE = MODELS / "cache"

# Upstream Pixal3D at the pinned commit. Stage 2 runs it through the adapter,
# which applies third_party/patches/pixal3d.tracked.patch in a throwaway copy.
PIXAL3D_SUBMODULE = ROOT / "third_party" / "Pixal3D"

# MegaSAM with its UniDepth, copied and patched into the environment by
# environments/install_extensions.sh.
MEGASAM_HOME = Path(sys.prefix) / "opt" / "megasam"

# The cuDNN 9.13.1 the Wan runner loads, which install_extensions.sh also adds.
CUDNN_LIB = Path(sys.prefix) / "opt" / "cudnn" / "nvidia" / "cudnn" / "lib"


def script(name: str, *, root: Path | None = None) -> Path:
    path = (root or ROOT) / _SCRIPTS[name]
    if not path.is_file():
        raise FileNotFoundError(f"cannot locate the {name} script at {path}")
    return path


def required_path(
    name: str,
    *,
    directory: bool = False,
    default: Path | None = None,
) -> Path:
    """Resolve a configured path, naming the variable when it is unusable.

    Relative values resolve from the checkout root, which is how the bundled
    defaults are written; components.env itself should use absolute paths.
    """
    value = os.environ.get(name)
    if not value and default is None:
        kind = "directory" if directory else "file"
        raise RuntimeError(f"Set {name} to the required {kind} path")
    # Keep environment executables as configured: Conda/venv ``python`` is
    # commonly a symlink, and resolving it would bypass that environment.
    path = Path(value).expanduser() if value else Path(default)
    if not path.is_absolute():
        path = ROOT / path
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def megasam_home() -> Path:
    if not MEGASAM_HOME.is_dir():
        raise FileNotFoundError(
            f"MegaSAM is not installed at {MEGASAM_HOME}; run environments/install_extensions.sh"
        )
    return MEGASAM_HOME


def cudnn_lib() -> Path:
    if not (CUDNN_LIB / "libcudnn_graph.so.9.13.1").exists():
        raise FileNotFoundError(
            f"cuDNN 9.13.1 is not installed at {CUDNN_LIB}; run environments/install_extensions.sh"
        )
    return CUDNN_LIB


def use_checkout_caches() -> None:
    """Keep every download inside this checkout, and let it happen.

    HF_HOME, HF_HUB_CACHE, and TORCH_HOME point into ./models/cache for this
    process and every subprocess it starts, whatever the shell set, so a run
    never reads or fills another checkout's or the machine's cache. Offline
    mode is lifted, since that cache starts empty.
    """
    hf_home = MODEL_CACHE / "huggingface"
    os.environ.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "TORCH_HOME": str(MODEL_CACHE / "torch"),
        }
    )
    for name in (
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        os.environ.pop(name, None)


def load_component_environment(root: Path | None = None) -> Path | None:
    """Apply the checkout's untracked ``components.env``, if there is one.

    The COMPONENT_SETTINGS come from this file or not at all: whatever the
    shell exported under those names is dropped first.
    """
    for name in COMPONENT_SETTINGS:
        os.environ.pop(name, None)
    path = (root or ROOT) / COMPONENT_ENV_FILE
    if not path.is_file():
        return None
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator:
            raise ValueError(f"{path}:{number} is not NAME=value")
        if name not in COMPONENT_SETTINGS:
            raise ValueError(
                f"{path}:{number} sets {name}, but {COMPONENT_ENV_FILE} can set only "
                + ", ".join(COMPONENT_SETTINGS)
            )
        os.environ[name] = value.strip().strip("'\"")
    return path

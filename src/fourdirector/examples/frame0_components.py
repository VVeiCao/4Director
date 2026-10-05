"""Where each Frame-0 step finds its worker and weights, and the argv that runs it.

Every model runs as a subprocess of this same environment, so the app process
never imports one. This module resolves what each step needs, fails closed
when something is missing, and builds the argv, keeping that detail out of the
workflow itself.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fourdirector import paths
from fourdirector.paths import required_path
from fourdirector.stages.external import ExternalCommandRunner
from fourdirector.stages.static_megasam import (
    LockedEntrypoint,
    StaticMegaSAMConfig,
    StaticMegaSAMProvider,
)

WORKERS = Path(__file__).resolve().parents[1] / "workers"
CAPTION_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# The revision of CAPTION_MODEL this release was validated with.
CAPTION_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
DEFAULT_CAPTION_SEED = 42
MOGE2_SCRIPT = paths.script("moge2_worker")


def sam2_checkpoint() -> Path:
    return required_path("FOURDIRECTOR_SAM2_CHECKPOINT", default=paths.SAM2_CHECKPOINT)


def motion_adapter_checkpoint() -> Path:
    return required_path("FOURDIRECTOR_CHECKPOINT", default=paths.MOTION_ADAPTER_CHECKPOINT)


def unidepth_root() -> Path:
    """UniDepth as MegaSAM vendors it, in the copy install_extensions.sh patched."""
    return paths.megasam_home() / "UniDepth"


def wan_store_complete(root: Path) -> bool:
    """Whether WAN_ROOT holds the index and every shard it lists, as stage 4 needs."""
    model = root / "Wan-AI" / "Wan2.1-VACE-14B"
    index = model / "diffusion_pytorch_model.safetensors.index.json"
    if not index.is_file():
        return False
    shards = set(json.loads(index.read_text())["weight_map"].values())
    return all((model / shard).is_file() for shard in shards)


def _wan_store() -> Path:
    root = required_path("WAN_ROOT", directory=True, default=paths.MODELS)
    if not wan_store_complete(root):
        raise FileNotFoundError(f"WAN_ROOT has no complete Wan2.1-VACE-14B: {root}")
    return root


# What each step needs besides the environment, named the way the startup
# report shows it: the variable that moves it, or the step that installs it.
REQUIREMENTS: dict[str, tuple[tuple[str, Callable[[], Path]], ...]] = {
    "Generate mask": (("FOURDIRECTOR_SAM2_CHECKPOINT", sam2_checkpoint),),
    "Generate 3D": (("UniDepth (environments/install_extensions.sh)", unidepth_root),),
    "Printed commands": (
        ("FOURDIRECTOR_CHECKPOINT", motion_adapter_checkpoint),
        ("WAN_ROOT", _wan_store),
        ("cuDNN 9.13.1 (environments/install_extensions.sh)", paths.cudnn_lib),
    ),
}


def missing_requirements() -> dict[str, tuple[str, ...]]:
    """Report which steps cannot run yet, so the UI can say so up front."""
    missing: dict[str, tuple[str, ...]] = {}
    for step, requirements in REQUIREMENTS.items():
        unusable = []
        for label, locate in requirements:
            try:
                locate()
            except (RuntimeError, FileNotFoundError):
                unusable.append(label)
        if unusable:
            missing[step] = tuple(unusable)
    return missing


def sam2_command(*, image: Path, clicks_json: Path, output: Path) -> tuple[list[str], Path]:
    """Build the SAM2 argv and the directory it runs from."""
    argv = [
        sys.executable,
        str(WORKERS / "sam2_single_image.py"),
        "--checkpoint", str(sam2_checkpoint()),
        "--image", str(image),
        "--clicks-json", str(clicks_json),
        "--output", str(output),
    ]
    return argv, WORKERS


def caption_command(*, image: Path, output: Path) -> list[str]:
    """Build the captioning argv. Qwen3-VL samples, so it always gets a seed."""
    return [
        sys.executable,
        str(WORKERS / "qwen3_vl_caption.py"),
        "--image", str(image),
        "--output-json", str(output),
        "--model", CAPTION_MODEL,
        "--revision", CAPTION_REVISION,
        "--seed", os.environ.get("FOURDIRECTOR_QWEN3_VL_SEED", str(DEFAULT_CAPTION_SEED)),
    ]


def static_geometry_provider(runner: ExternalCommandRunner) -> StaticMegaSAMProvider:
    """Build the MoGe-2 + UniDepth provider for the single-image scene."""
    unidepth = unidepth_root()
    return StaticMegaSAMProvider(
        StaticMegaSAMConfig(
            moge2=LockedEntrypoint.lock(
                python_executable=Path(sys.executable),
                script=MOGE2_SCRIPT,
            ),
            unidepth_v2=LockedEntrypoint.lock(
                python_executable=Path(sys.executable),
                script=unidepth / "scripts" / "demo_mega-sam.py",
            ),
            moge2_working_directory=MOGE2_SCRIPT.parent,
            unidepth_v2_working_directory=unidepth,
        ),
        runner=runner,
    )


def caption_text(payload: Any) -> str:
    """Normalize whatever a caption source returned into one non-empty string."""
    text = payload.get("text", "") if isinstance(payload, dict) else payload
    text = str(text).strip()
    if not text:
        raise RuntimeError("caption generator returned empty text")
    return text

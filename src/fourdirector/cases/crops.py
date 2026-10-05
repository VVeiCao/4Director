"""Real padded RGBA object crops and the correspondence Pixal3D command."""

from __future__ import annotations

import importlib.util
import math
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

from fourdirector import paths

GENERATOR = paths.script("pixal3d_generator")
ADAPTER = paths.script("pixal3d_adapter")
ALIGNER = paths.script("aligner")


def alpha_bbox(mask: np.ndarray, padding: float = 1.1) -> tuple[float, float, float, float]:
    yx = np.argwhere(mask)
    if not len(yx):
        raise ValueError("object mask is empty")
    y0, x0 = yx.min(axis=0)
    y1, x1 = yx.max(axis=0)
    center_x = (float(x0) + float(x1)) * 0.5
    center_y = (float(y0) + float(y1)) * 0.5
    size = max(2, int(max(float(x1 - x0), float(y1 - y0)) * padding))
    half = size // 2
    return (
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
    )


def write_padded_rgba(
    image: Path,
    mask: Path,
    destination: Path,
    *,
    padding: float = 1.1,
) -> dict[str, float]:
    rgb = np.asarray(Image.open(image).convert("RGB"))
    alpha = np.asarray(Image.open(mask).convert("L"))
    if rgb.shape[:2] != alpha.shape:
        raise ValueError(f"image/mask size mismatch: {rgb.shape} vs {alpha.shape}")
    binary = alpha > 0
    bbox = alpha_bbox(binary, padding=padding)
    rgba = np.concatenate([rgb, (binary.astype(np.uint8) * 255)[..., None]], axis=-1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(destination)
    return {
        "x0": bbox[0],
        "y0": bbox[1],
        "x1": bbox[2],
        "y1": bbox[3],
        "padding": padding,
    }


@cache
def _pinned_pixal3d_commit() -> str:
    spec = importlib.util.spec_from_file_location("pixal3d_run_patched", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PIXAL3D_COMMIT


def holds_pinned_pixal3d(repository: Path) -> bool:
    """Whether a Pixal3D checkout contains the commit the adapter patches."""
    repository = repository.expanduser()
    if not repository.is_dir():
        return False
    probe = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{_pinned_pixal3d_commit()}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def correspondence_command(
    *,
    python: Path,
    repository: Path,
    image: Path,
    mesh: Path,
    correspondence: Path,
    preprocessed: Path,
    fov_x: float,
    seed: int,
    decimation_target: int = 100000,
    texture_size: int = 1024,
    low_vram: bool = False,
) -> list[str]:
    if not (Path(repository) / "pixal3d").is_dir():
        raise FileNotFoundError(
            f"Pixal3D is missing at {repository}; fetch the submodules with "
            "`git submodule update --init --recursive`"
        )
    if holds_pinned_pixal3d(Path(repository)):
        # Run a patched copy of the pinned commit, whatever state the checkout's
        # own work tree is in; the adapter hands the generator its location
        # through PIXAL3D_ROOT.
        prefix = [
            str(python), str(ADAPTER), "--repository", str(repository), "--",
            str(python), str(GENERATOR),
        ]
    else:
        prefix = [str(python), str(GENERATOR), "--repository", str(repository)]
    command = [
        *prefix,
        "--image",
        str(image),
        "--output",
        str(mesh),
        "--correspondence",
        str(correspondence),
        "--preprocessed",
        str(preprocessed),
        "--fov",
        str(fov_x),
        "--model-path",
        "TencentARC/Pixal3D",
        "--seed",
        str(seed),
        "--resolution",
        "1024",
        "--correspondence-resolution",
        "512",
        "--decimation-target",
        str(decimation_target),
        "--texture-size",
        str(texture_size),
        "--crop-padding",
        "1.1",
    ]
    if low_vram:
        command.append("--low-vram")
    return command


def crop_fov_x(bbox: dict[str, float], focal: float) -> float:
    crop_width = max(float(bbox["x1"] - bbox["x0"]), 1.0)
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError(f"invalid focal length {focal}")
    return float(2.0 * math.atan(crop_width / (2.0 * focal)))


def alignment_command(
    *,
    python: Path,
    mesh: Path,
    correspondence: Path,
    camera_contract: Path,
    target: Path,
    motion: Path,
    output: Path,
    erosion_pixels: int,
) -> list[str]:
    return [
        str(python),
        str(ALIGNER),
        "--mesh",
        str(mesh),
        "--correspondence",
        str(correspondence),
        "--camera-contract",
        str(camera_contract),
        "--target",
        str(target),
        "--motion",
        str(motion),
        "--output",
        str(output),
        "--reference-frame",
        "0",
        "--erosion-pixels",
        str(erosion_pixels),
        "--median-threshold",
        "0.15",
        "--p90-threshold",
        "0.30",
        "--p95-threshold",
        "0.50",
    ]

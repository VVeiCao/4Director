#!/usr/bin/env python3
"""Download the model weights 4Director runs with into ./models.

    python download_models.py

- SAM 2 hiera-large, for the app's mask, from Meta's release URL.
- Wan2.1-VACE-14B, for stage 4, from Hugging Face; its google/umt5-xxl
  tokenizer comes with it.
- The step-2610 Motion Adapter, from Hugging Face (vveicao/4Director).

Every stage looks in ./models by default. MoGe-2, UniDepth, Qwen3-VL, and
Pixal3D download themselves into ./models/cache the first time they run, and
MegaSAM with its RAFT weights comes from environments/install_extensions.sh.
A file that is already present with the right SHA-256 is not fetched again.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fourdirector import paths  # noqa: E402
from fourdirector.constants import FULL_VACE_SHA256  # noqa: E402

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAM2_SHA256 = "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"
WAN_REPOSITORY = "Wan-AI/Wan2.1-VACE-14B"
# The revision the teaser videos were reproduced with.
WAN_REVISION = "539c162b1387eac9dc4c20bd3f74671309e76a4c"
WAN_FILES = [
    "config.json",
    "diffusion_pytorch_model-*.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "google/umt5-xxl/*",
]
CHECKPOINT_REPOSITORY = "vveicao/4Director"
CHECKPOINT_FILE = "step-2610.safetensors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, sha256: str) -> bool:
    return path.is_file() and sha256_file(path) == sha256


def download_sam2(destination: Path) -> None:
    if verified(destination, SAM2_SHA256):
        print(f"SAM2 already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    print(f"Downloading SAM2 hiera-large to {destination}")
    with urllib.request.urlopen(SAM2_URL) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if sha256_file(partial) != SAM2_SHA256:
        partial.unlink()
        raise RuntimeError(f"{SAM2_URL} did not match SHA-256 {SAM2_SHA256}; rerun to fetch it again")
    partial.replace(destination)


def download_wan(models: Path) -> None:
    from huggingface_hub import snapshot_download

    target = models / "Wan-AI" / "Wan2.1-VACE-14B"
    print(f"Downloading {WAN_REPOSITORY} to {target} (about 75 GB)")
    snapshot_download(WAN_REPOSITORY, revision=WAN_REVISION, local_dir=target, allow_patterns=WAN_FILES)


def download_checkpoint(repository: str, destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    if verified(destination, FULL_VACE_SHA256):
        print(f"Motion Adapter already present: {destination}")
        return
    print(f"Downloading the Motion Adapter from {repository} to {destination}")
    hf_hub_download(repository, CHECKPOINT_FILE, local_dir=destination.parent)
    if sha256_file(destination) != FULL_VACE_SHA256:
        raise RuntimeError(
            f"{destination} is not the released step-2610 Motion Adapter; delete it and rerun"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-repo",
        default=CHECKPOINT_REPOSITORY,
        help=f"Hugging Face repository that holds {CHECKPOINT_FILE} (default: %(default)s)",
    )
    parser.add_argument("--skip-wan", action="store_true", help="keep an existing Wan store")
    args = parser.parse_args()

    paths.use_checkout_caches()
    download_sam2(paths.SAM2_CHECKPOINT)
    if not args.skip_wan:
        download_wan(paths.MODELS)
    download_checkpoint(args.checkpoint_repo, paths.MOTION_ADAPTER_CHECKPOINT)
    print(f"Models are under {paths.MODELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

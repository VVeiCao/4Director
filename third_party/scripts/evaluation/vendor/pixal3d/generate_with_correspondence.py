#!/usr/bin/env python3
"""Generate one Pixal3D GLB and its pre-export visible-surface map."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=os.environ.get("PIXAL3D_ROOT"),
        help="Pixal3D checkout (default: $PIXAL3D_ROOT, set by pixal3d_adapter/run_patched.py).",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--fov", type=float, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--correspondence-resolution", type=int, default=512)
    parser.add_argument("--decimation-target", type=int, default=100000)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--crop-padding", type=float, default=1.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repository is None:
        raise SystemExit("pass --repository or run under pixal3d_adapter/run_patched.py")
    repository = Path(args.repository).expanduser().resolve()
    adapter = Path(__file__).resolve().parents[4] / "pixal3d_adapter"
    # Pixal3D was written against utils3d 0.0.2; the environment's own utils3d is
    # the newer one MoGe-2 needs. install_extensions.sh sets Pixal3D's copy aside
    # here, and only this process looks there first.
    legacy_utils3d = Path(sys.prefix) / "opt" / "pixal3d-utils3d"
    if legacy_utils3d.is_dir():
        sys.path.insert(0, str(legacy_utils3d))
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(adapter.parent))
    os.chdir(repository)
    from PIL import Image
    import numpy as np

    from pixal3d_adapter.batch_davis_hike_sequence import (
        alpha_bbox,
        build_pipeline,
        run_frame,
    )

    pipeline = build_pipeline(args.model_path, args.low_vram)
    image = np.asarray(Image.open(args.image).convert("RGBA"))
    crop_bbox = alpha_bbox(image[..., 3], args.crop_padding)
    result = run_frame(
        pipeline=pipeline,
        image_path=args.image.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        fov_x=args.fov,
        seed=args.seed,
        image_resolution=512,
        resolution=args.resolution,
        max_num_tokens=49152,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=False,
        preprocessed_path=args.preprocessed.expanduser().resolve(),
        correspondence_path=args.correspondence.expanduser().resolve(),
        correspondence_resolution=args.correspondence_resolution,
        crop_bbox=crop_bbox,
    )
    if not args.output.is_file() or not args.correspondence.is_file():
        raise RuntimeError(f"Incomplete Pixal3D output: {result}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write MoGe-2 inverse-depth priors in MegaSAM's per-frame NPY contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model import import_model_class_by_version


MOGE2_REPOSITORY = "Ruicheng/moge-2-vitl-normal"
# The revision of MOGE2_REPOSITORY this release was validated with.
MOGE2_REVISION = "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"
DEFAULT_MODEL = os.environ.get("MOGE2_MODEL", MOGE2_REPOSITORY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--fov_x", type=float)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resolution-level", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        [
            *args.img_path.glob("*.jpg"),
            *args.img_path.glob("*.jpeg"),
            *args.img_path.glob("*.png"),
        ]
    )
    if not paths:
        raise FileNotFoundError(f"No image frames under {args.img_path}")
    device = torch.device("cuda")
    model = (
        import_model_class_by_version("v2")
        .from_pretrained(
            args.model,
            **({"revision": MOGE2_REVISION} if args.model == MOGE2_REPOSITORY else {}),
        )
        .to(device)
        .eval()
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    valid_fractions = []
    inferred_fovs = []
    for index, path in enumerate(paths):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Unable to decode {path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .to(device=device, dtype=torch.float32)
            / 255.0
        )
        output = model.infer(
            tensor,
            fov_x=args.fov_x,
            resolution_level=args.resolution_level,
            use_fp16=True,
        )
        depth = output["depth"].detach().float().cpu().numpy()
        mask = output["mask"].detach().cpu().numpy().astype(bool)
        valid = mask & np.isfinite(depth) & (depth > 0)
        inverse_depth = np.zeros(depth.shape, dtype=np.float32)
        inverse_depth[valid] = 1.0 / depth[valid]
        if int(valid.sum()) < 100:
            raise ValueError(f"{path}: MoGe-2 produced too little valid depth")
        np.save(args.outdir / f"{path.stem}.npy", inverse_depth)
        intrinsics = output["intrinsics"].detach().float().cpu().numpy()
        inferred_fov = float(
            np.degrees(2.0 * np.arctan(1.0 / (2.0 * intrinsics[0, 0])))
        )
        inferred_fovs.append(inferred_fov)
        valid_fractions.append(float(valid.mean()))
        print(
            json.dumps(
                {
                    "frame": index,
                    "path": str(path),
                    "valid_fraction": valid_fractions[-1],
                    "inferred_fov_x_degrees": inferred_fov,
                }
            ),
            flush=True,
        )
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "4director-megasam-moge2-prior-v1",
                "model": args.model,
                "frames": len(paths),
                "provided_fov_x_degrees": args.fov_x,
                "median_inferred_fov_x_degrees": float(
                    np.median(inferred_fovs)
                ),
                "minimum_valid_fraction": min(valid_fractions),
                "output": "inverse depth float32 NPY; invalid=0",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

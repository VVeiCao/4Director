#!/usr/bin/env python3
"""Run SAM2 on one image using positive and negative point prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    # SAM 2 hiera-large, the model download_models.py fetches.
    parser.add_argument("--model-config", default="configs/sam2/sam2_hiera_l.yaml")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--clicks-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.image, args.clicks_json):
        if not path.exists():
            raise FileNotFoundError(path)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SAM2 requested CUDA, but torch.cuda.is_available() is false")
    payload = json.loads(args.clicks_json.read_text(encoding="utf-8"))
    clicks = payload.get("clicks", payload)
    if not clicks or not any(int(item.get("label", 1)) > 0 for item in clicks):
        raise ValueError("SAM2 requires at least one positive click")
    points = np.asarray([[item["x"], item["y"]] for item in clicks], dtype=np.float32)
    labels = np.asarray([item.get("label", 1) for item in clicks], dtype=np.int32)
    image = np.asarray(Image.open(args.image).convert("RGB"))
    model = build_sam2(
        str(args.model_config),
        str(args.checkpoint),
        device=str(args.device),
    )
    predictor = SAM2ImagePredictor(model)
    with torch.inference_mode():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
    index = int(np.argmax(scores))
    mask = np.asarray(masks[index], dtype=bool)
    if not mask.any():
        raise RuntimeError("SAM2 returned an empty mask")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(args.output)
    print(json.dumps({"score": float(scores[index]), "foreground_pixels": int(mask.sum())}))


if __name__ == "__main__":
    main()

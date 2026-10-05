"""Hash a depth control by its decoded frames."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from fourdirector.constants import FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH


def decoded_gray_identity(video: Path) -> dict[str, Any]:
    """Hash canonical decoded gray8 frames, independent of MKV container bytes."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode control video: {video}")
    digest = hashlib.sha256()
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise ValueError(
                    f"control frame has shape {frame.shape}, expected "
                    f"{(FRAME_HEIGHT, FRAME_WIDTH, 3)}"
                )
            if not (
                np.array_equal(frame[..., 0], frame[..., 1])
                and np.array_equal(frame[..., 0], frame[..., 2])
            ):
                raise ValueError("control video is not lossless grayscale")
            digest.update(np.ascontiguousarray(frame[..., 0]).tobytes())
            frame_count += 1
    finally:
        capture.release()
    if frame_count != FRAME_COUNT:
        raise ValueError(
            f"control decoded {frame_count} frames, expected {FRAME_COUNT}"
        )
    return {
        "frames": frame_count,
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "format": "gray8",
        "sha256": digest.hexdigest(),
    }

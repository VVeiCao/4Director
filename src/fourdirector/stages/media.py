"""The fixed 81-frame, 832x480, 16 fps media contract every stage shares."""

from __future__ import annotations

from dataclasses import dataclass

from fourdirector.constants import FPS, FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH


@dataclass(frozen=True)
class MediaContract:
    num_frames: int = FRAME_COUNT
    width: int = FRAME_WIDTH
    height: int = FRAME_HEIGHT
    fps: int = FPS

    def __post_init__(self) -> None:
        if (self.num_frames, self.width, self.height, self.fps) != (
            FRAME_COUNT,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            FPS,
        ):
            raise ValueError(
                f"production media contract is fixed at "
                f"{FRAME_COUNT}/{FRAME_WIDTH}x{FRAME_HEIGHT}/{FPS}"
            )


PRODUCTION_MEDIA = MediaContract()

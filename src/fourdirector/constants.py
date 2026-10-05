"""Fixed media contract and the released Motion Adapter identity."""

from __future__ import annotations

FRAME_COUNT = 81
FRAME_WIDTH = 832
FRAME_HEIGHT = 480
FPS = 16

# SHA-256 of the released Full-VACE Motion Adapter weights (step 2610).
FULL_VACE_SHA256 = "91c56861c58dfa618b358d80093bee8561c082844fd215c968b00eadb132c9e3"

__all__ = ["FPS", "FRAME_COUNT", "FRAME_HEIGHT", "FRAME_WIDTH", "FULL_VACE_SHA256"]

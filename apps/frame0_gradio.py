#!/usr/bin/env python3
"""Checkout-local wrapper for the Frame-0 Gradio gallery."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fourdirector.apps.frame0_gradio import main  # noqa: E402

if __name__ == "__main__":
    main()

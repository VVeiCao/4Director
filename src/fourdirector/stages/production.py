"""Build the video MegaSAM provider that stage 1 uses for a custom Case."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fourdirector import paths

from .megasam import (
    VIDEO_COMPONENTS,
    ComponentEntrypointLock,
    VideoMegaSAMConfig,
    VideoMegaSAMProvider,
    sha256_file,
)

ORCHESTRATOR = paths.script("megasam_orchestrator")
# The orchestrator runs the MoGe-2 worker that sits beside it.
MOGE2_SCRIPT = paths.script("moge2_worker")


def megasam_repository() -> Path:
    """The MegaSAM checkout: camera tracking, RAFT, CVD, and its vendored UniDepth."""
    return paths.megasam_home()


def create_video_megasam_provider() -> VideoMegaSAMProvider:
    """Build MoGe-2 + UniDepth V2 + MegaSAM/DROID + RAFT + CVD, all run by this
    environment's interpreter."""
    megasam_python = moge2_python = Path(sys.executable)
    root = megasam_repository()
    component_paths = {
        "moge2": MOGE2_SCRIPT,
        "unidepth_v2": root / "UniDepth" / "scripts" / "demo_mega-sam.py",
        "megasam_droid": root / "camera_tracking_scripts" / "test_demo.py",
        "raft": root / "cvd_opt" / "preprocess_flow.py",
        "cvd": root / "cvd_opt" / "cvd_opt.py",
    }
    missing = [path for path in component_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MegaSAM entrypoints are missing: {missing}")
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    config = VideoMegaSAMConfig(
        python_executable=megasam_python,
        orchestrator_script=ORCHESTRATOR,
        orchestrator_sha256=sha256_file(ORCHESTRATOR),
        component_entrypoints=tuple(
            ComponentEntrypointLock(
                component=component,
                path=component_paths[component],
                sha256=sha256_file(component_paths[component]),
            )
            for component in VIDEO_COMPONENTS
        ),
        environment=tuple(
            sorted(
                {
                    "MEGASAM_ROOT": str(root),
                    "MEGASAM_PYTHON": str(megasam_python),
                    "MOGE2_PYTHON": str(moge2_python),
                    "MOGE2_HF_HOME": hf_home,
                    "TMPDIR": os.environ.get("TMPDIR", tempfile.gettempdir()),
                }.items()
            )
        ),
    )
    config.verify()
    return VideoMegaSAMProvider(config)

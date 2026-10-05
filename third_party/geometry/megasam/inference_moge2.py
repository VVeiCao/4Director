#!/usr/bin/env python3
"""MegaSAM orchestration with MoGe-2 mono prior and UniDepth V2 metric prior."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


# Set by fourdirector.stages.production from components.env.
MEGASAM_ROOT = Path(os.environ["MEGASAM_ROOT"])
MEGASAM_PYTHON = Path(os.environ["MEGASAM_PYTHON"])
MOGE2_PYTHON = Path(os.environ["MOGE2_PYTHON"])
MOGE2_SCRIPT = Path(__file__).with_name("moge2_infer_video.py")
# Lets UniDepth import the xformers NystromAttention newer xformers dropped.
UNIDEPTH_COMPAT = Path(__file__).with_name("compat")
MOGE2_HF_HOME = Path(os.environ["MOGE2_HF_HOME"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--fov_x", type=float)
    parser.add_argument("--depth_model", default="moge")
    parser.add_argument("--resolution", type=int, default=384 * 512)
    return parser.parse_args()


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    pythonpath: Path,
    environment_overrides: dict[str, str] | None = None,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(pythonpath)
    environment.update(environment_overrides or {})
    print(json.dumps({"command": command, "cwd": str(cwd)}), flush=True)
    subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
    )


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    checkpoints = MEGASAM_ROOT / "checkpoints"
    raft_checkpoint = MEGASAM_ROOT / "cvd_opt" / "raft-things.pth"
    required = [
        checkpoints / "megasam_final.pth",
        raft_checkpoint,
        MOGE2_SCRIPT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MegaSAM/MoGe-2 assets: {missing}")

    with TemporaryDirectory(
        dir=os.environ.get("TMPDIR"),
        prefix="4director-megasam-moge2-",
    ) as temporary_name:
        root = Path(temporary_name)
        image_root = root / "images" / "test"
        image_root.parent.mkdir(parents=True)
        shutil.copytree(args.input_dir, image_root)

        moge_command = [
            str(MOGE2_PYTHON),
            str(MOGE2_SCRIPT),
            "--img-path",
            "images/test",
            "--outdir",
            "MoGe2/video_visualization/test",
        ]
        if args.fov_x is not None:
            moge_command.extend(["--fov_x", str(args.fov_x)])
        run_checked(
            moge_command,
            cwd=root,
            pythonpath=MOGE2_SCRIPT.parent,
            environment_overrides={
                "HF_HOME": str(MOGE2_HF_HOME),
                "HF_HUB_CACHE": str(MOGE2_HF_HOME / "hub"),
                "HUGGINGFACE_HUB_CACHE": str(MOGE2_HF_HOME / "hub"),
            },
        )

        unidepth_root = MEGASAM_ROOT / "UniDepth"
        run_checked(
            [
                str(MEGASAM_PYTHON),
                str(unidepth_root / "scripts" / "demo_mega-sam.py"),
                "--scene-name",
                "test",
                "--img-path",
                "images/test",
                "--outdir",
                "UniDepth/outputs",
            ],
            cwd=root,
            pythonpath=f"{UNIDEPTH_COMPAT}{os.pathsep}{unidepth_root}",
        )

        camera_command = [
            str(MEGASAM_PYTHON),
            str(
                MEGASAM_ROOT
                / "camera_tracking_scripts"
                / "test_demo.py"
            ),
            "--datapath=images/test",
            f"--weights={checkpoints / 'megasam_final.pth'}",
            "--scene_name=test",
            "--mono_depth_path=MoGe2/video_visualization",
            "--metric_depth_path=UniDepth/outputs",
            "--depth_for_cvd=MoGe2/video_visualization",
            "--disable_vis",
            f"--resolution={args.resolution}",
        ]
        if args.fov_x is not None:
            camera_command.append(f"--fov_x={args.fov_x}")
        run_checked(
            camera_command,
            cwd=root,
            pythonpath=MEGASAM_ROOT,
        )
        run_checked(
            [
                str(MEGASAM_PYTHON),
                str(MEGASAM_ROOT / "cvd_opt" / "preprocess_flow.py"),
                "--datapath=images/test",
                f"--model={raft_checkpoint}",
                "--scene_name=test",
                "--mixed_precision",
                f"--resolution={args.resolution}",
            ],
            cwd=root,
            pythonpath=MEGASAM_ROOT,
        )
        run_checked(
            [
                str(MEGASAM_PYTHON),
                str(MEGASAM_ROOT / "cvd_opt" / "cvd_opt.py"),
                "--scene_name",
                "test",
                "--w_grad",
                "2.0",
                "--w_normal",
                "5.0",
                "--freeze_shift",
            ],
            cwd=root,
            pythonpath=MEGASAM_ROOT,
        )
        result = root / "outputs_cvd" / "test_sgd_cvd_hr.npz"
        if not result.is_file():
            raise RuntimeError(f"MegaSAM did not produce {result}")
        with np.load(result) as archive:
            required_keys = {
                "images",
                "depths",
                "depths_raw",
                "intrinsic",
                "cam_c2w",
            }
            absent = required_keys.difference(archive.files)
            if absent:
                raise ValueError(
                    f"MegaSAM result missing keys: {sorted(absent)}"
                )
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output_path.with_name(
            f".{args.output_path.name}.tmp.{os.getpid()}"
        )
        shutil.copyfile(result, temporary_output)
        os.replace(temporary_output, args.output_path)


if __name__ == "__main__":
    main()

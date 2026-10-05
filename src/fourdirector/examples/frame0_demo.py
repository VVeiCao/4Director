"""Persistent single-upload workflow for the Frame-0 demo."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fourdirector import paths
from fourdirector.cases.schema import load_case_yaml
from fourdirector.cases.stages import _background_from_geometry, stage2_objects
from fourdirector.constants import FPS, FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH
from fourdirector.examples.frame0_components import (
    caption_command,
    caption_text,
    motion_adapter_checkpoint,
    sam2_command,
    static_geometry_provider,
)
from fourdirector.stages.external import ExternalCommandRunner

SCHEMA = "4director-frame0-upload-v1"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a receipt in one step, so a crash cannot leave it half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_component_environment(research_root: Path | None = None) -> Path | None:
    """Apply the checkout's ``components.env``, if there is one."""
    return paths.load_component_environment(research_root)


def _load_image(image: Any) -> Image.Image:
    """Return the upload as an in-memory RGB image, closing any file it came from."""
    if isinstance(image, (str, os.PathLike)):
        with Image.open(Path(image).expanduser()) as source:
            return source.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    raise TypeError("image must be a path, PIL image, or NumPy array")


def upload_size_error(size: tuple[int, int]) -> str | None:
    """Explain why an upload of this size is refused, or return None.

    Every stage works at exactly 832x480; resizing a different image here
    would silently stretch it, so the app asks for the right size instead.
    """
    width, height = size
    if (width, height) == (FRAME_WIDTH, FRAME_HEIGHT):
        return None
    return (
        f"Images must be exactly {FRAME_WIDTH}×{FRAME_HEIGHT} pixels; "
        f"this one is {width}×{height}. Crop or resize it to {FRAME_WIDTH}×{FRAME_HEIGHT} "
        "and upload it again."
    )


class Frame0DemoWorkflow:
    """Manage one object in each durable upload run."""

    def __init__(
        self,
        *,
        research_root: Path | None = None,
        state_root: Path | None = None,
        runner: ExternalCommandRunner | None = None,
        preset_builder: Callable[..., Any] | None = None,
    ) -> None:
        self.research_root = (research_root or paths.ROOT).expanduser().resolve()
        self.state_root = (
            state_root or self.research_root / "output/frame0-demo"
        ).expanduser().resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.active_path = self.state_root / "active-run.json"
        self.runner = runner or ExternalCommandRunner()
        self.preset_builder = preset_builder
        example = self.research_root / "data/examples/dog.png"
        self.example_image = example if example.is_file() else None

    @property
    def gradio_allowed_paths(self) -> tuple[str, ...]:
        allowed = [str(self.state_root)]
        if self.example_image is not None:
            allowed.append(str(self.example_image.parent))
        return tuple(allowed)

    def _root(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(str(run_id)):
            raise ValueError("run_id must contain only letters, digits, dot, underscore, or dash")
        root = (self.state_root / str(run_id)).resolve()
        if root.parent != self.state_root:
            raise ValueError("run_id escapes the Frame-0 output root")
        if not root.is_dir():
            raise FileNotFoundError(f"unknown Frame-0 run: {run_id}")
        return root

    def _receipt(self, run_id: str) -> dict[str, Any]:
        return json.loads((self._root(run_id) / "receipt.json").read_text(encoding="utf-8"))

    def receipt(self, run_id: str) -> dict[str, Any]:
        """Return a stored run's receipt without making it the active run."""
        return self._receipt(run_id)

    def _write_receipt(self, run_id: str, **updates: Any) -> dict[str, Any]:
        receipt = self._receipt(run_id)
        receipt.update(updates)
        receipt["updated_ns"] = time.time_ns()
        _atomic_json(self._root(run_id) / "receipt.json", receipt)
        _atomic_json(self.active_path, receipt)
        return receipt

    def active_run_receipt(self) -> dict[str, Any] | None:
        """Return the active run, or nothing if that run has since been deleted.

        The pointer outlives the run it names, so a deleted run must read as no
        active run rather than taking the app down at startup.
        """
        if not self.active_path.is_file():
            return None
        receipt = json.loads(self.active_path.read_text(encoding="utf-8"))
        run_id = str(receipt.get("run_id", ""))
        if not run_id or not (self.state_root / run_id / "receipt.json").is_file():
            return None
        return receipt

    def list_runs(self) -> list[dict[str, Any]]:
        """Summarize every stored run so a session can be resumed later."""
        runs = []
        for path in sorted(self.state_root.iterdir(), reverse=True):
            receipt_path = path / "receipt.json"
            if not path.is_dir() or not receipt_path.is_file():
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            runs.append(
                {
                    "run_id": str(receipt.get("run_id", path.name)),
                    "status": str(receipt.get("status", "unknown")),
                    "has_mesh": bool(receipt.get("mesh")),
                    "preset": receipt.get("preset"),
                    "prompt": str(receipt.get("prompt", "")),
                }
            )
        return runs

    def activate_run(self, run_id: str) -> dict[str, Any]:
        """Make a stored run current again without touching its files."""
        receipt = self._receipt(run_id)
        _atomic_json(self.active_path, receipt)
        return receipt

    def create_run(self, image: Any, run_id: str | None = None) -> dict[str, Any]:
        pixels = _load_image(image)
        error = upload_size_error(pixels.size)
        if error:
            raise ValueError(error)
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, digits, dot, underscore, or dash")
        root = self.state_root / run_id
        if root.exists():
            raise FileExistsError(f"Frame-0 run already exists: {run_id}")
        image_path = root / "inputs/image.png"
        image_path.parent.mkdir(parents=True)
        pixels.save(image_path)
        receipt = {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "root": str(root),
            "status": "uploaded",
            "image": str(image_path),
            "mask": None,
            "mask_confirmed": False,
            "mesh": None,
            "prompt": "",
            "clicks": [],
            "object_id": "object",
            "frame_count": FRAME_COUNT,
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": FPS,
            "created_ns": time.time_ns(),
        }
        _atomic_json(root / "receipt.json", receipt)
        _atomic_json(self.active_path, receipt)
        return receipt

    def add_click(self, run_id: str, x: int, y: int, label: int = 1) -> dict[str, Any]:
        if not (0 <= int(x) < FRAME_WIDTH and 0 <= int(y) < FRAME_HEIGHT):
            raise ValueError(f"click must be inside {FRAME_WIDTH}x{FRAME_HEIGHT}")
        if int(label) not in (0, 1):
            raise ValueError("click label must be 0 or 1")
        receipt = self._receipt(run_id)
        clicks = [*receipt["clicks"], {"x": int(x), "y": int(y), "label": int(label)}]
        return self._write_receipt(run_id, clicks=clicks, mask_confirmed=False, status="clicked")

    def undo_click(self, run_id: str) -> dict[str, Any]:
        clicks = list(self._receipt(run_id)["clicks"])
        if clicks:
            clicks.pop()
        return self._write_receipt(run_id, clicks=clicks, mask_confirmed=False, status="clicked")

    def clear_clicks(self, run_id: str) -> dict[str, Any]:
        return self._write_receipt(
            run_id, clicks=[], mask=None, mask_confirmed=False, status="uploaded"
        )

    def _production_mask(self, receipt: dict[str, Any], output: Path) -> dict[str, Any]:
        clicks_path = output.with_suffix(".clicks.json")
        _atomic_json(clicks_path, {"clicks": receipt["clicks"]})
        argv, repository = sam2_command(
            image=Path(receipt["image"]),
            clicks_json=clicks_path,
            output=output,
        )
        result = self.runner.run(argv, cwd=repository)
        return {"command": argv, "stdout": result.stdout.strip()}

    def generate_mask(
        self,
        run_id: str,
        mask_fn: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        receipt = self._receipt(run_id)
        if not any(int(click["label"]) > 0 for click in receipt["clicks"]):
            raise ValueError("At least one positive click is required to generate a mask")
        output = self._root(run_id) / "inputs/objects/object/mask.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        details = (
            mask_fn(
                image=Path(receipt["image"]),
                clicks=list(receipt["clicks"]),
                output=output,
            )
            if mask_fn is not None
            else self._production_mask(receipt, output)
        )
        if isinstance(details, (Image.Image, np.ndarray)):
            Image.fromarray(np.asarray(details).astype(np.uint8)).convert("L").save(output)
        elif isinstance(details, (str, os.PathLike)):
            Image.open(details).convert("L").save(output)
        if not output.is_file():
            raise RuntimeError(f"mask generator did not create {output}")
        mask = Image.open(output).convert("L")
        if mask.size != (FRAME_WIDTH, FRAME_HEIGHT):
            mask.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.NEAREST).save(output)
        if not np.any(np.asarray(Image.open(output).convert("L")) > 0):
            raise RuntimeError("mask generator produced an empty mask")
        return self._write_receipt(
            run_id,
            mask=str(output),
            mask_confirmed=False,
            mask_details=details if isinstance(details, dict) else None,
            status="mask_ready",
        )

    def confirm_mask(self, run_id: str) -> dict[str, Any]:
        receipt = self._receipt(run_id)
        mask_path = Path(str(receipt.get("mask") or ""))
        if not mask_path.is_file():
            raise RuntimeError("Generate a mask before confirming it")
        if not np.any(np.asarray(Image.open(mask_path).convert("L")) > 0):
            raise RuntimeError("Cannot confirm an empty mask")
        return self._write_receipt(
            run_id,
            mask_confirmed=True,
            status="mask_confirmed",
        )

    def _caption(self, image: Path, output: Path) -> str:
        self.runner.run(caption_command(image=image, output=output), cwd=self.research_root)
        if not output.is_file():
            raise RuntimeError(f"caption worker did not create {output}")
        return caption_text(json.loads(output.read_text(encoding="utf-8")))

    def _write_case_config(self, run_id: str, prompt: str) -> Path:
        root = self._root(run_id)
        trajectory = root / "objects/object/trajectory.npz"
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        matrices = np.repeat(np.eye(4, dtype=np.float32)[None], FRAME_COUNT, axis=0)
        np.savez_compressed(trajectory, canonical_to_world=matrices)
        # Seeds. Stage 4 samples with inference_seed, and some seeds give a
        # visibly worse video for the same control, as sweeping the motion presets
        # over a few seeds shows. If a result looks off, rerun only the printed
        # Stage-4 command with another --seed: each seed writes its own
        # output-seed<seed>.mp4, and the stage-3 depth control does not depend on it.
        # pixal_seed below drives Pixal3D in "Generate 3D"; change it only when
        # the reconstructed mesh itself is wrong, then generate 3D again. The
        # caption "Generate 3D" writes is sampled with FOURDIRECTOR_QWEN3_VL_SEED.
        config = {
            "schema_version": "4director-case-v1",
            "case_id": run_id,
            "image": str(root / "inputs/image.png"),
            "prompt": prompt,
            "inference_seed": 123,
            "frame_count": FRAME_COUNT,
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": FPS,
            "camera": {"path": str(root / "cameras/camera.npz")},
            "objects": [{
                "id": "object",
                "image": str(root / "inputs/image.png"),
                "mask": str(root / "inputs/objects/object/mask.png"),
                "in_scene": True,
                "pixal_seed": 42,
                "trajectory": {"path": str(trajectory)},
            }],
        }
        path = root / "case.yaml"
        _atomic_json(path, config)
        return path

    def generate_3d(
        self,
        run_id: str,
        geometry_fn: Callable[..., Any] | None = None,
        object_fn: Callable[..., Any] | None = None,
        caption_fn: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        receipt = self._receipt(run_id)
        if not receipt.get("mask_confirmed") or not receipt.get("mask"):
            raise RuntimeError("Generate and confirm a mask before generating 3D")
        root = self._root(run_id)
        image = Path(receipt["image"])
        mask = Path(receipt["mask"])
        geometry = root / "scene/static_geometry.npz"
        geometry.parent.mkdir(parents=True, exist_ok=True)
        geometry_details = (
            geometry_fn(image=image, output=geometry)
            if geometry_fn is not None
            else static_geometry_provider(self.runner).run(image, geometry)
        )
        if not geometry.is_file():
            raise RuntimeError(f"geometry generator did not create {geometry}")
        background = root / "scene/background.npz"
        intrinsics, background_details = _background_from_geometry(
            geometry, image, [mask], background
        )
        camera = root / "cameras/camera.npz"
        camera.parent.mkdir(parents=True, exist_ok=True)
        identities = np.repeat(np.eye(4, dtype=np.float32)[None], FRAME_COUNT, axis=0)
        np.savez_compressed(
            camera,
            intrinsics=np.repeat(intrinsics[None], FRAME_COUNT, axis=0),
            extrinsics=identities,
            w2c=identities,
            image_size=np.asarray([FRAME_HEIGHT, FRAME_WIDTH], dtype=np.int32),
        )
        caption_output = root / "caption.json"
        caption = (
            caption_fn(image=image, output=caption_output)
            if caption_fn is not None
            else self._caption(image, caption_output)
        )
        if isinstance(caption, dict):
            caption = caption.get("text", "")
        caption = str(caption).strip()
        if not caption:
            raise RuntimeError("caption generator returned empty text")
        object_image = root / "inputs/objects/object/image.png"
        object_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, object_image)
        config = self._write_case_config(run_id, caption)
        object_details = stage2_objects(
            config,
            root,
            research_root=self.research_root,
            force=True,
            reconstruct=object_fn,
        )
        mesh = root / "objects/object/mesh.glb"
        if not mesh.is_file():
            raise RuntimeError(f"object reconstruction did not create {mesh}")
        return self._write_receipt(
            run_id,
            status="3d_ready",
            geometry=str(geometry),
            geometry_details=geometry_details,
            background=str(background),
            background_details=background_details,
            mesh=str(mesh),
            prompt=caption,
            caption_source="auto",
            object_details=object_details,
            config=str(config),
        )

    def set_prompt(self, run_id: str, prompt: str) -> dict[str, Any]:
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        receipt = self._write_receipt(
            run_id, prompt=prompt, caption_source="edited", status=self._receipt(run_id)["status"]
        )
        if receipt.get("config"):
            config_path = Path(receipt["config"])
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["prompt"] = prompt
            _atomic_json(config_path, config)
        case_path = self._root(run_id) / "case.json"
        if case_path.is_file():
            case = json.loads(case_path.read_text(encoding="utf-8"))
            case["prompt"] = prompt
            _atomic_json(case_path, case)
        return receipt

    def apply_preset(
        self,
        run_id: str,
        preset: str,
        *,
        builder: Callable[..., Any] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        hook = builder or self.preset_builder
        if hook is None:
            raise RuntimeError("No preset builder is configured")
        result = hook(run_id=run_id, run_root=self._root(run_id), preset=preset, **options)
        return self._write_receipt(run_id, preset=preset, preset_details=result)

    def render_command(self, run_id: str) -> str:
        root = self._root(run_id)
        self._receipt(run_id)
        return shlex.join([
            sys.executable, str(self.research_root / "run.py"), "stage3",
            "--config", str(root / "case.yaml"), "--run-root", str(root),
            "--force",
        ])

    def generate_command(self, run_id: str) -> str:
        root = self._root(run_id)
        # The seed is spelled out so another one is a one-word edit; the video
        # lands in output-seed<seed>.mp4, beside those of other seeds.
        seed = load_case_yaml(root / "case.yaml").get("inference_seed", 123)
        return shlex.join([
            sys.executable, str(self.research_root / "run.py"), "stage4",
            "--config", str(root / "case.yaml"), "--run-root", str(root),
            "--checkpoint", str(motion_adapter_checkpoint()), "--seed", str(seed), "--force",
        ])


def create_workflow() -> Frame0DemoWorkflow:
    return Frame0DemoWorkflow()

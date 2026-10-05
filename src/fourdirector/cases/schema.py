"""Case YAML contract: one image, per-object masks, prompt, seeds, trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .yaml_lite import load_mapping

SCHEMA_VERSION = "4director-case-v1"
DEFAULT_PIXAL_SEED = 42
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, warped geometry, flickering, jitter, "
    "artifacts, text, watermark, overexposed, underexposed, duplicated "
    "subject, deformed subject, inconsistent appearance"
)


def load_case_yaml(path: Path) -> dict[str, Any]:
    payload = load_mapping(path)
    validate_case_config(payload)
    return payload


def validate_case_config(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported case schema: {payload.get('schema_version')!r}"
        )
    for key in ("case_id", "image", "prompt", "objects"):
        if not payload.get(key):
            raise ValueError(f"case YAML missing {key}")
    objects = payload["objects"]
    if not isinstance(objects, list) or not objects:
        raise ValueError("case YAML needs at least one object")
    seen: set[str] = set()
    for spec in objects:
        if not isinstance(spec, dict) or not spec.get("id") or not spec.get("mask"):
            raise ValueError("each object needs id and mask")
        name = str(spec["id"])
        if name in seen:
            raise ValueError(f"duplicate object id: {name}")
        seen.add(name)
        try:
            int(spec.get("pixal_seed", DEFAULT_PIXAL_SEED))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.pixal_seed must be an integer") from exc
        if "pixal_fov" in spec and float(spec["pixal_fov"]) <= 0:
            raise ValueError(f"{name}.pixal_fov must be positive")
        if int(spec.get("pixal_decimation_target", 100000)) <= 0:
            raise ValueError(f"{name}.pixal_decimation_target must be positive")
        if int(spec.get("pixal_texture_size", 1024)) <= 0:
            raise ValueError(f"{name}.pixal_texture_size must be positive")
        if "in_scene" in spec and not isinstance(spec["in_scene"], bool):
            raise ValueError(f"{name}.in_scene must be true or false")
        if spec.get("in_scene") and not spec.get("image"):
            raise ValueError(
                f"{name}.in_scene requires its reconstruction image so the "
                "single reconstruction mask can be projected into the scene"
            )
        if "trajectory" not in spec and "trajectory_preset" not in spec:
            raise ValueError(f"{name} is missing a trajectory")
    camera = payload.get("camera")
    if camera is None:
        raise ValueError("case YAML missing camera")
    if not isinstance(camera, dict) or (
        "preset" not in camera and "matrices" not in camera and "path" not in camera
    ):
        raise ValueError("camera must be a named preset, path, or inline matrices")
    if int(payload.get("frame_count", 81)) != 81:
        raise ValueError("Cases have 81 frames")
    if int(payload.get("width", 832)) != 832 or int(payload.get("height", 480)) != 480:
        raise ValueError("Cases are 832x480")
    if int(payload.get("fps", 16)) != 16:
        raise ValueError("Cases run at 16 fps")

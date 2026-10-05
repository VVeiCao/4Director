"""Production adapter from an uploaded Frame-0 run to authored presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fourdirector.cases.trajectories import Frame0Geometry, build_frame0_preset


def _mesh_bounds(path: Path) -> tuple[np.ndarray, float]:
    # Keep trimesh out of module import and therefore out of UI startup.
    import trimesh

    loaded = trimesh.load(path, force="scene")
    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"invalid mesh bounds: {path}")
    span = float(bounds[1, 0] - bounds[0, 0])
    if span <= 1e-6:
        raise ValueError(f"mesh has no usable local X extent: {path}")
    return bounds.mean(axis=0), span


def _frame0_geometry(run_root: Path) -> Frame0Geometry:
    object_root = run_root / "objects/object"
    transform_path = object_root / "alignment.npz"
    with np.load(transform_path, allow_pickle=False) as archive:
        transforms = np.asarray(archive["canonical_to_world"], dtype=np.float64)
    with np.load(run_root / "cameras/camera.npz", allow_pickle=False) as archive:
        key = "extrinsics" if "extrinsics" in archive.files else "w2c"
        cameras = np.asarray(archive[key], dtype=np.float64)
    with np.load(run_root / "scene/static_geometry.npz", allow_pickle=False) as archive:
        camera_z = np.asarray(archive["camera_z"], dtype=np.float64)
    mask = np.asarray(
        Image.open(run_root / "inputs/objects/object/mask.png").convert("L")
    ) > 0
    depth = camera_z[0] if camera_z.ndim == 3 else camera_z
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("Frame-0 object mask has no valid positive scene depth")
    asset_center, asset_local_x_span = _mesh_bounds(object_root / "mesh.glb")
    return Frame0Geometry(
        canonical_to_world=transforms[0],
        world_to_camera=cameras[0],
        asset_center=asset_center,
        asset_local_x_span=asset_local_x_span,
        median_camera_z=float(np.median(depth[valid])),
    )


def apply_frame0_preset(
    *,
    run_id: str,
    run_root: Path,
    preset: str,
    body_widths: float | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Build and persist one of the six production Frame-0 presets."""
    resolved = run_root.expanduser().resolve()
    result = build_frame0_preset(
        preset,
        _frame0_geometry(resolved),
        body_widths=body_widths,
    )
    object_path = resolved / "objects/object/trajectory.npz"
    alignment_path = resolved / "objects/object/alignment.npz"
    camera_path = resolved / "cameras/camera.npz"
    object_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        object_path,
        canonical_to_world=result.object_canonical_to_world.astype(np.float32),
    )
    # The viewer and staged case contract use alignment.npz as the placed mesh path.
    np.savez_compressed(
        alignment_path,
        canonical_to_world=result.object_canonical_to_world.astype(np.float32),
    )
    with np.load(camera_path, allow_pickle=False) as archive:
        retained = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in {"extrinsics", "w2c"}
        }
    np.savez_compressed(
        camera_path,
        **retained,
        extrinsics=result.camera_world_to_camera.astype(np.float32),
        w2c=result.camera_world_to_camera.astype(np.float32),
    )
    metadata_path = resolved / "frame0-preset.json"
    metadata_path.write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "preset": preset,
        "object_trajectory": str(object_path),
        "camera": str(camera_path),
        "metadata": str(metadata_path),
    }

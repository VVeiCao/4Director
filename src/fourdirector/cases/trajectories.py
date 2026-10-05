"""Named camera presets and 81-frame object trajectories from Case YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fourdirector.constants import FPS, FRAME_COUNT
from fourdirector.stages.trajectory import (
    PRODUCTION_TRAJECTORY,
    static_trajectory,
    validate_trajectory_matrices,
)


def _yaw_matrices(degrees: float, frames: int = 81) -> np.ndarray:
    angles = np.linspace(0.0, np.deg2rad(degrees), frames, dtype=np.float64)
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None, ...], frames, axis=0)
    cos = np.cos(angles)
    sin = np.sin(angles)
    matrices[:, 0, 0] = cos
    matrices[:, 0, 2] = sin
    matrices[:, 2, 0] = -sin
    matrices[:, 2, 2] = cos
    return validate_trajectory_matrices(matrices.astype(np.float32))


CAMERA_PRESETS = {
    "identity": lambda: static_trajectory(),
    "static": lambda: static_trajectory(),
    "yaw-80": lambda: _yaw_matrices(-80.0),
    "yaw-90": lambda: _yaw_matrices(-90.0),
}


def resolve(relative: str, base: Path | None) -> Path:
    """Paths in a Case YAML are relative to that YAML, not the shell's cwd."""
    path = Path(relative).expanduser()
    if path.is_absolute() or base is None:
        return path
    return (base / path).resolve()


def load_camera_matrices(spec: dict[str, Any], base: Path | None = None) -> np.ndarray:
    if "path" in spec:
        with np.load(resolve(spec["path"], base), allow_pickle=False) as archive:
            key = "extrinsics" if "extrinsics" in archive.files else "w2c"
            return validate_trajectory_matrices(np.asarray(archive[key]))
    if "preset" in spec:
        preset = str(spec["preset"])
        if preset not in CAMERA_PRESETS:
            raise ValueError(f"unknown camera preset: {preset}")
        return CAMERA_PRESETS[preset]()
    if "matrices" in spec:
        return validate_trajectory_matrices(np.asarray(spec["matrices"], dtype=np.float32))
    raise ValueError("camera YAML needs preset, path, or matrices")


def _compose_relative(spec: dict[str, Any]) -> np.ndarray:
    frames = PRODUCTION_TRAJECTORY.frame_count
    if spec.get("preset") in {"identity", "static"} or spec == {"preset": "identity"}:
        return static_trajectory()
    keyframes = spec.get("relative_keyframes") or spec.get("keyframes")
    if not keyframes:
        return static_trajectory()
    ordered = sorted(keyframes, key=lambda item: int(item["frame"]))
    times = np.asarray([int(item["frame"]) for item in ordered], dtype=np.float64)
    translations = np.asarray(
        [
            [
                float(item.get("tx", 0.0)),
                float(item.get("ty", 0.0)),
                float(item.get("tz", 0.0)),
            ]
            for item in ordered
        ],
        dtype=np.float64,
    )
    yaws = np.asarray([float(item.get("yaw", 0.0)) for item in ordered], dtype=np.float64)
    sample = np.arange(frames, dtype=np.float64)
    tx = np.interp(sample, times, translations[:, 0])
    ty = np.interp(sample, times, translations[:, 1])
    tz = np.interp(sample, times, translations[:, 2])
    yaw = np.interp(sample, times, yaws)
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None, ...], frames, axis=0)
    cos = np.cos(yaw)
    sin = np.sin(yaw)
    matrices[:, 0, 0] = cos
    matrices[:, 0, 2] = sin
    matrices[:, 2, 0] = -sin
    matrices[:, 2, 2] = cos
    matrices[:, 0, 3] = tx
    matrices[:, 1, 3] = ty
    matrices[:, 2, 3] = tz
    return validate_trajectory_matrices(matrices.astype(np.float32))


def load_object_trajectory(spec: dict[str, Any], base: Path | None = None) -> np.ndarray:
    if "path" in spec:
        with np.load(resolve(spec["path"], base), allow_pickle=False) as archive:
            key = (
                "canonical_to_world"
                if "canonical_to_world" in archive.files
                else "world_transforms"
            )
            return validate_trajectory_matrices(np.asarray(archive[key]))
    return _compose_relative(spec)


# Frame-0 presets: the six authored DAVIS motions, rebuilt from one frame-zero
# scene measurement instead of any video trajectory estimation.

FRAME0_PRESET_SCHEMA = "4director-frame0-preset-v1"
# How far a translation preset travels, measured in the object's own width.
# Override per build with body_widths.
BODY_WIDTH_MULTIPLIER = 1.0
ORBIT_REFERENCE = "visulization/scripts/util/trajectory.py:build_orbit_camera_poses"
FRAME0_PRESETS: dict[str, dict[str, Any]] = {
    "object-rotate-90": {
        "motion": "object_clockwise_yaw",
        "display_name": "Object clockwise 90-degree in-place rotation",
        "degrees": 90.0,
    },
    "object-rotate-180": {
        "motion": "object_clockwise_yaw",
        "display_name": "Object clockwise 180-degree in-place rotation",
        "degrees": 180.0,
    },
    "object-move-right": {
        "motion": "object_translation_camera_follow",
        "display_name": "Object right in body widths / parallel camera follow",
        "direction": "right",
        "direction_sign": 1,
    },
    "object-move-left": {
        "motion": "object_translation_camera_follow",
        "display_name": "Object left in body widths / parallel camera follow",
        "direction": "left",
        "direction_sign": -1,
    },
    "camera-yaw-p120": {
        "motion": "static_object_camera_orbit_yaw",
        "display_name": "Static object / camera yaw +120 degrees",
        "yaw_degrees": 120.0,
    },
    "camera-yaw-n120": {
        "motion": "static_object_camera_orbit_yaw",
        "display_name": "Static object / camera yaw -120 degrees",
        "yaw_degrees": -120.0,
    },
}


@dataclass(frozen=True)
class Frame0Geometry:
    """One frame-zero measurement of the object and the camera that sees it."""

    canonical_to_world: np.ndarray
    world_to_camera: np.ndarray
    asset_center: np.ndarray
    asset_local_x_span: float
    median_camera_z: float

    def __post_init__(self) -> None:
        _decompose_sim3(self.canonical_to_world, name="canonical_to_world")
        _, camera_scale, _ = _decompose_sim3(self.world_to_camera, name="world_to_camera")
        if abs(camera_scale - 1.0) > 1e-4:
            raise ValueError("world_to_camera must be a rigid pose with unit scale")
        center = np.asarray(self.asset_center, dtype=np.float64).reshape(3).copy()
        if not np.isfinite(center).all():
            raise ValueError("asset_center contains NaN or infinity")
        span = float(self.asset_local_x_span)
        if not np.isfinite(span) or span <= 1e-6:
            raise ValueError("asset_local_x_span must be a positive asset-local extent")
        depth = float(self.median_camera_z)
        if not np.isfinite(depth) or depth <= 0.0:
            raise ValueError("median_camera_z must be finite and positive")
        object.__setattr__(
            self,
            "canonical_to_world",
            np.asarray(self.canonical_to_world, dtype=np.float64).copy(),
        )
        object.__setattr__(
            self,
            "world_to_camera",
            np.asarray(self.world_to_camera, dtype=np.float64).copy(),
        )
        object.__setattr__(self, "asset_center", center)
        object.__setattr__(self, "asset_local_x_span", span)
        object.__setattr__(self, "median_camera_z", depth)

    @property
    def object_sim3(self) -> tuple[np.ndarray, float, np.ndarray]:
        """Frame-zero object rotation, uniform scale, and translation."""
        return _decompose_sim3(self.canonical_to_world, name="canonical_to_world")

    @property
    def scale(self) -> float:
        """Frame-zero asset scale multiplier carried by the Sim(3)."""
        return self.object_sim3[1]

    @property
    def body_width(self) -> float:
        """Asset-local X AABB span expressed in frame-zero scene units."""
        return self.asset_local_x_span * self.scale

    @property
    def pivot(self) -> np.ndarray:
        """World-space asset center that every in-place rotation turns around."""
        rotation, scale, translation = self.object_sim3
        return translation + scale * (rotation @ self.asset_center)

    @property
    def camera_from_world(self) -> np.ndarray:
        """Frame-zero camera-to-world pose, ``inv(world_to_camera)``, RDF convention."""
        return np.linalg.inv(self.world_to_camera)


@dataclass(frozen=True)
class Frame0Preset:
    """One preset resolved into production matrices plus its authoring receipt."""

    preset: str
    object_canonical_to_world: np.ndarray
    camera_world_to_camera: np.ndarray
    metadata: dict[str, Any]


def frame0_preset_names() -> tuple[str, ...]:
    return tuple(FRAME0_PRESETS)


def build_frame0_preset(
    preset: str,
    geometry: Frame0Geometry,
    *,
    body_widths: float | None = None,
) -> Frame0Preset:
    """Rebuild one authored DAVIS motion from a frame-zero measurement.

    ``body_widths`` overrides how far the translation presets travel, so a
    sweep can vary the distance without editing this module.
    """
    name = str(preset)
    definition = FRAME0_PRESETS.get(name)
    if definition is None:
        raise ValueError(f"unknown frame-0 preset: {name}")
    motion = definition["motion"]
    if motion == "object_clockwise_yaw":
        objects, cameras, receipt = _object_yaw_preset(definition, geometry)
    elif motion == "object_translation_camera_follow":
        objects, cameras, receipt = _object_translation_preset(
            definition, geometry, body_widths=body_widths
        )
    else:
        objects, cameras, receipt = _camera_orbit_preset(definition, geometry)
    return Frame0Preset(
        preset=name,
        object_canonical_to_world=validate_trajectory_matrices(objects),
        camera_world_to_camera=validate_trajectory_matrices(cameras),
        metadata={
            "schema_version": FRAME0_PRESET_SCHEMA,
            "preset": name,
            "display_name": definition["display_name"],
            "motion": motion,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "object_convention": "canonical_to_world",
            "camera_convention": "world_to_camera",
            "source_geometry": "frame0_only_no_video_trajectory_estimation",
            **receipt,
        },
    )


def build_frame0_presets(
    geometry: Frame0Geometry,
    *,
    body_widths: float | None = None,
) -> dict[str, Frame0Preset]:
    return {
        name: build_frame0_preset(name, geometry, body_widths=body_widths)
        for name in FRAME0_PRESETS
    }


def _decompose_sim3(matrix: Any, *, name: str) -> tuple[np.ndarray, float, np.ndarray]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    axis_scales = np.linalg.norm(value[:3, :3], axis=0)
    scale = float(axis_scales.mean())
    if scale <= 0.0 or not np.allclose(axis_scales, scale, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{name} must have positive uniform scale")
    rotation = value[:3, :3] / scale
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not np.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(f"{name} rotation is not right-handed orthonormal")
    return rotation, scale, value[:3, 3].copy()


def _frame_alphas() -> np.ndarray:
    return np.linspace(0.0, 1.0, FRAME_COUNT, dtype=np.float64)


def _yaw_about_vertical(degrees: np.ndarray) -> np.ndarray:
    """Yaw about the scene's vertical axis, which is world Y here.

    Reconstruction hands us a right-down-forward world: X is right, Y is down,
    Z is depth. Turning an upright object therefore rotates about Y; rotating
    about Z would spin it like a pinwheel in the image plane. A positive angle
    is clockwise seen from above, because Y points down.
    """
    angles = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None, ...], len(angles), axis=0)
    cos = np.cos(angles)
    sin = np.sin(angles)
    matrices[:, 0, 0] = cos
    matrices[:, 0, 2] = sin
    matrices[:, 2, 0] = -sin
    matrices[:, 2, 2] = cos
    return matrices


def _camera_positions(world_to_camera: np.ndarray) -> np.ndarray:
    """World-space camera centers read back out of world-to-camera matrices."""
    return -np.einsum(
        "fji,fj->fi",
        world_to_camera[:, :3, :3],
        world_to_camera[:, :3, 3],
    )


def _sim3_frames(rotations: np.ndarray, scale: float, translations: np.ndarray) -> np.ndarray:
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None, ...], FRAME_COUNT, axis=0)
    matrices[:, :3, :3] = scale * rotations
    matrices[:, :3, 3] = translations
    return matrices


def _object_yaw_preset(
    definition: dict[str, Any],
    geometry: Frame0Geometry,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Clockwise yaw about the vertical (world Y) axis through the asset center.

    The translation is compensated analytically so the pivot stays fixed; the
    camera is untouched.
    """
    degrees = float(definition["degrees"])
    rotation, scale, _ = geometry.object_sim3
    pivot = geometry.pivot
    rotations = _yaw_about_vertical(degrees * _frame_alphas()) @ rotation
    objects = _sim3_frames(
        rotations,
        scale,
        pivot[None, :] - scale * (rotations @ geometry.asset_center),
    )
    centers = objects[:, :3, :3] @ geometry.asset_center + objects[:, :3, 3]
    drift = float(np.abs(centers - pivot[None, :]).max())
    if drift > 1e-9 * max(float(np.abs(pivot).max()), 1.0):
        raise RuntimeError(f"rotation pivot drifted by {drift}")
    cameras = np.repeat(geometry.world_to_camera[None, ...], FRAME_COUNT, axis=0)
    return (
        objects,
        cameras,
        {
            "object": {
                "motion": "clockwise_yaw",
                "axis": "world_Y_vertical",
                "direction": "clockwise_viewed_from_above",
                "degrees": degrees,
                "pivot_world": pivot.tolist(),
                "pivot_policy": "frame0_canonical_to_world_asset_center",
                "center_translation": "analytical_pivot_compensation",
                "pivot_drift_max": drift,
                "frame0_asset_scale_multiplier": scale,
            },
            "camera": {
                "motion": "static",
                "source": "frame0_world_to_camera",
            },
            "reference": {"contract": "davis2017-clockwise-yaw180-v1"},
        },
    )


def _object_translation_preset(
    definition: dict[str, Any],
    geometry: Frame0Geometry,
    *,
    body_widths: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """World-X move measured in body widths, matched one-for-one by the camera."""
    sign = int(definition["direction_sign"])
    multiplier = BODY_WIDTH_MULTIPLIER if body_widths is None else float(body_widths)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("body_widths must be a positive multiple of the object width")
    body_width = geometry.body_width
    displacement = sign * multiplier * body_width
    offsets = _frame_alphas() * displacement
    objects = np.repeat(geometry.canonical_to_world[None, ...], FRAME_COUNT, axis=0)
    objects[:, 0, 3] += offsets
    camera_from_world = np.repeat(geometry.camera_from_world[None, ...], FRAME_COUNT, axis=0)
    camera_from_world[:, 0, 3] += offsets
    cameras = np.linalg.inv(camera_from_world)
    positions = _camera_positions(cameras)
    lock_error = float(
        np.abs((positions - positions[0]) - (objects[:, :3, 3] - objects[0, :3, 3])).max()
    )
    if lock_error > 1e-9 * max(abs(displacement), 1.0):
        raise RuntimeError(f"camera follow diverged from the object by {lock_error}")
    return (
        objects,
        cameras,
        {
            "object": {
                "motion": "translation_only",
                "axis": "world_X",
                "direction": str(definition["direction"]),
                "direction_sign": sign,
                "distance_body_widths": multiplier,
                "body_width_scene_units": body_width,
                "body_width_definition": "object_mesh_asset_local_x_aabb_span_times_frame0_scale",
                "asset_local_x_span": geometry.asset_local_x_span,
                "frame0_asset_scale_multiplier": geometry.scale,
                "displacement_world": [displacement, 0.0, 0.0],
                "rotation_degrees": 0.0,
                "scale_change": 0.0,
            },
            "camera": {
                "motion": "parallel_world_x_translation",
                "displacement_world": [displacement, 0.0, 0.0],
                "orientation": "constant_frame0_camera_orientation",
                "object_camera_lock": "equal_per_frame_translation",
                "lock_error_max": lock_error,
            },
            "reference": {
                "contract": "davis2017-object-mesh-width-xtranslation-follow-v3",
            },
        },
    )


def _camera_orbit_preset(
    definition: dict[str, Any],
    geometry: Frame0Geometry,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Static object while the camera orbits the vertical (world Y) axis.

    The orbit center sits ``median_camera_z`` in front of the frame-0 camera,
    so the radius equals the object's median depth.
    """
    yaw = float(definition["yaw_degrees"])
    depth = geometry.median_camera_z
    base = geometry.camera_from_world
    center = base[:3, 3] + depth * (base[:3, :3] @ np.asarray([0.0, 0.0, 1.0]))
    deltas = _yaw_about_vertical(yaw * _frame_alphas())
    camera_from_world = np.repeat(np.eye(4, dtype=np.float64)[None, ...], FRAME_COUNT, axis=0)
    camera_from_world[:, :3, :3] = deltas @ base[:3, :3]
    camera_from_world[:, :3, 3] = center + (deltas @ (base[:3, 3] - center))
    cameras = np.linalg.inv(camera_from_world)
    radii = np.linalg.norm(_camera_positions(cameras) - center, axis=1)
    radius_drift = float(np.abs(radii - depth).max())
    if radius_drift > 1e-9 * max(depth, 1.0):
        raise RuntimeError(f"camera orbit radius drifted by {radius_drift}")
    objects = np.repeat(geometry.canonical_to_world[None, ...], FRAME_COUNT, axis=0)
    return (
        objects,
        cameras,
        {
            "object": {
                "motion": "object_static",
                "source": "frame0_canonical_to_world",
            },
            "camera": {
                "motion": "orbit_yaw",
                "axis": "world_Y_vertical",
                "yaw_degrees": yaw,
                "pitch_degrees": 0.0,
                "roll_degrees": 0.0,
                "center_world": center.tolist(),
                "center_depth_from_frame0_camera": depth,
                "radius": depth,
                "radius_drift_max": radius_drift,
                "rotation_order": "vertical_yaw_only",
            },
            "reference": {
                "contract": "davis2017-object-static-camera-yaw-v1",
                "implementation": ORBIT_REFERENCE,
            },
        },
    )

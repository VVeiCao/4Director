"""Viser playback of a Case: the background point cloud, and the objects and the
camera moving along their paths, looping at the video's frame rate."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def _load_transforms(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive[key], dtype=np.float32)


def decompose_sim3(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    import viser.transforms as tf

    value = np.asarray(matrix, dtype=np.float64)
    linear = value[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("object transform has invalid Sim(3) scale")
    return (
        value[:3, 3].astype(np.float32),
        tf.SO3.from_matrix(linear / scale).wxyz.astype(np.float32),
        scale,
    )


def configure_trajectory_viewer(
    server: Any,
    *,
    case_root: Path,
    case: dict[str, Any],
    namespace: str = "case",
    autoplay: bool = True,
) -> dict[str, Any]:
    import viser

    case_root = case_root.expanduser().resolve()
    frame_count = int(case["frame_count"])
    fps = float(case.get("fps", 16))
    with np.load(case_root / "scene/background.npz", allow_pickle=False) as archive:
        points = np.asarray(archive["points"], dtype=np.float32)
        colors = np.asarray(archive["colors"], dtype=np.uint8)
    display_from_shared = np.diag(
        np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float32)
    )
    server.scene.set_up_direction("+y")
    server.scene.add_point_cloud(
        f"/{namespace}/background",
        points=points @ display_from_shared[:3, :3].T,
        colors=colors,
        point_size=0.003,
        point_shape="rounded",
    )

    def path_segments(positions: np.ndarray) -> np.ndarray:
        if frame_count < 2:
            return np.zeros((0, 2, 3), dtype=np.float32)
        return np.stack((positions[:-1], positions[1:]), axis=1)

    object_transforms: dict[str, np.ndarray] = {}
    object_handles: dict[str, Any] = {}
    palette = ((50, 190, 90), (245, 125, 45), (190, 80, 200), (240, 200, 40))
    details = {str(item["id"]): item for item in case["objects_detail"]}
    for index, name in enumerate(case["objects"]):
        mesh_path = case_root / "objects" / name / "mesh.glb"
        trajectory_path = case_root / details[name]["trajectory"]
        transforms = _load_transforms(trajectory_path, "canonical_to_world")
        if len(transforms) != frame_count:
            raise ValueError(f"{name} trajectory length {len(transforms)} != {frame_count}")
        transforms = (display_from_shared[None] @ transforms).astype(np.float32)
        object_transforms[name] = transforms
        position, wxyz, scale = decompose_sim3(transforms[0])
        object_handles[name] = server.scene.add_glb(
            f"/{namespace}/objects/{name}",
            mesh_path.read_bytes(),
            position=position,
            wxyz=wxyz,
            scale=scale,
        )
        server.scene.add_line_segments(
            f"/{namespace}/trajectories/{name}",
            points=path_segments(transforms[:, :3, 3]),
            colors=palette[index % len(palette)],
            line_width=2.0,
        )

    camera_w2c = _load_transforms(case_root / case["camera"], "extrinsics")
    if len(camera_w2c) != frame_count:
        raise ValueError(f"camera length {len(camera_w2c)} != {frame_count}")
    camera_c2w = (display_from_shared[None] @ np.linalg.inv(camera_w2c)).astype(np.float32)
    server.scene.add_line_segments(
        f"/{namespace}/trajectories/camera",
        points=path_segments(camera_c2w[:, :3, 3]),
        colors=(50, 125, 245),
        line_width=2.5,
    )
    camera_position, camera_wxyz, _ = decompose_sim3(camera_c2w[0])
    camera_handle = server.scene.add_camera_frustum(
        f"/{namespace}/camera",
        fov=np.deg2rad(55.0),
        aspect=float(case.get("width", 832)) / float(case.get("height", 480)),
        scale=0.075,
        color=(50, 125, 245),
        position=camera_position,
        wxyz=camera_wxyz,
    )

    # Open behind and above the frame-0 camera, looking at the objects, so the
    # scene and every path are in view rather than a distant speck.
    focus = (
        np.mean([transforms[0][:3, 3] for transforms in object_transforms.values()], axis=0)
        if object_transforms
        else np.median(points @ display_from_shared[:3, :3].T, axis=0)
    )
    reach = float(np.linalg.norm(focus - camera_position)) or 1.0
    right, up, forward = (
        camera_c2w[0][:3, 0],
        -camera_c2w[0][:3, 1],
        camera_c2w[0][:3, 2],
    )
    view_position = camera_position + reach * (0.6 * right + 1.0 * up - 1.4 * forward)

    @server.on_client_connect
    def frame_scene(client: Any) -> None:
        client.camera.position = view_position
        client.camera.look_at = focus

    def show_frame(index: int) -> None:
        with server.atomic():
            for name, transforms in object_transforms.items():
                position, wxyz, scale = decompose_sim3(transforms[index])
                object_handles[name].position = position
                object_handles[name].wxyz = wxyz
                object_handles[name].scale = scale
            position, wxyz, _ = decompose_sim3(camera_c2w[index])
            camera_handle.position = position
            camera_handle.wxyz = wxyz

    state = {"frame": 0, "playing": autoplay}
    toggle = server.gui.add_button(
        "Pause" if autoplay else "Play",
        icon=viser.Icon.PLAYER_PAUSE if autoplay else viser.Icon.PLAYER_PLAY,
    )

    @toggle.on_click
    def toggle_playback(_: Any) -> None:
        state["playing"] = not state["playing"]
        toggle.label = "Pause" if state["playing"] else "Play"
        toggle.icon = viser.Icon.PLAYER_PAUSE if state["playing"] else viser.Icon.PLAYER_PLAY

    def advance() -> None:
        state["frame"] = (state["frame"] + 1) % frame_count
        show_frame(state["frame"])

    def play() -> None:
        while True:
            started = time.monotonic()
            if state["playing"] and frame_count > 1:
                advance()
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))

    threading.Thread(target=play, name=f"{namespace}-playback", daemon=True).start()

    return {
        "status": "ready",
        "case_id": case["case_id"],
        "frames": frame_count,
        "object_ids": list(object_transforms),
        "state": state,
        "toggle": toggle_playback,
        "advance": advance,
    }

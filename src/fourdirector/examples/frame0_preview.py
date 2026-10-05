"""Build a single GLB that shows the scene and the authored motion.

The Gradio page renders this file directly, so the preview needs no second
server, no WebSocket, and no forwarded port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Reconstruction works in a right-down-forward world; flip to Y-up for viewers.
DISPLAY_FROM_WORLD = np.diag(np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float64))
# The animated clip carries the motion, so the 3D scene shows one pose plus paths.
GHOST_FRAMES = 1
OBJECT_COLOR = (60, 200, 110, 255)
CAMERA_COLOR = (70, 150, 250, 255)
# Every file the two builders read, relative to the run root.
PREVIEW_INPUTS = (
    "objects/object/mesh.glb",
    "objects/object/trajectory.npz",
    "objects/object/alignment.npz",
    "cameras/camera.npz",
    "scene/background.npz",
)


def is_current(run_root: str | Path, preview: str | Path) -> bool:
    """Whether ``preview`` is newer than every input it is drawn from.

    The comparison is strict, so on a filesystem with coarse timestamps an input
    written in the same tick as the preview still forces a rebuild.
    """
    output = Path(preview)
    if not output.is_file():
        return False
    built = output.stat().st_mtime_ns
    root = Path(run_root)
    return all(
        not (root / name).is_file() or (root / name).stat().st_mtime_ns < built
        for name in PREVIEW_INPUTS
    )


def _object_transforms(object_root: Path) -> np.ndarray:
    """Prefer the authored trajectory, but fall back to the fitted placement.

    Reconstruction seeds ``trajectory.npz`` with identities, so before a preset
    is applied only ``alignment.npz`` knows where the mesh belongs.
    """
    candidates = [object_root / "trajectory.npz", object_root / "alignment.npz"]
    for source in candidates:
        if not source.is_file():
            continue
        with np.load(source, allow_pickle=False) as archive:
            transforms = np.asarray(archive["canonical_to_world"], dtype=np.float64)
        identity = np.allclose(transforms, np.eye(4, dtype=np.float64)[None])
        if not identity or source is candidates[-1]:
            return transforms
    raise FileNotFoundError(f"no object placement found under {object_root}")


def _load_trajectories(run_root: Path) -> tuple[np.ndarray, np.ndarray]:
    objects = _object_transforms(run_root / "objects/object")
    with np.load(run_root / "cameras/camera.npz", allow_pickle=False) as archive:
        key = "extrinsics" if "extrinsics" in archive.files else "w2c"
        world_to_camera = np.asarray(archive[key], dtype=np.float64)
    cameras = np.linalg.inv(world_to_camera)
    return (
        DISPLAY_FROM_WORLD[None] @ objects,
        DISPLAY_FROM_WORLD[None] @ cameras,
    )


def _make_material_visible(mesh: Any) -> None:
    """Pixal3D exports fully metallic PBR, which renders black without an
    environment map. Gradio's viewer has none, so treat the surface as a
    dielectric and keep the baked texture."""
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if material is None:
        return
    if hasattr(material, "metallicFactor"):
        material.metallicFactor = 0.0
    if hasattr(material, "roughnessFactor"):
        material.roughnessFactor = 0.7


def _path_geometry(positions: np.ndarray, color: tuple[int, int, int, int], radius: float) -> Any:
    """Draw a trajectory as beads, which every glTF viewer can display."""
    import trimesh

    unique = [positions[0]]
    for position in positions[1:]:
        if float(np.linalg.norm(position - unique[-1])) > radius * 0.35:
            unique.append(position)
    if not np.allclose(unique[-1], positions[-1]):
        unique.append(positions[-1])
    marker = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    marker.visual.vertex_colors = np.tile(
        np.asarray(color, dtype=np.uint8), (len(marker.vertices), 1)
    )
    beads = []
    for position in unique:
        bead = marker.copy()
        bead.apply_translation(position)
        beads.append(bead)
    return trimesh.util.concatenate(beads)


def _start_encoder(output: Path, *, width: int, height: int, fps: int) -> Any:
    """Stream raw frames into the ffmpeg the environment already requires."""
    import subprocess

    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-to-camera for a right-down-forward camera aimed at ``target``."""
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.asarray([0.0, -1.0, 0.0])
    if abs(float(np.dot(world_up, forward))) > 0.999:
        world_up = np.asarray([0.0, 0.0, -1.0])
    right = np.cross(forward, world_up)
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ eye
    return matrix


def _frustum_points(
    camera_to_world: np.ndarray,
    *,
    scale: float,
    samples: int = 40,
) -> np.ndarray:
    """Sample the edges of a camera frustum so it can be splatted as points."""
    half = scale * 0.4
    corners = np.asarray(
        [
            [-half, -half, scale],
            [half, -half, scale],
            [half, half, scale],
            [-half, half, scale],
        ]
    )
    origin = np.zeros(3)
    edges = [(origin, corner) for corner in corners]
    edges += [(corners[index], corners[(index + 1) % 4]) for index in range(4)]
    local = np.concatenate(
        [
            start + (end - start) * np.linspace(0.0, 1.0, samples)[:, None]
            for start, end in edges
        ]
    )
    return local @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def _external_view(
    *,
    scene_points: np.ndarray,
    object_points: np.ndarray,
    transforms: np.ndarray,
    camera_to_world: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pick one fixed vantage point that keeps the whole animation in frame.

    The eye sits above and behind the authored camera so its frustum stays
    visible, and the focal length is then solved so nothing leaves the frame.
    """
    # Frame the action, not the scene: a reconstruction runs tens of units deep,
    # and fitting all of it shrinks the subject to a speck.
    path = transforms[:, :3, 3]
    positions = camera_to_world[:, :3, 3]
    radius = float(np.linalg.norm(object_points.max(axis=0) - object_points.min(axis=0))) * 0.5
    corners = path[:, None, :] + radius * np.asarray(
        [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
         [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]],
        dtype=np.float64,
    )[None]
    action = np.concatenate((corners.reshape(-1, 3), positions))
    center = (action.min(axis=0) + action.max(axis=0)) * 0.5
    span = float(np.linalg.norm(action - center, axis=1).max()) or 1.0
    direction = np.asarray([0.35, -0.75, -1.1])
    direction = direction / float(np.linalg.norm(direction))
    eye = center + direction * (2.4 * span)
    w2c = _look_at(eye, center)

    # Fit to the true extremes of the action, not a percentile: the object and the
    # camera marker must never be clipped, even in one frame. Background may fall
    # outside, which only crops scenery.
    frustums = np.concatenate(
        [
            _frustum_points(pose, scale=max(span * 0.15, radius * 0.5))
            for pose in camera_to_world
        ]
    )
    fit = np.concatenate((action, frustums))
    local = fit @ w2c[:3, :3].T + w2c[:3, 3]
    local = local[local[:, 2] > 1e-3]
    horizontal = np.abs(local[:, 0] / local[:, 2]).max()
    vertical = np.abs(local[:, 1] / local[:, 2]).max()
    focal = 0.92 * min(
        width / (2.0 * max(float(horizontal), 1e-6)),
        height / (2.0 * max(float(vertical), 1e-6)),
    )
    intrinsics = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )
    return w2c, intrinsics, max(span * 0.15, radius * 0.5)


def _splat(
    canvas: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    brush: int = 2,
) -> None:
    """Project points and paint them nearest-last, so near points win."""
    height, width = canvas.shape[:2]
    camera_points = points @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera_points[:, 2]
    visible = depth > 1e-3
    camera_points, colors, depth = camera_points[visible], colors[visible], depth[visible]
    u = np.rint(
        intrinsics[0, 0] * camera_points[:, 0] / depth + intrinsics[0, 2]
    ).astype(np.int64)
    v = np.rint(
        intrinsics[1, 1] * camera_points[:, 1] / depth + intrinsics[1, 2]
    ).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, colors, depth = u[inside], v[inside], colors[inside], depth[inside]
    order = np.argsort(-depth)
    u, v, colors = u[order], v[order], colors[order]
    for offset_y in range(brush):
        for offset_x in range(brush):
            shifted_u, shifted_v = u + offset_x, v + offset_y
            keep = (shifted_u < width) & (shifted_v < height)
            canvas[shifted_v[keep], shifted_u[keep]] = colors[keep]


def _background_cloud(run_root: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the scene points, dropping non-finite ones and thinning evenly."""
    background = run_root / "scene/background.npz"
    if not background.is_file():
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    with np.load(background, allow_pickle=False) as archive:
        points = np.asarray(archive["points"], dtype=np.float64)
        colors = np.asarray(archive["colors"], dtype=np.uint8)[:, :3]
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if len(points) > limit:
        selected = np.linspace(0, len(points) - 1, num=limit).astype(int)
        points, colors = points[selected], colors[selected]
    return points, colors


def _load_mesh(run_root: Path) -> Any:
    """Load the object mesh with a material that is visible without an env map."""
    import trimesh

    mesh = trimesh.load(run_root / "objects/object/mesh.glb", force="mesh")
    _make_material_visible(mesh)
    return mesh


def _object_point_cloud(mesh: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample the textured surface so the object can be splatted like the scene."""
    import trimesh

    points, face_index = trimesh.sample.sample_surface(mesh, count)
    colors = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.uint8)[:, :3]
    face_colors = colors[mesh.faces[face_index]].mean(axis=1)
    return np.asarray(points, dtype=np.float64), face_colors.astype(np.uint8)


def build_motion_video(
    run_root: str | Path,
    destination: str | Path,
    *,
    fps: int = 16,
    background_points: int = 150_000,
    object_points: int = 80_000,
    view: str = "external",
) -> Path:
    """Animate the scene as a point cloud, one frame at a time.

    ``view="external"`` watches from a fixed vantage point, so both the object
    and the camera frustum are visible as they move. ``view="camera"`` renders
    what the authored camera sees, which is the framing of the final video.
    """
    if view not in {"external", "camera"}:
        raise ValueError("view must be 'external' or 'camera'")

    root = Path(run_root).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    with np.load(root / "cameras/camera.npz", allow_pickle=False) as archive:
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
        key = "extrinsics" if "extrinsics" in archive.files else "w2c"
        world_to_camera = np.asarray(archive[key], dtype=np.float64)
        height, width = (int(value) for value in np.asarray(archive["image_size"])[:2])
    transforms = _object_transforms(root / "objects/object")

    scene_points, scene_colors = _background_cloud(root, background_points)
    canonical, object_colors = _object_point_cloud(_load_mesh(root), object_points)

    camera_to_world = np.linalg.inv(world_to_camera)
    if view == "external":
        external_w2c, external_k, frustum_scale = _external_view(
            scene_points=scene_points,
            object_points=canonical,
            transforms=transforms,
            camera_to_world=camera_to_world,
            width=width,
            height=height,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    # Encode beside the destination and move it into place only once complete,
    # so an interrupted build never leaves a clip that looks current.
    partial = output.with_suffix(".partial.mp4")
    encoder = _start_encoder(partial, width=width, height=height, fps=int(fps))
    for index in range(len(transforms)):
        placed = canonical @ transforms[index][:3, :3].T + transforms[index][:3, 3]
        w2c = external_w2c if view == "external" else world_to_camera[index]
        k = external_k if view == "external" else intrinsics[index]
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        _splat(
            canvas,
            np.concatenate((scene_points, placed)),
            np.concatenate((scene_colors, object_colors)),
            w2c=w2c,
            intrinsics=k,
        )
        if view == "external":
            # Drawn last and unoccluded: the camera marker must never be hidden.
            frustum = _frustum_points(camera_to_world[index], scale=frustum_scale)
            _splat(
                canvas,
                frustum,
                np.tile(np.asarray(CAMERA_COLOR[:3], dtype=np.uint8), (len(frustum), 1)),
                w2c=w2c,
                intrinsics=k,
                brush=4,
            )
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed to encode {output}")
    partial.replace(output)
    return output


def build_preview(
    run_root: str | Path,
    destination: str | Path,
    *,
    max_points: int = 120_000,
    ghosts: int = GHOST_FRAMES,
) -> Path:
    """Write background points, motion ghosts, and both paths into one GLB."""
    import trimesh

    root = Path(run_root).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    objects, cameras = _load_trajectories(root)
    scene = trimesh.Scene()

    mesh = _load_mesh(root)
    extent = float(np.linalg.norm(mesh.extents)) or 1.0
    count = max(int(ghosts), 1)
    indices = (
        np.asarray([0])
        if count == 1
        else np.unique(np.linspace(0, len(objects) - 1, num=count).astype(int))
    )
    for index in indices:
        ghost = mesh.copy()
        ghost.apply_transform(objects[index])
        scene.add_geometry(ghost, node_name=f"object_frame_{int(index):03d}")

    points, colors = _background_cloud(root, max_points)
    if len(points):
        display_points = points @ DISPLAY_FROM_WORLD[:3, :3].T
        scene.add_geometry(
            trimesh.PointCloud(display_points, colors), node_name="background"
        )

    # Thin beads vanish against a full point cloud, so keep the paths chunky.
    radius = max(extent * 0.06, 1e-4)
    scene.add_geometry(
        _path_geometry(objects[:, :3, 3], OBJECT_COLOR, radius),
        node_name="object_path",
    )
    scene.add_geometry(
        _path_geometry(cameras[:, :3, 3], CAMERA_COLOR, radius),
        node_name="camera_path",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.glb")
    partial.write_bytes(scene.export(file_type="glb"))
    partial.replace(output)
    return output

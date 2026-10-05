#!/usr/bin/env python3
"""Render multi-object raw camera-Z and the VACE depth controls stage 4 takes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline_eval.adapters.v5_rotation import (
    CONTROL_SCHEMA_VERSION,
    DEPTH_ENCODING_SCHEMA_VERSION,
)
from baseline_eval.core.io import (
    atomic_write_json,
    image_files,
    sha256_file,
)


def balanced_inverse_range(
    depth: np.ndarray,
    front_id: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    if depth.shape != front_id.shape:
        raise ValueError("Depth and front-entity arrays have different shapes")
    per_entity: dict[str, Any] = {}
    inverse_lows: list[float] = []
    inverse_highs: list[float] = []
    for entity in sorted(np.unique(front_id[front_id >= 0]).tolist()):
        values = depth[
            (front_id == entity) & np.isfinite(depth) & (depth > 0)
        ]
        if not len(values):
            continue
        inverse = 1.0 / values.astype(np.float64)
        low, high = np.quantile(inverse, [0.01, 0.99])
        inverse_lows.append(float(low))
        inverse_highs.append(float(high))
        per_entity[str(entity)] = {
            "valid_pixels": int(len(values)),
            "inverse_q01": float(low),
            "inverse_q99": float(high),
        }
    if not inverse_lows:
        raise ValueError("Rendered control has no valid depth")
    inverse_far = min(inverse_lows)
    inverse_near = max(inverse_highs)
    if inverse_near <= inverse_far:
        raise ValueError("Degenerate entity-balanced inverse-depth range")
    return float(inverse_far), float(inverse_near), per_entity


def encode_inverse_depth(
    depth: np.ndarray,
    front_id: np.ndarray,
    invalid_value: int = 128,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0 <= invalid_value <= 255:
        raise ValueError("Invalid grayscale value must be in [0,255]")
    inverse_far, inverse_near, per_entity = balanced_inverse_range(
        depth,
        front_id,
    )
    valid = np.isfinite(depth) & (depth > 0) & (front_id >= 0)
    encoded = np.full(depth.shape, invalid_value, dtype=np.uint8)
    inverse = 1.0 / depth[valid].astype(np.float64)
    normalized = (inverse - inverse_far) / (inverse_near - inverse_far)
    encoded[valid] = np.rint(
        np.clip(normalized, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return encoded, {
        "curve": "inverse_z",
        "scope": f"{depth.shape[0]}-frame entity-balanced",
        "entity_robust_percentiles": [1.0, 99.0],
        "combination": (
            "inverse_far=min(entity_q01), inverse_near=max(entity_q99)"
        ),
        "inverse_far": inverse_far,
        "inverse_near": inverse_near,
        "depth_far": float(1.0 / inverse_far),
        "depth_near": float(1.0 / inverse_near),
        "margin": None,
        "quantization": "numpy.rint(clip(normalized,0,1)*255)",
        "near": "white_255",
        "far": "black_0",
        "invalid": f"neutral_gray_{invalid_value}",
        "per_entity": per_entity,
    }


def compose_entity_depths(entity_depths: np.ndarray) -> dict[str, np.ndarray]:
    """Compose [entity, frame, height, width] raw camera-Z arrays."""
    depths = np.asarray(entity_depths, dtype=np.float32)
    if depths.ndim != 4 or depths.shape[0] < 1:
        raise ValueError(
            "Entity depths must have shape [entity,frame,height,width]"
        )
    if depths.shape[0] > np.iinfo(np.int16).max:
        raise ValueError("Too many entities for int16 sidecar IDs")
    safe = np.where(
        np.isfinite(depths) & (depths > 0),
        depths,
        np.inf,
    )
    order = np.argsort(safe, axis=0, kind="stable")
    front_order = order[0:1]
    front_depth = np.take_along_axis(safe, front_order, axis=0)[0]
    front_id = order[0].astype(np.int16)
    front_valid = np.isfinite(front_depth)
    front_depth = np.where(front_valid, front_depth, 0.0).astype(np.float32)
    front_id = np.where(front_valid, front_id, -1).astype(np.int16)
    if depths.shape[0] > 1:
        second_order = order[1:2]
        second_depth = np.take_along_axis(safe, second_order, axis=0)[0]
        second_id = order[1].astype(np.int16)
        second_valid = np.isfinite(second_depth)
        second_id = np.where(second_valid, second_id, -1).astype(np.int16)
        second_depth = np.where(
            second_valid,
            second_depth,
            np.nan,
        ).astype(np.float32)
    else:
        second_valid = np.zeros_like(front_valid)
        second_id = np.full(front_id.shape, -1, dtype=np.int16)
        second_depth = np.full(front_depth.shape, np.nan, dtype=np.float32)
    depth_gap = np.where(
        front_valid & second_valid,
        second_depth - front_depth,
        np.nan,
    ).astype(np.float32)
    object_ids = np.arange(1, depths.shape[0], dtype=np.int16)
    object_visible = (
        front_id[:, None] == object_ids[None, :, None, None]
    )
    return {
        "camera_z": front_depth,
        "front_id": front_id,
        "second_id": second_id,
        "second_depth": second_depth,
        "depth_gap": depth_gap,
        "object_visible": object_visible,
        "validity": front_valid,
    }


def object_edit_mask(object_masks: list[np.ndarray]) -> np.ndarray:
    if not object_masks:
        raise ValueError("At least one object mask is required")
    shape = np.asarray(object_masks[0]).shape
    if any(np.asarray(mask).shape != shape for mask in object_masks):
        raise ValueError("Object mask arrays have different shapes")
    return np.logical_or.reduce(object_masks).astype(np.uint8) * 255


def read_gray_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for decoding: {path}")
    decoded: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame if frame.ndim == 2 else frame[..., 0])
    finally:
        capture.release()
    if not decoded:
        raise RuntimeError(f"Video decoded no frames: {path}")
    return np.stack(decoded).astype(np.uint8)


def write_lossless_gray_video(
    path: Path,
    frames: np.ndarray,
    fps: float,
) -> None:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 3 or not len(frames):
        raise ValueError("Lossless video frames must have shape [T,H,W]")
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    height, width = frames.shape[1:]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"FFV1"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open FFV1 writer: {path}")
    try:
        for frame in frames:
            writer.write(np.repeat(frame[..., None], 3, axis=-1))
    finally:
        writer.release()
    decoded = read_gray_video(path)
    if decoded.shape != frames.shape:
        raise RuntimeError(
            f"Lossless video decoded shape {decoded.shape}, expected "
            f"{frames.shape}"
        )
    if not np.array_equal(decoded, frames):
        raise RuntimeError(f"Lossless grayscale roundtrip changed values: {path}")


def write_lossless_binary_video(
    path: Path,
    masks: np.ndarray,
    fps: float,
) -> None:
    masks = np.asarray(masks, dtype=np.uint8)
    if not set(np.unique(masks).tolist()).issubset({0, 255}):
        raise ValueError("Binary mask contains values outside {0,255}")
    write_lossless_gray_video(path, masks, fps)


def _write_preview(path: Path, frames: np.ndarray, fps: float) -> None:
    frames = np.asarray(frames, dtype=np.uint8)
    height, width = frames.shape[1:]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open preview writer: {path}")
    try:
        for frame in frames:
            writer.write(np.repeat(frame[..., None], 3, axis=-1))
    finally:
        writer.release()


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    mesh = (
        loaded.dump(concatenate=True)
        if isinstance(loaded, trimesh.Scene)
        else loaded
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"Invalid mesh vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Invalid mesh faces: {path}")
    return vertices, faces


def _pytorch3d_cameras(
    intrinsics: Any,
    world_to_camera: Any,
    image_size: tuple[int, int],
) -> Any:
    """Screen-space PyTorch3D cameras for a batch of OpenCV pinhole cameras.

    ``intrinsics`` are (B, 3, 3) pixel matrices and ``world_to_camera`` (B, 4, 4)
    OpenCV extrinsics (+X right, +Y down, +Z forward). PyTorch3D's camera has
    +X left and +Y up, so the camera-to-world rotation gets its first two axes
    negated, and PyTorch3D right-multiplies row vectors, so R is the transpose
    of the world-to-camera rotation.
    """
    import torch
    from pytorch3d.renderer import PerspectiveCameras

    intrinsics = intrinsics.float()
    camera_to_world = torch.linalg.inv(world_to_camera.float())
    camera_to_world[:, :3, :2] = -camera_to_world[:, :3, :2]
    pytorch3d_world_to_camera = torch.linalg.inv(camera_to_world)
    return PerspectiveCameras(
        device=intrinsics.device,
        focal_length=torch.stack(
            [intrinsics[:, 0, 0], intrinsics[:, 1, 1]], dim=1
        ),
        principal_point=torch.stack(
            [intrinsics[:, 0, 2], intrinsics[:, 1, 2]], dim=1
        ),
        R=pytorch3d_world_to_camera[:, :3, :3].transpose(1, 2),
        T=pytorch3d_world_to_camera[:, :3, 3],
        in_ndc=False,
        image_size=[tuple(image_size)] * len(intrinsics),
    )


def _render_point_depth_batch(
    points: Any,
    intrinsics: Any,
    world_to_camera: Any,
    image_size: tuple[int, int],
    *,
    radius: float,
) -> Any:
    """Camera-Z of the nearest point at each pixel, 0 where none lands."""
    import torch
    from pytorch3d.renderer import (
        PointsRasterizationSettings,
        PointsRasterizer,
    )
    from pytorch3d.structures import Pointclouds

    height, width = image_size
    if not len(points):
        return torch.zeros(
            (len(intrinsics), height, width),
            dtype=torch.float32,
            device=points.device,
        )
    rasterizer = PointsRasterizer(
        cameras=_pytorch3d_cameras(intrinsics, world_to_camera, image_size),
        raster_settings=PointsRasterizationSettings(
            image_size=(height, width),
            radius=radius,
            points_per_pixel=8,
            bin_size=128,
        ),
    )
    with torch.no_grad():
        fragments = rasterizer(
            Pointclouds(points=[points.float()] * len(intrinsics))
        )
    hit = fragments.idx[..., 0] != -1
    depth = fragments.zbuf[..., 0].float()
    depth[~hit] = 0.0
    return depth


def _render_mesh_depth_batch(
    meshes_list: list[Any],
    intrinsics: Any,
    extrinsics: Any,
    image_size: tuple[int, int],
) -> tuple[Any, Any]:
    from pytorch3d.renderer import (
        MeshRasterizer,
        RasterizationSettings,
    )
    from pytorch3d.structures import join_meshes_as_batch

    height, width = image_size
    cameras = _pytorch3d_cameras(intrinsics, extrinsics, image_size)
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        ),
    )
    import torch

    with torch.no_grad():
        fragments = rasterizer(join_meshes_as_batch(meshes_list))
    mask = fragments.pix_to_face[..., 0] != -1
    depth = fragments.zbuf[..., 0].float()
    depth[~mask] = 0.0
    return depth, mask


def render_reference_silhouette(
    *,
    mesh_path: Path,
    transform_path: Path,
    target_mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    reference_frame: int,
    device_name: str = "cuda",
) -> dict[str, Any]:
    import torch
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes

    vertices_np, faces_np = _load_mesh(mesh_path)
    with np.load(transform_path) as archive:
        transforms = np.asarray(
            archive["canonical_to_world"],
            dtype=np.float32,
        )
    if not 0 <= reference_frame < len(transforms):
        raise ValueError("Reference frame is outside object transforms")
    target = np.asarray(target_mask, dtype=bool)
    height, width = target.shape
    device = torch.device(device_name)
    vertices = torch.from_numpy(vertices_np.astype(np.float32)).to(device)
    faces = torch.from_numpy(faces_np.astype(np.int64)).to(device)
    vertices_h = torch.cat(
        [vertices, torch.ones((len(vertices), 1), device=device)],
        dim=1,
    )
    transform = torch.from_numpy(transforms[reference_frame]).to(device)
    transformed = (transform @ vertices_h.T).T[:, :3]
    texture = TexturesVertex(
        verts_features=torch.ones(
            (1, len(vertices), 3),
            dtype=torch.float32,
            device=device,
        )
    )
    mesh = Meshes(verts=[transformed], faces=[faces], textures=texture)
    _, rendered_batch = _render_mesh_depth_batch(
        [mesh],
        torch.from_numpy(np.asarray(intrinsic, dtype=np.float32))[None].to(
            device
        ),
        torch.from_numpy(np.asarray(extrinsic, dtype=np.float32))[None].to(
            device
        ),
        (height, width),
    )
    rendered = rendered_batch[0].detach().cpu().numpy().astype(bool)
    intersection = int(np.logical_and(rendered, target).sum())
    union = int(np.logical_or(rendered, target).sum())
    return {
        "reference_frame": int(reference_frame),
        "reference_iou": float(intersection / max(union, 1)),
        "rendered_pixels": int(rendered.sum()),
        "target_pixels": int(target.sum()),
    }


def render_case_controls(
    *,
    mesh_paths: list[Path],
    transform_paths: list[Path],
    object_ids: list[str],
    object_masks: list[np.ndarray],
    object_reference_frames: list[int],
    background_path: Path,
    camera_depth_path: Path,
    source_stage: Path,
    output_root: Path,
    fps: float = 16.0,
    batch_size: int = 9,
    point_size: float = 0.005,
    device_name: str = "cuda",
    expected_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    import torch
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes

    count = len(object_ids)
    if not (
        len(mesh_paths)
        == len(transform_paths)
        == len(object_masks)
        == len(object_reference_frames)
        == count
    ):
        raise ValueError("Per-object render inputs have different lengths")
    if not count:
        raise ValueError("At least one object is required")
    with np.load(camera_depth_path) as archive:
        intrinsics_np = np.asarray(archive["intrinsics"], dtype=np.float32)
        extrinsics_np = np.asarray(archive["extrinsics"], dtype=np.float32)
        height, width = np.asarray(
            archive["image_size"],
            dtype=np.int64,
        ).tolist()
    if expected_size is not None and expected_size != (width, height):
        raise ValueError(
            f"Requested control size {expected_size} differs from canonical "
            f"camera size {(width, height)}; the controls do not rescale "
            "camera geometry"
        )
    frame_count = len(intrinsics_np)
    for object_id, masks in zip(object_ids, object_masks):
        if masks.shape != (frame_count, height, width):
            raise ValueError(
                f"Masks for {object_id!r} have shape {masks.shape}, expected "
                f"{(frame_count, height, width)}"
            )
    with np.load(background_path) as archive:
        background_points_np = np.asarray(
            archive["points"],
            dtype=np.float32,
        )
    meshes_np = [_load_mesh(path) for path in mesh_paths]
    transforms_np: list[np.ndarray] = []
    for path in transform_paths:
        with np.load(path) as archive:
            transforms_np.append(
                np.asarray(archive["canonical_to_world"], dtype=np.float32)
            )
    if any(len(value) != frame_count for value in transforms_np):
        raise ValueError("Object transform frame counts differ from camera")

    device = torch.device(device_name)
    intrinsics = torch.from_numpy(intrinsics_np).to(device)
    extrinsics = torch.from_numpy(extrinsics_np).to(device)
    background_points = torch.from_numpy(background_points_np).to(device)
    vertices_t = [
        torch.from_numpy(vertices.astype(np.float32)).to(device)
        for vertices, _ in meshes_np
    ]
    faces_t = [
        torch.from_numpy(faces.astype(np.int64)).to(device)
        for _, faces in meshes_np
    ]

    raw_depth_parts: list[np.ndarray] = []
    front_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    second_depth_parts: list[np.ndarray] = []
    gap_parts: list[np.ndarray] = []
    object_visible_parts: list[np.ndarray] = []
    validity_parts: list[np.ndarray] = []
    object_raw_mask_reference: list[np.ndarray | None] = [
        None for _ in object_ids
    ]
    actual_batch_size = max(1, int(batch_size))
    for start in range(0, frame_count, actual_batch_size):
        end = min(start + actual_batch_size, frame_count)
        background_depth = _render_point_depth_batch(
            background_points,
            intrinsics[start:end],
            extrinsics[start:end],
            (height, width),
            radius=point_size,
        )
        entity_depths = [background_depth.detach().cpu().numpy()]
        for object_index, (vertices, faces) in enumerate(
            zip(vertices_t, faces_t)
        ):
            frame_meshes = []
            base_color = (
                torch.tensor([0.8, 0.7, 0.5], device=device)[None]
                .expand(len(vertices), 3)
            )
            vertices_h = torch.cat(
                [
                    vertices,
                    torch.ones((len(vertices), 1), device=device),
                ],
                dim=1,
            )
            for frame in range(start, end):
                transform = torch.from_numpy(
                    transforms_np[object_index][frame]
                ).to(device)
                transformed = (transform @ vertices_h.T).T[:, :3]
                frame_meshes.append(
                    Meshes(
                        verts=[transformed],
                        faces=[faces],
                        textures=TexturesVertex(
                            verts_features=base_color[None]
                        ),
                    )
                )
            object_depth, object_mask = _render_mesh_depth_batch(
                frame_meshes,
                intrinsics[start:end],
                extrinsics[start:end],
                (height, width),
            )
            object_valid = object_mask & (object_depth > 0)
            entity_depths.append(object_depth.detach().cpu().numpy())
            reference_frame = int(object_reference_frames[object_index])
            if start <= reference_frame < end:
                object_raw_mask_reference[object_index] = (
                    object_valid[reference_frame - start]
                    .detach()
                    .cpu()
                    .numpy()
                )
        composed = compose_entity_depths(np.stack(entity_depths, axis=0))
        raw_depth_parts.append(composed["camera_z"])
        front_parts.append(composed["front_id"])
        second_parts.append(composed["second_id"])
        second_depth_parts.append(composed["second_depth"])
        gap_parts.append(composed["depth_gap"])
        object_visible_parts.append(composed["object_visible"])
        validity_parts.append(composed["validity"])

    raw_depth = np.concatenate(raw_depth_parts)
    front_id = np.concatenate(front_parts)
    second_id = np.concatenate(second_parts)
    second_depth = np.concatenate(second_depth_parts)
    depth_gap = np.concatenate(gap_parts)
    object_visible = np.concatenate(object_visible_parts, axis=0)
    validity = np.concatenate(validity_parts)
    encoded, encoding = encode_inverse_depth(raw_depth, front_id)

    output_root.mkdir(parents=True, exist_ok=True)
    depth_video = output_root / "vace_depth.mkv"
    write_lossless_gray_video(depth_video, encoded, fps)
    depth_preview = output_root / "vace_depth_preview.mp4"
    _write_preview(depth_preview, encoded, fps)
    first_depth = output_root / "vace_depth_frame000.png"
    Image.fromarray(encoded[0], mode="L").save(first_depth)
    white = np.full_like(encoded, 255, dtype=np.uint8)
    vace_mask = output_root / "vace_video_mask.mkv"
    write_lossless_binary_video(vace_mask, white, fps)
    edit = object_edit_mask(object_masks)
    object_edit_video = output_root / "object_edit_mask.mkv"
    write_lossless_binary_video(object_edit_video, edit, fps)
    reference = output_root / "vace_reference_image.png"
    Image.open(source_stage / "rgb" / "000000.png").convert("RGB").save(
        reference
    )

    sidecar = output_root / "raw_sidecars.npz"
    fill_source = np.zeros_like(front_id, dtype=np.uint8)
    fill_source[front_id == 0] = 1
    fill_source[front_id > 0] = 2
    np.savez_compressed(
        sidecar,
        camera_z=raw_depth.astype(np.float32),
        validity=validity,
        front_id=front_id.astype(np.int16),
        second_id=second_id.astype(np.int16),
        second_depth=second_depth.astype(np.float32),
        depth_gap=depth_gap.astype(np.float32),
        object_visible=object_visible,
        fill_source=fill_source,
        vace_video_mask=white,
        object_edit_mask=edit,
        source_object_masks=np.stack(object_masks).astype(bool),
        object_ids=np.asarray(object_ids, dtype=np.str_),
        intrinsics=intrinsics_np,
        extrinsics=extrinsics_np,
        invalid_gray=np.asarray(128, dtype=np.uint8),
        inverse_far=np.asarray(encoding["inverse_far"], dtype=np.float32),
        inverse_near=np.asarray(encoding["inverse_near"], dtype=np.float32),
        fps=np.asarray(fps, dtype=np.float32),
    )
    encoding_path = output_root / "depth_encoding.json"
    atomic_write_json(
        encoding_path,
        {
            "schema_version": DEPTH_ENCODING_SCHEMA_VERSION,
            "authoritative_sidecar": str(sidecar),
            "authoritative_sidecar_sha256": sha256_file(sidecar),
            "derived_depth_video": str(depth_video),
            "derived_depth_video_sha256": sha256_file(depth_video),
            "definition": (
                "OpenCV camera-axis Z from the shared entity z-buffer"
            ),
            "validity_definition": (
                "validity boolean; do not infer invalidity from gray value"
            ),
            "encoding": encoding,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
            "reencoding": (
                "camera_z + validity + saved inverse range are sufficient "
                "to change invalid gray or regenerate the control video"
            ),
        },
    )
    silhouettes: dict[str, Any] = {}
    for index, object_id in enumerate(object_ids):
        rendered = object_raw_mask_reference[index]
        if rendered is None:
            raise RuntimeError(f"No reference-frame render for {object_id}")
        reference_frame = int(object_reference_frames[index])
        target = object_masks[index][reference_frame]
        intersection = np.logical_and(rendered, target).sum()
        union = np.logical_or(rendered, target).sum()
        silhouettes[object_id] = {
            "reference_frame": reference_frame,
            "reference_iou": float(intersection / max(union, 1)),
            "frame0_iou": (
                float(intersection / max(union, 1))
                if reference_frame == 0
                else None
            ),
            "rendered_pixels": int(rendered.sum()),
            "target_pixels": int(target.sum()),
        }
    manifest = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "status": "complete",
        "frames": frame_count,
        "size": [width, height],
        "fps": fps,
        "object_ids": object_ids,
        "entity_ids": {
            "invalid": -1,
            "background": 0,
            "objects": {
                object_id: index + 1
                for index, object_id in enumerate(object_ids)
            },
        },
        "composition": (
            "background and every object compete in one camera-Z z-buffer"
        ),
        "encoding": encoding,
        "mask_semantics": {
            "vace_video_mask": (
                "lossless FFV1, exact all-white 255: regenerate full frame"
            ),
            "object_edit_mask": (
                "lossless FFV1, exact {0,255} union of tracked objects"
            ),
        },
        "valid_depth_fraction": float(validity.mean()),
        "silhouettes": silhouettes,
        "outputs": {
            "vace_depth": str(depth_video),
            "vace_depth_preview": str(depth_preview),
            "vace_depth_frame000": str(first_depth),
            "vace_video_mask": str(vace_mask),
            "object_edit_mask": str(object_edit_video),
            "vace_reference_image": str(reference),
            "raw_sidecars": str(sidecar),
            "depth_encoding": str(encoding_path),
        },
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def _load_runtime_masks(
    source_stage: Path,
    object_ids: list[str],
) -> list[np.ndarray]:
    output = []
    for object_id in object_ids:
        paths = image_files(source_stage / "masks" / object_id)
        if not paths:
            raise FileNotFoundError(f"No canonical masks for {object_id!r}")
        output.append(
            np.stack(
                [
                    np.asarray(Image.open(path).convert("L")) > 127
                    for path in paths
                ]
            )
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, action="append", required=True)
    parser.add_argument(
        "--transform",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--object-id", action="append", required=True)
    parser.add_argument(
        "--reference-frame",
        type=int,
        action="append",
        required=True,
    )
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--camera-depth", type=Path, required=True)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--point-size", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = render_case_controls(
        mesh_paths=args.mesh,
        transform_paths=args.transform,
        object_ids=args.object_id,
        object_masks=_load_runtime_masks(args.source_stage, args.object_id),
        object_reference_frames=args.reference_frame,
        background_path=args.background,
        camera_depth_path=args.camera_depth,
        source_stage=args.source_stage,
        output_root=args.output_root,
        fps=args.fps,
        batch_size=args.batch_size,
        point_size=args.point_size,
        device_name=args.device,
        expected_size=(args.width, args.height),
    )
    print(args.output_root / "manifest.json")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())


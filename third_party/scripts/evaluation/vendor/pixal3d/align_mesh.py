#!/usr/bin/env python3
"""Align Pixal3D visible-surface coordinates to the frame-0 scene geometry."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


MESH_TO_CORRESPONDENCE = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--maximum-pairs", type=int, default=50000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def solve_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    covariance = target_zero.T @ source_zero / len(source)
    left, singular, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right
    source_variance = float(np.mean(np.sum(source_zero**2, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) / source_variance)
    translation = target_mean - scale * rotation @ source_mean
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (
        transform[:3, :3] @ points.T
    ).T + transform[:3, 3]


def robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
    iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.ones(len(source), dtype=bool)
    minimum = max(8, int(np.ceil(len(source) * 0.5)))
    transform = np.eye(4)
    for _ in range(iterations):
        transform = solve_similarity(source[active], target[active])
        residual = np.linalg.norm(apply_transform(source, transform) - target, axis=1)
        median = float(np.median(residual[active]))
        mad = float(np.median(np.abs(residual[active] - median)))
        threshold = median + max(3.0 * 1.4826 * mad, 1e-6)
        updated = residual <= threshold
        if updated.sum() < minimum:
            selected = np.argsort(residual)[:minimum]
            updated = np.zeros(len(source), dtype=bool)
            updated[selected] = True
        if np.array_equal(updated, active):
            break
        active = updated
    return solve_similarity(source[active], target[active]), active


def main() -> None:
    args = parse_args()
    for path in (
        args.mesh,
        args.correspondence,
        args.camera_contract,
        args.target,
        args.motion,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(args.correspondence) as archive:
        virtual_xy = np.asarray(archive["crop_xy"], dtype=np.float64)
        pixal_points = np.asarray(archive["coord_glb"], dtype=np.float64)
        correspondence_resolution = int(archive["render_resolution"])
        processed_bbox = np.asarray(
            archive["crop_bbox_xyxy_processed"], dtype=np.float64
        )
    with np.load(args.camera_contract) as archive:
        virtual_to_full = np.asarray(
            archive["virtual_to_full_homography"], dtype=np.float64
        )
        virtual_resolution = int(archive["crop_resolution"])
        input_space = str(archive["pixal_input_space"].tolist())
    with np.load(args.target) as archive:
        world_dense = np.asarray(
            archive["world_points_from_depth"], dtype=np.float64
        )
        object_mask = np.asarray(archive["frame0_mask"], dtype=bool)
    with np.load(args.motion) as archive:
        motion = np.asarray(archive["world_transforms"], dtype=np.float64)
        pose_gate_passed = bool(archive["pose_gate_passed"])
    if not 0 <= args.reference_frame < len(motion):
        raise ValueError(
            f"reference frame {args.reference_frame} is outside "
            f"[0,{len(motion) - 1}]"
        )
    if processed_bbox.shape == (4,):
        left, top, right, bottom = processed_bbox.tolist()
        virtual_xy = np.stack(
            [
                left
                + (virtual_xy[:, 0] + 0.5)
                * (right - left)
                / correspondence_resolution
                - 0.5,
                top
                + (virtual_xy[:, 1] + 0.5)
                * (bottom - top)
                / correspondence_resolution
                - 0.5,
            ],
            axis=1,
        )
    else:
        virtual_xy = (
            (virtual_xy + 0.5)
            * virtual_resolution
            / correspondence_resolution
            - 0.5
        )
    if input_space == "full_frame_rgba" and processed_bbox.shape == (4,):
        full_xy = virtual_xy
    else:
        virtual_h = np.concatenate(
            [virtual_xy, np.ones((len(virtual_xy), 1))],
            axis=1,
        )
        full_h = (virtual_to_full @ virtual_h.T).T
        full_xy = full_h[:, :2] / full_h[:, 2:3]
    rounded = np.rint(full_xy).astype(np.int64)
    height, width = object_mask.shape
    in_frame = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    target_points = np.full_like(pixal_points, np.nan)
    valid_indices = np.flatnonzero(in_frame)
    target_points[valid_indices] = world_dense[
        rounded[valid_indices, 1],
        rounded[valid_indices, 0],
    ]
    valid = in_frame & np.isfinite(target_points).all(axis=1)
    valid[valid_indices] &= object_mask[
        rounded[valid_indices, 1],
        rounded[valid_indices, 0],
    ]
    source = pixal_points[valid]
    target = target_points[valid]
    pixels = rounded[valid]
    if len(source) < 1500:
        raise ValueError(
            f"Need at least 1500 Pixal3D/scene correspondences, found {len(source)}"
        )
    order = np.lexsort((pixels[:, 0], pixels[:, 1]))
    if len(order) > args.maximum_pairs:
        order = order[
            np.rint(
                np.linspace(0, len(order) - 1, args.maximum_pairs)
            ).astype(np.int64)
        ]
    source = source[order]
    target = target[order]
    pixels = pixels[order]
    holdout = np.arange(len(source)) % 5 == 0
    if holdout.sum() < 500 or (~holdout).sum() < 1000:
        raise ValueError("Insufficient train/holdout correspondence split")
    transform, train_inliers = robust_similarity(
        source[~holdout],
        target[~holdout],
    )
    holdout_residual = np.linalg.norm(
        apply_transform(source[holdout], transform) - target[holdout],
        axis=1,
    )
    target_extent = np.quantile(target, 0.98, axis=0) - np.quantile(
        target, 0.02, axis=0
    )
    diameter = max(float(np.linalg.norm(target_extent)), 1e-8)
    holdout_median = float(np.median(holdout_residual) / diameter)
    holdout_p90 = float(np.quantile(holdout_residual, 0.90) / diameter)
    holdout_p95 = float(np.quantile(holdout_residual, 0.95) / diameter)
    holdout_inlier_coverage = float(
        np.mean(holdout_residual / diameter <= 0.05)
    )
    alignment_gate_passed = bool(
        holdout_median <= 0.03
        and holdout_p90 <= 0.10
        and holdout_inlier_coverage >= 0.80
    )
    mesh_to_world_reference = transform @ MESH_TO_CORRESPONDENCE
    mesh_to_world_frame0 = (
        np.linalg.inv(motion[args.reference_frame])
        @ mesh_to_world_reference
    )
    correspondence_to_world_frame0 = (
        np.linalg.inv(motion[args.reference_frame]) @ transform
    )
    transforms = motion @ mesh_to_world_frame0[None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        canonical_to_world_frame0=mesh_to_world_frame0.astype(np.float32),
        canonical_to_world_reference=mesh_to_world_reference.astype(np.float32),
        reference_frame=np.asarray(args.reference_frame, dtype=np.int32),
        correspondence_to_world_frame0=correspondence_to_world_frame0.astype(
            np.float32
        ),
        correspondence_to_world_reference=transform.astype(np.float32),
        mesh_to_correspondence=MESH_TO_CORRESPONDENCE.astype(np.float32),
        canonical_to_world=transforms.astype(np.float32),
        relative_object_motion=motion.astype(np.float32),
        pose_gate_passed=np.asarray(pose_gate_passed),
        alignment_gate_passed=np.asarray(alignment_gate_passed),
        train_pairs=np.asarray((~holdout).sum(), dtype=np.int32),
        train_inliers=np.asarray(train_inliers.sum(), dtype=np.int32),
        holdout_pairs=np.asarray(holdout.sum(), dtype=np.int32),
        holdout_median_over_diameter=np.asarray(
            holdout_median, dtype=np.float32
        ),
        holdout_p90_over_diameter=np.asarray(
            holdout_p90, dtype=np.float32
        ),
        holdout_p95_over_diameter=np.asarray(holdout_p95, dtype=np.float32),
        holdout_inlier_coverage_at_0_05=np.asarray(
            holdout_inlier_coverage, dtype=np.float32
        ),
        pixal_mesh_sha256=np.asarray(sha256(args.mesh), dtype=np.str_),
        convention=np.asarray(
            "Pixal3D exported GLB coordinates to shared TAPIP world",
            dtype=np.str_,
        ),
    )


if __name__ == "__main__":
    main()

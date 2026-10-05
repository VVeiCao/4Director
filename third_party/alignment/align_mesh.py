#!/usr/bin/env python3
"""Deduplicated blocked-holdout Pixal3D-to-MegaSAM Sim3 alignment."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluation.vendor.pixal3d.align_mesh import (
    MESH_TO_CORRESPONDENCE,
    apply_transform,
    robust_similarity,
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
    parser.add_argument("--erosion-pixels", type=int, default=0)
    parser.add_argument("--median-threshold", type=float, default=0.15)
    parser.add_argument("--p90-threshold", type=float, default=0.30)
    parser.add_argument("--p95-threshold", type=float, default=0.50)
    parser.add_argument("--minimum-unique-pairs", type=int, default=800)
    parser.add_argument("--maximum-train", type=int, default=40_000)
    parser.add_argument("--maximum-holdout", type=int, default=10_000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_limit(
    indices: np.ndarray,
    pixels: np.ndarray,
    maximum: int,
) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    selected_pixels = pixels[indices]
    hashed = (
        selected_pixels[:, 0].astype(np.int64) * 73_856_093
        ^ selected_pixels[:, 1].astype(np.int64) * 19_349_663
    )
    order = np.argsort(hashed, kind="stable")
    return indices[order[:maximum]]


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
        full_xy = np.asarray(archive["crop_xy"], dtype=np.float64)
        source = np.asarray(archive["coord_glb"], dtype=np.float64)
        alpha = (
            np.asarray(archive["source_alpha"], dtype=np.float64)
            if "source_alpha" in archive.files
            else np.ones(len(source), dtype=np.float64)
        )
        resolution = int(archive["render_resolution"])
        bbox = np.asarray(
            archive["crop_bbox_xyxy_processed"],
            dtype=np.float64,
        )
    if bbox.shape == (4,):
        left, top, right, bottom = bbox.tolist()
        full_xy = np.stack(
            [
                left
                + (full_xy[:, 0] + 0.5)
                * (right - left)
                / resolution
                - 0.5,
                top
                + (full_xy[:, 1] + 0.5)
                * (bottom - top)
                / resolution
                - 0.5,
            ],
            axis=1,
        )
    rounded = np.rint(full_xy).astype(np.int64)
    with np.load(args.target) as archive:
        world = np.asarray(
            archive["world_points_from_depth"],
            dtype=np.float64,
        )
        original_mask = np.asarray(archive["frame0_mask"], dtype=np.uint8)
        extrinsic = np.asarray(archive["extrinsic"], dtype=np.float64)
    height, width = original_mask.shape
    in_frame = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    target = np.full(source.shape, np.nan, dtype=np.float64)
    ids = np.flatnonzero(in_frame)
    target[ids] = world[rounded[ids, 1], rounded[ids, 0]]
    finite = in_frame & np.isfinite(source).all(axis=1)
    finite &= np.isfinite(target).all(axis=1)
    finite[ids] &= original_mask[rounded[ids, 1], rounded[ids, 0]] > 0
    source = source[finite]
    target = target[finite]
    full_xy = full_xy[finite]
    rounded = rounded[finite]
    alpha = alpha[finite]

    # One pair per full-frame MegaSAM target pixel.
    pixel_distance = np.sum(np.square(full_xy - rounded), axis=1)
    order = np.lexsort(
        (
            np.arange(len(source)),
            pixel_distance,
            -alpha,
            rounded[:, 0],
            rounded[:, 1],
        )
    )
    ordered_pixels = rounded[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(
        ordered_pixels[1:] != ordered_pixels[:-1],
        axis=1,
    )
    keep = order[first]
    source = source[keep]
    target = target[keep]
    pixels = rounded[keep]
    full_unique_count = len(source)

    fit_mask = original_mask
    if args.erosion_pixels:
        size = args.erosion_pixels * 2 + 1
        fit_mask = cv2.erode(
            fit_mask,
            np.ones((size, size), dtype=np.uint8),
        )
    active = fit_mask[pixels[:, 1], pixels[:, 0]] > 0
    source = source[active]
    target = target[active]
    pixels = pixels[active]
    retention = len(source) / max(full_unique_count, 1)
    cells = pixels // 8
    unique_cells = np.unique(cells, axis=0)
    if len(source) < args.minimum_unique_pairs:
        raise ValueError(
            "Insufficient alignment support after erosion: "
            f"pairs={len(source)}, retention={retention:.3f}, "
            f"cells={len(unique_cells)}"
        )
    unique_cell_hash = (
        unique_cells[:, 0].astype(np.int64) * 73_856_093
        ^ unique_cells[:, 1].astype(np.int64) * 19_349_663
    )
    cell_order = np.argsort(unique_cell_hash, kind="stable")
    holdout_cell_count = min(
        len(unique_cells) - 1,
        max(1, int(np.ceil(len(unique_cells) * 0.20))),
    )
    holdout_cells = unique_cells[cell_order[:holdout_cell_count]]
    holdout = np.zeros(len(cells), dtype=bool)
    for cell in holdout_cells:
        holdout |= np.all(cells == cell, axis=1)
    train_ids = deterministic_limit(
        np.flatnonzero(~holdout),
        pixels,
        args.maximum_train,
    )
    holdout_ids = deterministic_limit(
        np.flatnonzero(holdout),
        pixels,
        args.maximum_holdout,
    )
    if len(train_ids) < 3 or len(holdout_ids) < 1:
        raise ValueError(
            f"Insufficient blocked split: {len(train_ids)}/{len(holdout_ids)}"
        )
    transform, train_inliers = robust_similarity(
        source[train_ids],
        target[train_ids],
    )
    predicted = apply_transform(source[holdout_ids], transform)
    target_holdout = target[holdout_ids]
    residual = np.linalg.norm(predicted - target_holdout, axis=1)
    extent = np.quantile(target, 0.98, axis=0) - np.quantile(
        target,
        0.02,
        axis=0,
    )
    diameter = max(float(np.linalg.norm(extent)), 1e-8)
    normalized = residual / diameter
    median = float(np.median(normalized))
    p90 = float(np.quantile(normalized, 0.90))
    p95 = float(np.quantile(normalized, 0.95))
    coverage = float(np.mean(normalized <= 0.05))
    alignment_gate_passed = bool(
        median <= args.median_threshold
        and p90 <= args.p90_threshold
        and p95 <= args.p95_threshold
    )

    predicted_h = np.concatenate(
        [predicted, np.ones((len(predicted), 1))],
        axis=1,
    )
    target_h = np.concatenate(
        [target_holdout, np.ones((len(target_holdout), 1))],
        axis=1,
    )
    predicted_z = (extrinsic @ predicted_h.T).T[:, 2]
    target_z = (extrinsic @ target_h.T).T[:, 2]
    z_scale = max(float(np.median(np.abs(target_z))), 1e-8)
    z_error = np.abs(predicted_z - target_z) / z_scale

    with np.load(args.motion) as archive:
        motion = np.asarray(archive["world_transforms"], dtype=np.float64)
        pose_gate_passed = bool(archive["pose_gate_passed"])
    if not 0 <= args.reference_frame < len(motion):
        raise ValueError("Reference frame is outside motion sequence")
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
        erosion_pixels=np.asarray(args.erosion_pixels, dtype=np.int32),
        raw_pair_count=np.asarray(
            np.count_nonzero(finite),
            dtype=np.int32,
        ),
        raw_unique_pair_count=np.asarray(full_unique_count, dtype=np.int32),
        unique_pair_count=np.asarray(len(source), dtype=np.int32),
        retained_pair_count=np.asarray(len(source), dtype=np.int32),
        retained_fraction=np.asarray(retention, dtype=np.float32),
        occupied_cells=np.asarray(len(unique_cells), dtype=np.int32),
        train_pairs=np.asarray(len(train_ids), dtype=np.int32),
        train_inliers=np.asarray(train_inliers.sum(), dtype=np.int32),
        holdout_pairs=np.asarray(len(holdout_ids), dtype=np.int32),
        holdout_median_over_diameter=np.asarray(median, dtype=np.float32),
        holdout_p90_over_diameter=np.asarray(p90, dtype=np.float32),
        holdout_p95_over_diameter=np.asarray(p95, dtype=np.float32),
        holdout_inlier_coverage_at_0_05=np.asarray(
            coverage,
            dtype=np.float32,
        ),
        holdout_z_median=np.asarray(np.median(z_error), dtype=np.float32),
        holdout_z_p90=np.asarray(np.quantile(z_error, 0.90), dtype=np.float32),
        pixal_mesh_sha256=np.asarray(file_sha256(args.mesh), dtype=np.str_),
        convention=np.asarray(
            "V7 deduplicated blocked-holdout Pixal Sim3",
            dtype=np.str_,
        ),
    )


if __name__ == "__main__":
    main()

"""Strict 81-matrix rigid/similarity trajectory contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryContract:
    frame_count: int = 81
    convention: str = "canonical_to_world"

    def __post_init__(self) -> None:
        if self.frame_count != 81:
            raise ValueError("production trajectories contain exactly 81 matrices")
        if self.convention not in {"canonical_to_world", "world_to_camera"}:
            raise ValueError(f"unsupported trajectory convention: {self.convention}")


PRODUCTION_TRAJECTORY = TrajectoryContract()


def validate_trajectory_matrices(
    matrices: np.ndarray,
    *,
    contract: TrajectoryContract = PRODUCTION_TRAJECTORY,
    tolerance: float = 5e-4,
) -> np.ndarray:
    value = np.asarray(matrices, dtype=np.float64)
    expected = (contract.frame_count, 4, 4)
    if value.shape != expected:
        raise ValueError(
            f"trajectory must have shape {expected}, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("trajectory contains NaN or infinity")
    bottom = np.broadcast_to(
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        (contract.frame_count, 4),
    )
    if not np.allclose(value[:, 3, :], bottom, atol=tolerance, rtol=0.0):
        raise ValueError("trajectory matrices have invalid homogeneous bottom rows")
    linear = value[:, :3, :3]
    axis_scales = np.linalg.norm(linear, axis=1)
    scales = axis_scales.mean(axis=1)
    if np.any(scales <= np.finfo(np.float64).eps):
        raise ValueError("trajectory matrices must have positive uniform scale")
    if not np.allclose(
        axis_scales,
        scales[:, None],
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("trajectory matrices contain non-uniform scale")
    rotations = linear / scales[:, None, None]
    gram = rotations @ np.swapaxes(rotations, 1, 2)
    if not np.allclose(
        gram,
        np.broadcast_to(np.eye(3), gram.shape),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("trajectory rotations are not orthonormal after scale removal")
    determinants = np.linalg.det(rotations)
    if not np.allclose(
        determinants,
        np.ones(contract.frame_count),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("trajectory rotations must have determinant +1")
    return np.ascontiguousarray(value.astype(np.float32))


def static_trajectory(
    matrix: np.ndarray | None = None,
    *,
    contract: TrajectoryContract = PRODUCTION_TRAJECTORY,
) -> np.ndarray:
    pose = (
        np.eye(4, dtype=np.float32)
        if matrix is None
        else np.asarray(matrix, dtype=np.float32)
    )
    if pose.shape != (4, 4):
        raise ValueError("static trajectory pose must be a 4x4 matrix")
    repeated = np.repeat(pose[None], contract.frame_count, axis=0)
    return validate_trajectory_matrices(repeated, contract=contract)

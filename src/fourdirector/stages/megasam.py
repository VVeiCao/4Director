"""Pinned video MegaSAM provider and strict canonical NPZ validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from .external import ExternalCommandRunner
from .media import PRODUCTION_MEDIA, MediaContract

VIDEO_COMPONENTS = (
    "moge2",
    "unidepth_v2",
    "megasam_droid",
    "raft",
    "cvd",
)


@dataclass(frozen=True)
class ComponentEntrypointLock:
    component: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if self.component not in VIDEO_COMPONENTS:
            raise ValueError(f"unsupported MegaSAM component: {self.component}")
        digest = self.sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("component sha256 must be a 64-character hex digest")

    def verify(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        actual = sha256_file(self.path)
        if actual != self.sha256.lower():
            raise RuntimeError(
                f"{self.component} entrypoint lock mismatch: expected {self.sha256}, got {actual}"
            )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VideoMegaSAMConfig:
    """Immutable lock for the vendor's complete production orchestrator."""

    python_executable: Path
    orchestrator_script: Path
    orchestrator_sha256: str
    component_entrypoints: tuple[ComponentEntrypointLock, ...]
    timeout_seconds: float = 7_200.0
    resolution: int = 384 * 512
    components: tuple[str, ...] = VIDEO_COMPONENTS
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.components != VIDEO_COMPONENTS:
            raise ValueError("video provider is locked to MoGe2+UniDepthV2+MegaSAM/DROID+RAFT+CVD")
        locked_components = tuple(lock.component for lock in self.component_entrypoints)
        if locked_components != VIDEO_COMPONENTS:
            raise ValueError(
                "component entrypoints must lock, in order, MoGe2+UniDepthV2+MegaSAM/DROID+RAFT+CVD"
            )
        digest = self.orchestrator_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("orchestrator_sha256 must be a 64-character hex digest")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.resolution != 384 * 512:
            raise ValueError("production MegaSAM resolution is locked to 384*512")
        environment_names = [name for name, _ in self.environment]
        if len(environment_names) != len(set(environment_names)):
            raise ValueError("MegaSAM environment contains duplicate names")

    def verify(self) -> None:
        if not self.python_executable.is_file():
            raise FileNotFoundError(self.python_executable)
        if not self.orchestrator_script.is_file():
            raise FileNotFoundError(self.orchestrator_script)
        actual = sha256_file(self.orchestrator_script)
        if actual != self.orchestrator_sha256.lower():
            raise RuntimeError(
                "MegaSAM orchestrator lock mismatch: "
                f"expected {self.orchestrator_sha256}, got {actual}"
            )
        for entrypoint in self.component_entrypoints:
            entrypoint.verify()


def _as_float32(name: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _broadcast_intrinsics(
    intrinsics: np.ndarray,
    *,
    frame_count: int,
) -> np.ndarray:
    value = np.asarray(intrinsics, dtype=np.float32)
    if value.shape == (3, 3):
        value = np.broadcast_to(value, (frame_count, 3, 3)).copy()
    return value


def _resize_camera_z(
    depths: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    source_height, source_width = depths.shape[1:]
    if (source_height, source_width) == (height, width):
        return np.ascontiguousarray(depths)
    if source_height < 1 or source_width < 1:
        raise ValueError("MegaSAM depths have an empty spatial size")
    sample_y = (np.arange(height, dtype=np.float64) + 0.5) * source_height / height - 0.5
    sample_x = (np.arange(width, dtype=np.float64) + 0.5) * source_width / width - 0.5
    sample_y = np.clip(sample_y, 0.0, source_height - 1.0)
    sample_x = np.clip(sample_x, 0.0, source_width - 1.0)
    y0 = np.floor(sample_y).astype(np.int64)
    x0 = np.floor(sample_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    weight_y = (sample_y - y0).astype(np.float32)[None, :, None]
    weight_x = (sample_x - x0).astype(np.float32)[None, None, :]
    top_left = depths[:, y0[:, None], x0[None, :]]
    top_right = depths[:, y0[:, None], x1[None, :]]
    bottom_left = depths[:, y1[:, None], x0[None, :]]
    bottom_right = depths[:, y1[:, None], x1[None, :]]
    return np.ascontiguousarray(
        (
            top_left * (1.0 - weight_y) * (1.0 - weight_x)
            + top_right * (1.0 - weight_y) * weight_x
            + bottom_left * weight_y * (1.0 - weight_x)
            + bottom_right * weight_y * weight_x
        ).astype(np.float32)
    )


def resample_video_geometry(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    *,
    contract: MediaContract = PRODUCTION_MEDIA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift native MegaSAM camera-Z to the locked 832x480 media contract."""

    depth_value = _as_float32("depths", depths)
    if depth_value.ndim != 3 or depth_value.shape[0] != contract.num_frames:
        raise ValueError(
            "depths must have shape "
            f"({contract.num_frames}, H, W), got {depth_value.shape}"
        )
    source_height, source_width = depth_value.shape[1:]
    intrinsic_value = validate_intrinsics(
        _broadcast_intrinsics(intrinsics, frame_count=contract.num_frames),
        frame_count=contract.num_frames,
    )
    if (source_height, source_width) != (contract.height, contract.width):
        scaled = intrinsic_value.copy()
        scaled[:, 0, :] *= contract.width / source_width
        scaled[:, 1, :] *= contract.height / source_height
        intrinsic_value = scaled
        depth_value = _resize_camera_z(
            depth_value,
            height=contract.height,
            width=contract.width,
        )
    return validate_video_geometry(
        depth_value,
        intrinsic_value,
        w2c,
        contract=contract,
    )


def validate_intrinsics(
    intrinsics: np.ndarray,
    *,
    frame_count: int = 81,
) -> np.ndarray:
    value = _as_float32("intrinsics", _broadcast_intrinsics(intrinsics, frame_count=frame_count))
    if value.shape != (frame_count, 3, 3):
        raise ValueError(f"intrinsics must have shape {(frame_count, 3, 3)}, got {value.shape}")
    if np.any(value[:, 0, 0] <= 0) or np.any(value[:, 1, 1] <= 0):
        raise ValueError("intrinsics must have positive focal lengths")
    expected = np.broadcast_to(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        (frame_count, 3),
    )
    if not np.allclose(value[:, 2, :], expected, atol=1e-4, rtol=0.0):
        raise ValueError("intrinsics must use the canonical homogeneous last row")
    return np.ascontiguousarray(value)


def validate_w2c(
    w2c: np.ndarray,
    *,
    frame_count: int = 81,
    tolerance: float = 5e-3,
) -> np.ndarray:
    value = _as_float32("w2c", w2c)
    if value.shape != (frame_count, 4, 4):
        raise ValueError(f"w2c must have shape {(frame_count, 4, 4)}, got {value.shape}")
    expected_bottom = np.broadcast_to(
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (frame_count, 4),
    )
    if not np.allclose(value[:, 3, :], expected_bottom, atol=tolerance, rtol=0.0):
        raise ValueError("w2c matrices have invalid homogeneous bottom rows")
    rotations = value[:, :3, :3].astype(np.float64)
    gram = rotations @ np.swapaxes(rotations, 1, 2)
    identity = np.broadcast_to(np.eye(3), gram.shape)
    if not np.allclose(gram, identity, atol=tolerance, rtol=tolerance):
        raise ValueError("w2c rotations are not orthonormal")
    determinant = np.linalg.det(rotations)
    if not np.allclose(
        determinant,
        np.ones(frame_count),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("w2c rotations must have determinant +1")
    return np.ascontiguousarray(value)


def validate_video_geometry(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    *,
    contract: MediaContract = PRODUCTION_MEDIA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_value = _as_float32("depths", depths)
    expected_depth_shape = (
        contract.num_frames,
        contract.height,
        contract.width,
    )
    if depth_value.shape != expected_depth_shape:
        raise ValueError(f"depths must have shape {expected_depth_shape}, got {depth_value.shape}")
    if np.any(depth_value < 0):
        raise ValueError("depths must be non-negative camera-Z")
    if np.any(np.count_nonzero(depth_value > 0, axis=(1, 2)) == 0):
        raise ValueError("every video frame must contain positive camera-Z")
    intrinsic_value = validate_intrinsics(
        intrinsics,
        frame_count=contract.num_frames,
    )
    w2c_value = validate_w2c(w2c, frame_count=contract.num_frames)
    return (
        np.ascontiguousarray(depth_value),
        intrinsic_value,
        w2c_value,
    )


def _pick(
    archive: Any,
    names: Sequence[str],
    *,
    label: str,
) -> tuple[str, np.ndarray]:
    for name in names:
        if name in archive.files:
            return name, np.asarray(archive[name])
    raise ValueError(f"MegaSAM NPZ is missing {label}; accepted keys: {list(names)}")


def canonicalize_video_npz(
    path: str | Path,
    *,
    contract: MediaContract = PRODUCTION_MEDIA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        _, depths = _pick(archive, ("depths",), label="depths")
        _, intrinsics = _pick(
            archive,
            ("intrinsics", "intrinsic"),
            label="intrinsics",
        )
        pose_key, poses = _pick(
            archive,
            ("w2c", "extrinsics", "cam_c2w"),
            label="world-to-camera poses",
        )
    poses = _as_float32(pose_key, poses)
    if pose_key == "cam_c2w":
        if poses.shape != (contract.num_frames, 4, 4):
            raise ValueError(
                f"cam_c2w must have shape {(contract.num_frames, 4, 4)}, got {poses.shape}"
            )
        try:
            poses = np.linalg.inv(poses).astype(np.float32)
        except np.linalg.LinAlgError as exc:
            raise ValueError("cam_c2w contains a singular matrix") from exc
    return resample_video_geometry(
        depths,
        intrinsics,
        poses,
        contract=contract,
    )


def write_canonical_video_npz(
    destination: str | Path,
    *,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    contract: MediaContract = PRODUCTION_MEDIA,
) -> Path:
    canonical = validate_video_geometry(
        depths,
        intrinsics,
        w2c,
        contract=contract,
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            depths=canonical[0],
            validity=canonical[0] > 0,
            intrinsics=canonical[1],
            w2c=canonical[2],
            camera_convention=np.asarray("world_to_camera"),
            depth_convention=np.asarray("camera_z"),
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


class VideoMegaSAMProvider:
    """Run the MegaSAM orchestrator after checking every entrypoint against its hash."""

    def __init__(
        self,
        config: VideoMegaSAMConfig,
        *,
        runner: ExternalCommandRunner | None = None,
        contract: MediaContract = PRODUCTION_MEDIA,
    ) -> None:
        self.config = config
        self.runner = runner or ExternalCommandRunner(
            default_timeout_seconds=config.timeout_seconds
        )
        self.contract = contract

    def build_command(
        self,
        input_frames: str | Path,
        output_path: str | Path,
    ) -> list[str]:
        return [
            str(self.config.python_executable),
            str(self.config.orchestrator_script),
            "--input_dir",
            str(input_frames),
            "--output_path",
            str(output_path),
            "--depth_model",
            "moge",
            "--resolution",
            str(self.config.resolution),
        ]

    def run(
        self,
        input_frames: str | Path,
        output_path: str | Path,
        *,
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        self.config.verify()
        frames_root = Path(input_frames).expanduser().resolve()
        if not frames_root.is_dir():
            raise NotADirectoryError(frames_root)
        frame_paths = sorted(
            path
            for path in frames_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(frame_paths) != self.contract.num_frames:
            raise ValueError(
                f"video MegaSAM requires {self.contract.num_frames} frames, got {len(frame_paths)}"
            )

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        raw = output.with_name(f".{output.stem}.vendor{output.suffix}")
        raw.unlink(missing_ok=True)
        try:
            command = self.build_command(frames_root, raw)
            result = self.runner.run(
                command,
                cwd=self.config.orchestrator_script.parent,
                env={**os.environ, **dict(self.config.environment)},
                timeout_seconds=self.config.timeout_seconds,
                cancel_event=cancel_event,
            )
            if not raw.is_file():
                raise RuntimeError(f"MegaSAM orchestrator did not produce expected NPZ: {raw}")
            with np.load(raw, allow_pickle=False) as archive:
                native_depth_shape = list(np.asarray(archive["depths"]).shape)
            depths, intrinsics, w2c = canonicalize_video_npz(
                raw,
                contract=self.contract,
            )
            write_canonical_video_npz(
                output,
                depths=depths,
                intrinsics=intrinsics,
                w2c=w2c,
                contract=self.contract,
            )
        finally:
            raw.unlink(missing_ok=True)

        manifest = {
            "schema_version": "4director-video-megasam-v1",
            "provider": "megasam",
            "camera_tracking": True,
            "components": list(self.config.components),
            "orchestrator": {
                "path": str(self.config.orchestrator_script),
                "sha256": self.config.orchestrator_sha256,
            },
            "component_entrypoints": {
                lock.component: {
                    "path": str(lock.path),
                    "sha256": lock.sha256,
                }
                for lock in self.config.component_entrypoints
            },
            "command": list(result.argv),
            "output": str(output),
            "output_sha256": sha256_file(output),
            "native_depth_shape": native_depth_shape,
            "resampled_to_media_contract": native_depth_shape
            != [self.contract.num_frames, self.contract.height, self.contract.width],
            "shape": {
                "depths": list(depths.shape),
                "validity": list((depths > 0).shape),
                "intrinsics": list(intrinsics.shape),
                "w2c": list(w2c.shape),
            },
        }
        manifest_path = output.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

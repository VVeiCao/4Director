"""Static single-image MoGe2/UniDepth fusion without camera tracking."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
from PIL import Image

from fourdirector import paths

from .external import ExternalCommandRunner
from .media import PRODUCTION_MEDIA, MediaContract
from .megasam import sha256_file, validate_intrinsics

# Lets UniDepth import the xformers NystromAttention newer xformers dropped.
UNIDEPTH_COMPAT = paths.ROOT / "third_party" / "geometry" / "megasam" / "compat"

FORBIDDEN_STATIC_COMPONENTS = ("droid", "raft", "cvd")


@dataclass(frozen=True)
class StaticFusion:
    inverse_depth: np.ndarray
    validity: np.ndarray
    scale: float
    shift: float
    fit_pixels: int
    median_absolute_residual: float


def robust_scale_shift_fusion(
    moge2_inverse_depth: np.ndarray,
    unidepth_inverse_depth: np.ndarray,
    *,
    validity: np.ndarray | None = None,
    max_iterations: int = 20,
    huber_delta: float = 1.345,
    minimum_pixels: int = 64,
    maximum_fit_pixels: int = 200_000,
) -> StaticFusion:
    """Robustly calibrate MoGe2 inverse depth to UniDepth's metric gauge.

    The fit is ``unidepth_inverse ~= scale * moge2_inverse + shift``.
    IRLS and all fusion operations are pure NumPy and deterministic.
    """

    moge = np.asarray(moge2_inverse_depth, dtype=np.float64)
    metric = np.asarray(unidepth_inverse_depth, dtype=np.float64)
    if moge.shape != metric.shape or moge.ndim != 2:
        raise ValueError("static inverse-depth inputs must be same-shape 2D arrays")
    valid = (
        np.isfinite(moge)
        & np.isfinite(metric)
        & (moge > 0)
        & (metric > 0)
    )
    if validity is not None:
        mask = np.asarray(validity)
        if mask.shape != moge.shape:
            raise ValueError("validity shape differs from inverse depth")
        valid &= mask.astype(bool)
    count = int(valid.sum())
    if count < minimum_pixels:
        raise ValueError(
            f"static depth fusion needs at least {minimum_pixels} valid pixels, "
            f"got {count}"
        )

    x = moge[valid]
    y = metric[valid]
    if len(x) > maximum_fit_pixels:
        selected = np.linspace(
            0,
            len(x) - 1,
            num=maximum_fit_pixels,
            dtype=np.int64,
        )
        x = x[selected]
        y = y[selected]
    design = np.column_stack((x, np.ones_like(x)))
    parameters, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(max_iterations):
        residual = design @ parameters - y
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        robust_sigma = max(1.4826 * mad, np.finfo(np.float64).eps)
        normalized = np.abs(residual - center) / (
            max(huber_delta, np.finfo(np.float64).eps) * robust_sigma
        )
        weights = np.ones_like(normalized)
        outside = normalized > 1.0
        weights[outside] = 1.0 / normalized[outside]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = y * np.sqrt(weights)
        updated, *_ = np.linalg.lstsq(
            weighted_design,
            weighted_target,
            rcond=None,
        )
        if np.allclose(updated, parameters, rtol=1e-8, atol=1e-10):
            parameters = updated
            break
        parameters = updated

    scale, shift = (float(parameters[0]), float(parameters[1]))
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
        raise ValueError(
            f"static depth fusion produced invalid scale/shift: {scale}, {shift}"
        )
    calibrated = scale * moge + shift
    output_valid = (
        np.isfinite(calibrated)
        & (calibrated > 0)
        & np.isfinite(moge)
        & (moge > 0)
    )
    fused = np.zeros(moge.shape, dtype=np.float32)
    fused[output_valid] = calibrated[output_valid].astype(np.float32)

    metric_only = (
        ~output_valid
        & np.isfinite(metric)
        & (metric > 0)
    )
    fused[metric_only] = metric[metric_only].astype(np.float32)
    output_valid |= metric_only
    fit_residual = scale * x + shift - y
    return StaticFusion(
        inverse_depth=fused,
        validity=output_valid.astype(bool),
        scale=scale,
        shift=shift,
        fit_pixels=len(x),
        median_absolute_residual=float(np.median(np.abs(fit_residual))),
    )


def fuse_static_inverse_depth(
    moge2_inverse_depth: np.ndarray,
    unidepth_metric_depth: np.ndarray,
    **kwargs: Any,
) -> StaticFusion:
    metric_depth = np.asarray(unidepth_metric_depth, dtype=np.float64)
    metric_inverse = np.zeros(metric_depth.shape, dtype=np.float64)
    valid = np.isfinite(metric_depth) & (metric_depth > 0)
    metric_inverse[valid] = 1.0 / metric_depth[valid]
    return robust_scale_shift_fusion(
        moge2_inverse_depth,
        metric_inverse,
        **kwargs,
    )


@dataclass(frozen=True)
class LockedEntrypoint:
    python_executable: Path
    script: Path
    sha256: str

    def __post_init__(self) -> None:
        digest = self.sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("entrypoint sha256 must be a 64-character hex digest")

    @classmethod
    def lock(
        cls,
        *,
        python_executable: str | Path,
        script: str | Path,
    ) -> LockedEntrypoint:
        path = Path(script).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return cls(
            python_executable=Path(python_executable).expanduser(),
            script=path,
            sha256=sha256_file(path),
        )

    def verify(self) -> None:
        if not self.python_executable.is_file():
            raise FileNotFoundError(self.python_executable)
        if not self.script.is_file():
            raise FileNotFoundError(self.script)
        actual = sha256_file(self.script)
        if actual != self.sha256.lower():
            raise RuntimeError(
                f"entrypoint lock mismatch for {self.script}: "
                f"expected {self.sha256}, got {actual}"
            )


@dataclass(frozen=True)
class StaticMegaSAMConfig:
    moge2: LockedEntrypoint
    unidepth_v2: LockedEntrypoint
    moge2_working_directory: Path | None = None
    unidepth_v2_working_directory: Path | None = None
    timeout_seconds: float = 3_600.0
    moge2_output_relative: Path = Path("moge2/input.npy")
    unidepth_depth_relative: Path = Path("unidepth/depth.npy")
    unidepth_intrinsics_relative: Path = Path("unidepth/intrinsics.npy")
    unidepth_npz_relative: Path = Path("unidepth/static/input.npz")

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for value in (
            self.moge2_output_relative,
            self.unidepth_depth_relative,
            self.unidepth_intrinsics_relative,
            self.unidepth_npz_relative,
        ):
            if value.is_absolute() or ".." in value.parts:
                raise ValueError("static output paths must be safe relative paths")
        for value in (
            self.moge2_working_directory,
            self.unidepth_v2_working_directory,
        ):
            if value is not None and not Path(value).expanduser().is_dir():
                raise NotADirectoryError(value)


def write_static_geometry(
    destination: str | Path,
    *,
    fusion: StaticFusion,
    intrinsics: np.ndarray,
    contract: MediaContract = PRODUCTION_MEDIA,
) -> dict[str, Any]:
    inverse_depth = np.asarray(fusion.inverse_depth, dtype=np.float32)
    validity = np.asarray(fusion.validity, dtype=bool)
    expected = (contract.height, contract.width)
    if inverse_depth.shape != expected or validity.shape != expected:
        raise ValueError(
            f"static geometry must have image shape {expected}, got "
            f"{inverse_depth.shape}/{validity.shape}"
        )
    if not np.isfinite(inverse_depth).all() or np.any(inverse_depth < 0):
        raise ValueError("static inverse depth must be finite and non-negative")
    if np.any(validity & (inverse_depth <= 0)):
        raise ValueError("valid static pixels must have positive inverse depth")
    k = np.asarray(intrinsics, dtype=np.float32)
    if k.shape == (3, 3):
        k = k[None]
    k = validate_intrinsics(k, frame_count=1)
    w2c = np.eye(4, dtype=np.float32)[None]
    camera_z = np.zeros_like(inverse_depth, dtype=np.float32)
    camera_z[validity] = 1.0 / inverse_depth[validity]

    output = Path(destination).expanduser().resolve()
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
            inverse_depth=inverse_depth[None],
            camera_z=camera_z[None],
            validity=validity[None],
            intrinsics=k,
            w2c=w2c,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema_version": "4director-static-megasam-v1",
        "provider": "moge2_unidepth_v2_static",
        "camera_tracking": False,
        "components": ["moge2", "unidepth_v2"],
        "forbidden_components": list(FORBIDDEN_STATIC_COMPONENTS),
        "source_frame_count": 1,
        "geometry_frame_count": 1,
        "renderer_broadcast_frame_count": contract.num_frames,
        "renderer_is_only_broadcast_owner": True,
        "camera": {
            "w2c": "identity",
            "trajectory": "static",
        },
        "fusion": {
            "domain": "inverse_depth",
            "method": "robust_scale_shift_irls_huber",
            "scale": fusion.scale,
            "shift": fusion.shift,
            "fit_pixels": fusion.fit_pixels,
            "median_absolute_residual": fusion.median_absolute_residual,
        },
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


class StaticMegaSAMProvider:
    """Run only locked MoGe2 and UniDepth V2 single-image entrypoints."""

    def __init__(
        self,
        config: StaticMegaSAMConfig,
        *,
        runner: ExternalCommandRunner | None = None,
        contract: MediaContract = PRODUCTION_MEDIA,
    ) -> None:
        self.config = config
        self.runner = runner or ExternalCommandRunner(
            default_timeout_seconds=config.timeout_seconds
        )
        self.contract = contract

    def build_commands(
        self,
        *,
        frames_dir: Path,
        work_dir: Path,
    ) -> tuple[list[str], list[str]]:
        moge_output = work_dir / "moge2"
        unidepth_output = work_dir / "unidepth"
        return (
            [
                str(self.config.moge2.python_executable),
                str(self.config.moge2.script),
                "--img-path",
                str(frames_dir),
                "--outdir",
                str(moge_output),
            ],
            [
                str(self.config.unidepth_v2.python_executable),
                str(self.config.unidepth_v2.script),
                "--scene-name",
                "static",
                "--img-path",
                str(frames_dir),
                "--outdir",
                str(unidepth_output),
            ],
        )

    def run(
        self,
        image: str | Path,
        output_path: str | Path,
        *,
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        self.config.moge2.verify()
        self.config.unidepth_v2.verify()
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=".4director-static-depth-",
        ) as temporary_name:
            work = Path(temporary_name)
            frames = work / "frames"
            frames.mkdir()
            shutil.copyfile(image_path, frames / "input.png")
            commands = self.build_commands(frames_dir=frames, work_dir=work)
            for command, entrypoint, configured_cwd in zip(
                commands,
                (self.config.moge2, self.config.unidepth_v2),
                (
                    self.config.moge2_working_directory,
                    self.config.unidepth_v2_working_directory,
                ),
                strict=True,
            ):
                working_directory = (
                    Path(configured_cwd).expanduser().resolve()
                    if configured_cwd is not None
                    else entrypoint.script.parent
                )
                environment = dict(os.environ)
                search_path = [str(working_directory)]
                if entrypoint is self.config.unidepth_v2:
                    search_path.insert(0, str(UNIDEPTH_COMPAT))
                if environment.get("PYTHONPATH"):
                    search_path.append(environment["PYTHONPATH"])
                environment["PYTHONPATH"] = os.pathsep.join(search_path)
                self.runner.run(
                    command,
                    cwd=working_directory,
                    env=environment,
                    timeout_seconds=self.config.timeout_seconds,
                    cancel_event=cancel_event,
                )
            moge_path = work / self.config.moge2_output_relative
            depth_path = work / self.config.unidepth_depth_relative
            k_path = work / self.config.unidepth_intrinsics_relative
            npz_path = work / self.config.unidepth_npz_relative
            if not moge_path.is_file():
                raise RuntimeError(
                    f"static depth entrypoints did not produce {moge_path}"
                )
            moge = np.load(moge_path, allow_pickle=False)
            if depth_path.is_file() and k_path.is_file():
                metric_depth = np.load(depth_path, allow_pickle=False)
                intrinsics = np.load(k_path, allow_pickle=False)
            elif npz_path.is_file():
                with np.load(npz_path, allow_pickle=False) as archive:
                    metric_depth = np.asarray(archive["depth"], dtype=np.float32)
                    horizontal_fov = float(np.asarray(archive["fov"]).reshape(()))
                expected = (self.contract.height, self.contract.width)
                if metric_depth.shape != expected:
                    metric_depth = np.asarray(
                        Image.fromarray(metric_depth).resize(
                            (self.contract.width, self.contract.height),
                            Image.Resampling.BILINEAR,
                        ),
                        dtype=np.float32,
                    )
                focal = self.contract.width / (
                    2.0 * np.tan(np.deg2rad(horizontal_fov) * 0.5)
                )
                intrinsics = np.asarray(
                    [
                        [focal, 0.0, self.contract.width * 0.5],
                        [0.0, focal, self.contract.height * 0.5],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            else:
                raise RuntimeError(
                    "static UniDepth entrypoint produced neither "
                    f"{depth_path}/{k_path} nor {npz_path}"
                )
            fusion = fuse_static_inverse_depth(moge, metric_depth)
            manifest = write_static_geometry(
                output,
                fusion=fusion,
                intrinsics=intrinsics,
                contract=self.contract,
            )
            manifest["commands"] = [list(command) for command in commands]
            output.with_suffix(".manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return manifest

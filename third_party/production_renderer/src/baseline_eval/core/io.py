from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_NATURAL_PARTS = re.compile(r"(\d+)")


def natural_key(path: Path | str) -> tuple[object, ...]:
    """Sort frame names numerically while retaining deterministic text ordering."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NATURAL_PARTS.split(Path(path).name)
    )


def image_files(directory: Path, glob_pattern: str = "*") -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        (
            path
            for path in directory.glob(glob_pattern)
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=natural_key,
    )


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, hash_contents: bool = False) -> dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        identity["sha256"] = sha256_file(path)
    return identity


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def assert_unique_paths(paths: Iterable[Path], description: str) -> None:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{description} contains duplicate files")


def npz_array_metadata(
    path: Path,
    key: str,
) -> tuple[tuple[int, ...], np.dtype[Any]]:
    """Read an NPZ member's shape and dtype without materializing the array."""
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise KeyError(f"{path} does not contain {key!r}")
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in {(2, 0), (3, 0)}:
                shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unsupported NPY version {version} in {path}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def npz_array_frames(
    path: Path,
    key: str,
    indices: Iterable[int],
) -> np.ndarray:
    """Read selected leading-axis frames from a C-order NPZ array."""
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise KeyError(f"{path} does not contain {key!r}")
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = (
                    np.lib.format.read_array_header_1_0(handle)
                )
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = (
                    np.lib.format.read_array_header_2_0(handle)
                )
            else:
                raise ValueError(f"Unsupported NPY version {version} in {path}")
            if fortran_order or len(shape) < 1:
                raise ValueError(
                    "Frame slicing requires a leading-axis C-order array"
                )
            selected = [int(index) for index in indices]
            if any(index < 0 or index >= shape[0] for index in selected):
                raise IndexError(f"Frame index outside {shape[0]} frames")
            data_start = handle.tell()
            frame_shape = tuple(int(value) for value in shape[1:])
            frame_bytes = int(np.prod(frame_shape)) * np.dtype(dtype).itemsize
            frames = []
            for index in selected:
                handle.seek(data_start + index * frame_bytes)
                payload = handle.read(frame_bytes)
                if len(payload) != frame_bytes:
                    raise EOFError(f"Truncated {key!r} frame {index} in {path}")
                frames.append(
                    np.frombuffer(payload, dtype=dtype).reshape(frame_shape)
                )
    return np.stack(frames)

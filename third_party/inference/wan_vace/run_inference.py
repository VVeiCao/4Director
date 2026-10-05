#!/usr/bin/env python3
"""Generate 4Director videos with Wan2.1-VACE-14B and a Full-VACE Motion Adapter.

Each manifest row names a reference image, a depth-control video, a prompt,
and a seed; each ``--model KEY=checkpoint`` writes ``<output-root>/KEY/<case_id>/``.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _load_cudnn() -> None:
    """Make Torch use cuDNN 9.13.1 rather than the 9.7.1 it ships.

    The released videos were generated with cuDNN 9.13.1, whose kernels give
    other bytes than 9.7.1's; environments/install_extensions.sh adds it to the
    environment. Loading its libcudnn.so.9 here, before Torch, makes it the one
    Torch uses. That library loads each sublibrary by full-version name first,
    which only a 9.13.1 copy has: the system's if LD_LIBRARY_PATH lists one,
    else this one. Either way every sublibrary comes from the same copy.
    """
    lib = Path(sys.prefix) / "opt" / "cudnn" / "nvidia" / "cudnn" / "lib"
    if not (lib / "libcudnn_graph.so.9.13.1").exists():
        raise SystemExit(f"cuDNN 9.13.1 is not at {lib}; run environments/install_extensions.sh")
    ctypes.CDLL(str(lib / "libcudnn.so.9"), mode=ctypes.RTLD_GLOBAL)


_load_cudnn()

import torch  # noqa: E402
from PIL import Image  # noqa: E402


# Import the pinned DiffSynth-Studio submodule ahead of any installed copy, so
# generation runs exactly the pipeline code the released results came from.
DIFFSYNTH_ROOT = Path(__file__).resolve().parents[2] / "DiffSynth-Studio"
if not (DIFFSYNTH_ROOT / "diffsynth" / "__init__.py").is_file():
    raise SystemExit(
        f"DiffSynth-Studio is missing at {DIFFSYNTH_ROOT}; fetch the submodules with "
        "`git submodule update --init --recursive`"
    )
sys.path.insert(0, str(DIFFSYNTH_ROOT))

from diffsynth.core import load_state_dict  # noqa: E402
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from diffsynth.utils.data import VideoData, save_video  # noqa: E402


DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, warped geometry, flickering, jitter, "
    "artifacts, text, watermark, overexposed, underexposed, duplicated subject, "
    "deformed subject, inconsistent appearance"
)


def local_wan_paths() -> tuple[Path, Path]:
    """Locate the local Wan2.1-VACE-14B store; nothing is downloaded here.

    WAN_ROOT defaults to the checkout's models/ directory, where
    download_models.py puts Wan-AI/Wan2.1-VACE-14B. Its bundled
    google/umt5-xxl tokenizer is used, or the byte-identical one under
    Wan-AI/Wan2.1-T2V-1.3B in stores laid out that way.
    """
    raw_root = os.environ.get("WAN_ROOT")
    root = (
        Path(raw_root).expanduser().resolve()
        if raw_root
        else Path(__file__).resolve().parents[3] / "models"
    )
    model_root = root / "Wan-AI" / "Wan2.1-VACE-14B"
    tokenizer_root = model_root / "google" / "umt5-xxl"
    if not tokenizer_root.is_dir():
        tokenizer_root = root / "Wan-AI" / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl"
    tokenizer_files = (
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [
        path
        for path in (
            model_root / "models_t5_umt5-xxl-enc-bf16.pth",
            model_root / "Wan2.1_VAE.pth",
            *(tokenizer_root / name for name in tokenizer_files),
        )
        if not path.is_file()
    ]
    # Every shard the index lists, so an interrupted download is caught here
    # rather than as a missing tensor after the model starts loading.
    index = model_root / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        missing.extend(model_root / shard for shard in shards if not (model_root / shard).is_file())
    else:
        missing.append(index)
    if missing:
        raise RuntimeError(
            f"the Wan store at {root} is incomplete (run download_models.py or set "
            "WAN_ROOT); missing: " + ", ".join(str(path) for path in missing)
        )
    return model_root, tokenizer_root


def model_spec(value: str) -> tuple[str, Path]:
    key, separator, path = value.partition("=")
    if not separator or not key or not path:
        raise argparse.ArgumentTypeError("Model must use KEY=/path/to/checkpoint")
    if not key.replace("-", "").replace("_", "").isalnum():
        raise argparse.ArgumentTypeError(f"Unsafe model key: {key}")
    return key, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 81-frame Full-VACE inference for every manifest row."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=model_spec,
        action="append",
        default=[],
        help="Repeatable KEY=/path/to/checkpoint specification.",
    )
    parser.add_argument("--negative-prompt", type=Path, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--save-png-frames",
        action="store_true",
        help=(
            "Save the pipeline's RGB PIL frames directly as lossless PNGs "
            "before video encoding."
        ),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def video_probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames,nb_frames,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}: {streams}")
    stream = streams[0]
    frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frames in {None, "N/A"}:
        raise RuntimeError(f"Could not count frames in {path}")
    return {
        "frames": int(frames),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream.get("avg_frame_rate", ""),
    }


def local_model_configs() -> list[ModelConfig]:
    model_root, _ = local_wan_paths()
    shard_paths = sorted(
        str(path)
        for path in model_root.glob("diffusion_pytorch_model-*.safetensors")
    )
    if not shard_paths:
        raise RuntimeError(f"No VACE model shards found under {model_root}")
    return [
        ModelConfig(path=shard_paths),
        ModelConfig(
            path=str(model_root / "models_t5_umt5-xxl-enc-bf16.pth")
        ),
        ModelConfig(path=str(model_root / "Wan2.1_VAE.pth")),
    ]


def load_pipeline() -> WanVideoPipeline:
    _, tokenizer_root = local_wan_paths()
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=local_model_configs(),
        tokenizer_config=ModelConfig(path=str(tokenizer_root)),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 8,
    )


def load_vace_checkpoint(
    pipe: WanVideoPipeline,
    checkpoint: Path,
) -> int:
    state_dict = load_state_dict(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    incompatible = pipe.vace.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    tensor_count = len(state_dict)
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    if missing or unexpected:
        raise RuntimeError(
            "VACE checkpoint key mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    return tensor_count


def load_checkpoint(
    pipe: WanVideoPipeline,
    checkpoint: Path,
    checkpoint_kind: str,
) -> int:
    if checkpoint_kind == "full_vace":
        return load_vace_checkpoint(pipe, checkpoint)
    raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind}")


def inferred_completed_epochs(model_key: str) -> int | None:
    normalized = model_key.lower().replace("-", "_")
    if "epoch1" in normalized or "step526" in normalized:
        return 1
    if "epoch2" in normalized or "step1052" in normalized:
        return 2
    if "epoch3" in normalized or "step1578" in normalized:
        return 3
    return None


def checkpoint_hashes(
    models: list[tuple[str, Path]],
    output_root: Path,
    rank: int,
    timeout_seconds: int = 900,
) -> dict[str, str]:
    cache_path = output_root / "checkpoint_hashes.json"
    expected = {
        key: {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for key, path in models
    }

    def valid_cache() -> dict[str, Any] | None:
        if not cache_path.is_file():
            return None
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        for key, info in expected.items():
            cached = cache.get(key, {})
            if any(cached.get(field) != value for field, value in info.items()):
                return None
            if not cached.get("sha256"):
                return None
        return cache

    cache = valid_cache()
    if cache is None and rank == 0:
        cache = {
            key: {
                **expected[key],
                "sha256": sha256_file(path),
            }
            for key, path in models
        }
        write_json_atomic(cache_path, cache)
    elif cache is None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(2)
            cache = valid_cache()
            if cache is not None:
                break
        if cache is None:
            raise TimeoutError(f"Timed out waiting for {cache_path}")
    assert cache is not None
    return {key: str(cache[key]["sha256"]) for key, _ in models}


def padded_control_frames(
    path: Path,
    *,
    height: int,
    width: int,
    input_frames: int,
    model_frames: int,
) -> list[Image.Image]:
    control = VideoData(str(path), height=height, width=width).raw_data()
    if len(control) != input_frames:
        raise RuntimeError(
            f"{path} decoded to {len(control)} frames, expected {input_frames}"
        )
    if not control:
        raise RuntimeError(f"No control frames decoded from {path}")
    if len(control) > model_frames:
        raise RuntimeError(
            f"Control has {len(control)} frames but model accepts {model_frames}"
        )
    return control + [
        control[-1].copy() for _ in range(model_frames - len(control))
    ]


def output_dir_for(output_root: Path, model_key: str, case_id: str) -> Path:
    return output_root / model_key / case_id


def completed_png_frames(
    frames_dir: Path,
    *,
    expected_frames: int,
    width: int,
    height: int,
) -> bool:
    manifest_path = frames_dir / "frames_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = sorted(frames_dir.glob("*.png"))
        if len(paths) != expected_frames:
            return False
        if (
            manifest.get("frames") != expected_frames
            or manifest.get("width") != width
            or manifest.get("height") != height
        ):
            return False
        with Image.open(paths[0]) as first, Image.open(paths[-1]) as last:
            return (
                first.mode == "RGB"
                and last.mode == "RGB"
                and first.size == (width, height)
                and last.size == (width, height)
            )
    except (OSError, ValueError, TypeError):
        return False


def save_png_frames_atomic(
    frames: list[Image.Image],
    frames_dir: Path,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    temporary = frames_dir.parent / (
        f".{frames_dir.name}.working-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    backup = frames_dir.parent / (
        f".{frames_dir.name}.backup-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    temporary.mkdir(parents=True, exist_ok=False)
    frame_records: list[dict[str, Any]] = []
    try:
        for index, frame in enumerate(frames):
            rgb = frame.convert("RGB")
            if rgb.size != (width, height):
                raise RuntimeError(
                    f"Generated frame {index} has size {rgb.size}, "
                    f"expected {(width, height)}"
                )
            path = temporary / f"{index:06d}.png"
            rgb.save(path, format="PNG", compress_level=6)
            frame_records.append(
                {
                    "index": index,
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema_version": "diffsynth-direct-rgb-frames-v1",
            "source": "WanVideoPipeline return value before save_video encoding",
            "format": "RGB PNG (lossless)",
            "frames": len(frames),
            "width": width,
            "height": height,
            "files": frame_records,
        }
        write_json_atomic(temporary / "frames_manifest.json", manifest)

        if frames_dir.exists():
            frames_dir.replace(backup)
        try:
            temporary.replace(frames_dir)
        except BaseException:
            if backup.exists() and not frames_dir.exists():
                backup.replace(frames_dir)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def completed_output(
    output: Path,
    metadata_path: Path,
    *,
    model_key: str,
    checkpoint_sha256: str,
    expected_frames: int,
    width: int,
    height: int,
    require_png_frames: bool,
    expected_metadata: dict[str, Any] | None = None,
) -> bool:
    if not output.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        probe = video_probe(output)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return (
        metadata.get("model_key") == model_key
        and metadata.get("checkpoint_sha256") == checkpoint_sha256
        and probe["frames"] == expected_frames
        and probe["width"] == width
        and probe["height"] == height
        and all(
            metadata.get(key) == value
            for key, value in (expected_metadata or {}).items()
        )
        and (
            not require_png_frames
            or completed_png_frames(
                output.parent / "generated_frames_png",
                expected_frames=expected_frames,
                width=width,
                height=height,
            )
        )
    )


def run_one(
    pipe: WanVideoPipeline,
    row: dict[str, str],
    *,
    manifest_path: Path,
    output_root: Path,
    model_key: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_tensor_count: int,
    negative_prompt: str,
    args: argparse.Namespace,
    file_hash_cache: dict[Path, str],
    checkpoint_kind: str = "full_vace",
    completed_training_epochs: int | None = None,
) -> dict[str, Any]:
    case_id = row["case_id"]
    output_dir = output_dir_for(output_root, model_key, case_id)
    output = output_dir / "generated.mp4"
    metadata_path = output_dir / "run_metadata.json"
    output_frames = int(row["output_frames"])
    control_path = Path(row["vace_video"]).expanduser().resolve()
    mask_value = row.get("vace_video_mask", "").strip()
    mask_path = (
        Path(mask_value).expanduser().resolve() if mask_value else None
    )
    reference_path = Path(row["vace_reference_image"]).expanduser().resolve()
    source_path = Path(row["video"]).expanduser().resolve()

    def cached_sha(path: Path) -> str:
        if path not in file_hash_cache:
            file_hash_cache[path] = sha256_file(path)
        return file_hash_cache[path]

    expected_metadata = {
        "source_sha256": cached_sha(source_path),
        "control_sha256": cached_sha(control_path),
        "control_mask_sha256": (
            cached_sha(mask_path) if mask_path is not None else None
        ),
        "reference_sha256": cached_sha(reference_path),
        "prompt": row["prompt"],
        "negative_prompt": negative_prompt,
        "seed": int(row["seed"]),
    }
    if (
        not args.force
        and completed_output(
            output,
            metadata_path,
            model_key=model_key,
            checkpoint_sha256=checkpoint_sha256,
            expected_frames=output_frames,
            width=args.width,
            height=args.height,
            require_png_frames=args.save_png_frames,
            expected_metadata=expected_metadata,
        )
    ):
        return {
            "case_id": case_id,
            "display_sequence": row["display_sequence"],
            "status": "skipped",
            "output": str(output),
        }

    input_frames = int(row["control_frames"])
    model_frames = int(row["model_frames"])
    if model_frames != args.num_frames:
        raise RuntimeError(
            f"{case_id}: manifest model_frames={model_frames}, "
            f"CLI num_frames={args.num_frames}"
        )

    control_frames = padded_control_frames(
        control_path,
        height=args.height,
        width=args.width,
        input_frames=input_frames,
        model_frames=model_frames,
    )
    mask_frames = (
        padded_control_frames(
            mask_path,
            height=args.height,
            width=args.width,
            input_frames=input_frames,
            model_frames=model_frames,
        )
        if mask_path is not None
        else None
    )
    reference_image = (
        Image.open(reference_path)
        .convert("RGB")
        .resize((args.width, args.height))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        generated = pipe(
            prompt=row["prompt"],
            negative_prompt=negative_prompt,
            vace_video=control_frames,
            vace_video_mask=mask_frames,
            vace_reference_image=reference_image,
            height=args.height,
            width=args.width,
            num_frames=model_frames,
            seed=int(row["seed"]),
            tiled=True,
            num_inference_steps=args.num_inference_steps,
        )
    if len(generated) != model_frames:
        raise RuntimeError(
            f"{case_id}: model returned {len(generated)} frames, "
            f"expected {model_frames}"
        )
    trimmed = generated[:output_frames]
    if len(trimmed) != output_frames:
        raise RuntimeError(
            f"{case_id}: trim returned {len(trimmed)} frames, "
            f"expected {output_frames}"
        )

    # Long inference can outlive transient cleanup of node-local scratch
    # directories. Recreate the case directory immediately before encoding.
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_frames_manifest: dict[str, Any] | None = None
    direct_frames_dir = output_dir / "generated_frames_png"
    if args.save_png_frames:
        direct_frames_manifest = save_png_frames_atomic(
            trimmed,
            direct_frames_dir,
            width=args.width,
            height=args.height,
        )
    temporary_output = output_dir / "generated.tmp.mp4"
    save_video(trimmed, str(temporary_output), fps=args.fps, quality=5)
    probe = video_probe(temporary_output)
    expected_probe = {
        "frames": output_frames,
        "width": args.width,
        "height": args.height,
    }
    for key, expected_value in expected_probe.items():
        if probe[key] != expected_value:
            temporary_output.unlink(missing_ok=True)
            raise RuntimeError(
                f"{case_id}: invalid generated video {probe}, "
                f"expected {expected_probe}"
            )
    temporary_output.replace(output)

    metadata = {
        "schema_version": 1,
        "case_id": case_id,
        "display_sequence": row["display_sequence"],
        "dataset": row["dataset"],
        "model_key": model_key,
        "model_id": "Wan-AI/Wan2.1-VACE-14B",
        "checkpoint": str(checkpoint),
        "checkpoint_kind": checkpoint_kind,
        "completed_training_epochs": completed_training_epochs,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_tensor_count": checkpoint_tensor_count,
        "manifest": str(manifest_path),
        "source_video": str(source_path),
        "source_sha256": cached_sha(source_path),
        "control_video": str(control_path),
        "control_sha256": cached_sha(control_path),
        "control_mask": str(mask_path) if mask_path is not None else None,
        "control_mask_sha256": (
            cached_sha(mask_path) if mask_path is not None else None
        ),
        "reference_image": str(reference_path),
        "reference_sha256": cached_sha(reference_path),
        "prompt": row["prompt"],
        "negative_prompt": negative_prompt,
        "seed": int(row["seed"]),
        "height": args.height,
        "width": args.width,
        "input_control_frames": input_frames,
        "model_frames": model_frames,
        "saved_output_frames": output_frames,
        "padding_mode": "repeat_last_control_frame",
        "repeated_control_frames": model_frames - input_frames,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "output": str(output),
        "output_probe": probe,
        "direct_png_frames": (
            str(direct_frames_dir) if direct_frames_manifest is not None else None
        ),
        "direct_png_frames_manifest": (
            str(direct_frames_dir / "frames_manifest.json")
            if direct_frames_manifest is not None
            else None
        ),
    }
    write_json_atomic(metadata_path, metadata)
    return {
        "case_id": case_id,
        "display_sequence": row["display_sequence"],
        "status": "generated",
        "output": str(output),
    }


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("Wan frame count must satisfy 4k+1")

    manifest_path = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    models = [
        (key, checkpoint.expanduser().resolve(), "full_vace")
        for key, checkpoint in args.model
    ]
    if not models:
        raise ValueError("At least one --model is required")
    if len({key for key, _checkpoint, _checkpoint_kind in models}) != len(models):
        raise ValueError(f"Duplicate model keys: {models}")
    for _, checkpoint, _checkpoint_kind in models:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    rank = (
        args.rank
        if args.rank is not None
        else int(os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0")))
    )
    world_size = (
        args.world_size
        if args.world_size is not None
        else int(os.environ.get("SLURM_NTASKS", "1"))
    )
    rows = read_rows(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    assigned = [
        row for index, row in enumerate(rows) if index % world_size == rank
    ]

    negative_prompt = DEFAULT_NEGATIVE_PROMPT
    if args.negative_prompt is not None:
        negative_prompt = (
            args.negative_prompt.expanduser()
            .resolve()
            .read_text(encoding="utf-8")
            .strip()
        )

    hashes = checkpoint_hashes(
        [(key, checkpoint) for key, checkpoint, _checkpoint_kind in models],
        output_root,
        rank,
    )
    print(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "manifest_rows": len(rows),
                "assigned_rows": len(assigned),
                "models": [
                    {"key": key, "checkpoint_kind": checkpoint_kind}
                    for key, _checkpoint, checkpoint_kind in models
                ],
                "num_frames": args.num_frames,
            },
            indent=2,
        ),
        flush=True,
    )
    pipe: WanVideoPipeline | None = None
    loaded_kind: str | None = None
    file_hash_cache: dict[Path, str] = {}
    any_failed = False
    for model_key, checkpoint, checkpoint_kind in models:
        if pipe is None or loaded_kind != checkpoint_kind:
            if pipe is not None:
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
            pipe = load_pipeline()
            loaded_kind = checkpoint_kind
        checkpoint_tensor_count = load_checkpoint(
            pipe,
            checkpoint,
            checkpoint_kind,
        )
        summary: dict[str, Any] = {
            "rank": rank,
            "world_size": world_size,
            "model_key": model_key,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashes[model_key],
            "checkpoint_tensor_count": checkpoint_tensor_count,
            "done": [],
            "skipped": [],
            "failed": [],
        }
        for index, row in enumerate(assigned, start=1):
            print(
                f"[{model_key} rank {rank}] "
                f"{index}/{len(assigned)} {row['display_sequence']}",
                flush=True,
            )
            try:
                result = run_one(
                    pipe,
                    row,
                    manifest_path=manifest_path,
                    output_root=output_root,
                    model_key=model_key,
                    checkpoint=checkpoint,
                    checkpoint_sha256=hashes[model_key],
                    checkpoint_tensor_count=checkpoint_tensor_count,
                    checkpoint_kind=checkpoint_kind,
                    completed_training_epochs=inferred_completed_epochs(
                        model_key
                    ),
                    negative_prompt=negative_prompt,
                    args=args,
                    file_hash_cache=file_hash_cache,
                )
                status_bucket = (
                    "done" if result["status"] == "generated" else result["status"]
                )
                summary[status_bucket].append(row["case_id"])
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                failure = {
                    "case_id": row["case_id"],
                    "display_sequence": row["display_sequence"],
                    "error": repr(exc),
                }
                summary["failed"].append(failure)
                print(
                    json.dumps(
                        {"status": "failed", **failure},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        write_json_atomic(
            output_root
            / model_key
            / "summaries"
            / f"rank_{rank:02d}.json",
            summary,
        )
        if summary["failed"]:
            any_failed = True
    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

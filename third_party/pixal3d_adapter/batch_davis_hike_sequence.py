import argparse
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ["ATTN_BACKEND"])
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "4director-pixal-autotune.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")

import numpy as np
import torch
from PIL import Image

import o_voxel
from pixal3d.renderers import MeshRenderer
from pixal3d.utils.render_utils import proj_camera_to_render_params
from inference import (
    IMAGE_COND_CONFIGS,
    MODEL_PATH,
    build_image_cond_model,
    distance_from_fov,
    init_pipeline,
)


ROT_PIXAL3D_TO_GLB = np.array(
    [
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

REQUIRED_CORRESPONDENCE_KEYS = {
    "crop_xy",
    "coord_glb",
    "depth",
    "normal",
    "base_color",
    "source_rgb",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Pixal3D GLBs for a sequence of frames.")
    parser.add_argument("--vggt-output-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dynamic-file", default="point_clouds/dynamic_object_points.npz")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="Exclusive end frame. Default: all frames.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-vram", action="store_true", default=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--decimation-target", type=int, default=50000)
    parser.add_argument("--texture-size", type=int, default=512)
    parser.add_argument("--remesh", action="store_true", help="Enable remeshing during GLB export. Slower.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate frames even if current outputs already exist.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--image-resolution", type=int, default=512, help="Pixal3D distance helper image resolution.")
    parser.add_argument("--crop-padding", type=float, default=1.1, help="Must match Pixal3D preprocess_image bbox padding.")
    parser.add_argument("--export-correspondence-maps", action="store_true")
    parser.add_argument(
        "--correspondence-resolution",
        type=int,
        default=512,
        help="Square Pixal3D crop resolution used for visible surface correspondence maps.",
    )
    return parser.parse_args()


def load_dynamic(output_dir: Path, dynamic_file: str) -> dict[str, np.ndarray]:
    path = Path(dynamic_file) if os.path.isabs(dynamic_file) else output_dir / dynamic_file
    if not path.is_file():
        raise FileNotFoundError(f"Dynamic point data not found: {path}")
    with np.load(path) as data:
        return {key: np.array(data[key]) for key in data.files}


def alpha_bbox(mask: np.ndarray, padding: float) -> tuple[float, float, float, float] | None:
    ys_xs = np.argwhere(mask > 0)
    if len(ys_xs) == 0:
        return None
    y0, x0 = ys_xs.min(axis=0)
    y1, x1 = ys_xs.max(axis=0)
    center_x = (float(x0) + float(x1)) * 0.5
    center_y = (float(y0) + float(y1)) * 0.5
    size = int(max(float(x1 - x0), float(y1 - y0)) * padding)
    half = size // 2
    return center_x - half, center_y - half, center_x + half, center_y + half


def crop_fov_x(intrinsic: np.ndarray, bbox: tuple[float, float, float, float] | None, width: int) -> float:
    fx = float(intrinsic[0, 0])
    if fx <= 0:
        raise ValueError(f"Invalid fx in intrinsic: {fx}")
    crop_width = float(width if bbox is None else max(bbox[2] - bbox[0], 1.0))
    return 2.0 * math.atan(crop_width / (2.0 * fx))


def make_rgba(image: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    image = image.astype(np.uint8)
    alpha = (mask > 0).astype(np.uint8) * 255
    rgba = np.concatenate([image, alpha[..., None]], axis=-1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path)


def append_jsonl(path: Path, item: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# What Pixal3D fetches at run time, at the revisions this release was validated
# with: its checkpoints, the DINOv3 encoder of its image conditioning, and the
# NAF feature upsampler it loads through torch.hub.
PIXAL3D_REPOSITORY = "TencentARC/Pixal3D"
PIXAL3D_REVISION = "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f"
DINOV3_REPOSITORY = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
NAF_REPOSITORY = "valeoai/NAF"
NAF_COMMIT = "37f2dfc180f2de53d98bd601109c0da0dd6b0f43"


def pinned_pixal3d_checkpoints() -> str:
    """A local snapshot of the pinned checkpoints that pipeline.json loads, and no others."""
    from huggingface_hub import hf_hub_download, snapshot_download

    config = hf_hub_download(PIXAL3D_REPOSITORY, "pipeline.json", revision=PIXAL3D_REVISION)
    with open(config, encoding="utf-8") as handle:
        names = json.load(handle)["args"]["models"].values()
    return snapshot_download(
        PIXAL3D_REPOSITORY,
        revision=PIXAL3D_REVISION,
        allow_patterns=["pipeline.json", *(f"{name}.{suffix}" for name in names for suffix in ("json", "safetensors"))],
    )


def pin_runtime_weights() -> None:
    """Point the DINOv3 configs at a pinned snapshot, and torch.hub's NAF at a pinned commit."""
    from huggingface_hub import snapshot_download

    dinov3 = snapshot_download(
        DINOV3_REPOSITORY, revision=DINOV3_REVISION, allow_patterns=["config.json", "model.safetensors"]
    )
    for config in IMAGE_COND_CONFIGS.values():
        if config.get("model_name") == DINOV3_REPOSITORY:
            config["model_name"] = dinov3

    load = torch.hub.load

    def load_pinned(repo_or_dir, *args, **kwargs):
        if repo_or_dir == NAF_REPOSITORY:
            repo_or_dir = f"{NAF_REPOSITORY}:{NAF_COMMIT}"
            # torch.hub validates a ref against branch and tag names only.
            kwargs["skip_validation"] = True
        return load(repo_or_dir, *args, **kwargs)

    torch.hub.load = load_pinned


def build_pipeline(model_path: str, low_vram: bool):
    pin_runtime_weights()
    if model_path == PIXAL3D_REPOSITORY:
        model_path = pinned_pixal3d_checkpoints()
    pipeline = init_pipeline(model_path=model_path, low_vram=low_vram)
    # Keep the config values in metadata even though init_pipeline already used them.
    return pipeline


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu()
    while tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    array = tensor.numpy()
    if array.ndim == 3 and array.shape[0] in (1, 2, 3, 4) and array.shape[-1] not in (1, 2, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def correspondence_map_is_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            return REQUIRED_CORRESPONDENCE_KEYS.issubset(set(data.files))
    except Exception:
        return False


def resize_source_image_for_correspondence(source_image: Image.Image, render_resolution: int) -> tuple[np.ndarray, np.ndarray]:
    source_rgba = source_image.convert("RGBA").resize(
        (int(render_resolution), int(render_resolution)),
        Image.Resampling.BILINEAR,
    )
    source_rgba_np = np.asarray(source_rgba, dtype=np.uint8)
    return source_rgba_np[..., :3], source_rgba_np[..., 3]


def export_visible_surface_correspondence(
    mesh,
    camera_params: dict,
    output_path: Path,
    render_resolution: int,
    crop_bbox: tuple[float, float, float, float] | None,
    pixal3d_to_glb: np.ndarray,
    source_image: Image.Image,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distance = float(camera_params["distance"])
    near = max(0.01, distance - 2.0)
    far = distance + 10.0
    renderer = MeshRenderer(
        rendering_options={
            "resolution": int(render_resolution),
            "near": near,
            "far": far,
            "ssaa": 1,
            "antialias": False,
            "chunk_size": None,
        },
        device="cuda",
    )
    extrinsic, intrinsic = proj_camera_to_render_params(float(camera_params["camera_angle_x"]), distance)
    with torch.no_grad():
        rendered = renderer.render(
            mesh.cuda(),
            extrinsic,
            intrinsic,
            return_types=["mask", "depth", "coord", "normal", "attr"],
        )

    mask = tensor_to_numpy_image(rendered["mask"]) > 0.5
    depth = tensor_to_numpy_image(rendered["depth"]).astype(np.float32)
    coord = tensor_to_numpy_image(rendered["coord"]).astype(np.float32)
    normal = tensor_to_numpy_image(rendered["normal"]).astype(np.float32)
    base_color = tensor_to_numpy_image(rendered["base_color"]).astype(np.float32)
    alpha = tensor_to_numpy_image(rendered["alpha"]).astype(np.float32)
    metallic = tensor_to_numpy_image(rendered["metallic"]).astype(np.float32)
    roughness = tensor_to_numpy_image(rendered["roughness"]).astype(np.float32)
    if coord.ndim != 3 or coord.shape[-1] != 3:
        raise RuntimeError(f"Unexpected coord render shape: {coord.shape}")
    if base_color.ndim != 3 or base_color.shape[-1] != 3:
        raise RuntimeError(f"Unexpected base_color render shape: {base_color.shape}")
    if normal.ndim == 3 and normal.shape[-1] == 3:
        normal = normal * 2.0 - 1.0
    source_rgb, source_alpha = resize_source_image_for_correspondence(source_image, render_resolution)

    ys, xs = np.nonzero(mask)
    coord_canonical = coord[ys, xs].astype(np.float32)
    coord_h = np.concatenate([coord_canonical, np.ones((len(coord_canonical), 1), dtype=np.float32)], axis=1)
    coord_glb = (pixal3d_to_glb.astype(np.float32) @ coord_h.T).T[:, :3].astype(np.float32)

    np.savez_compressed(
        output_path,
        render_resolution=np.array(render_resolution, dtype=np.int32),
        crop_xy=np.stack([xs, ys], axis=1).astype(np.int32),
        coord_canonical=coord_canonical,
        coord_glb=coord_glb,
        depth=depth[ys, xs].astype(np.float32),
        normal=normal[ys, xs].astype(np.float16),
        base_color=(np.clip(base_color[ys, xs], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        source_rgb=source_rgb[ys, xs].astype(np.uint8),
        source_alpha=source_alpha[ys, xs].astype(np.uint8),
        alpha=(np.clip(alpha[ys, xs], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        metallic=metallic[ys, xs].astype(np.float16),
        roughness=roughness[ys, xs].astype(np.float16),
        color_space=np.array("srgb"),
        crop_bbox_xyxy_processed=np.array([] if crop_bbox is None else crop_bbox, dtype=np.float32),
        camera_angle_x=np.array(float(camera_params["camera_angle_x"]), dtype=np.float32),
        distance=np.array(distance, dtype=np.float32),
        mesh_scale=np.array(float(camera_params.get("mesh_scale", 1.0)), dtype=np.float32),
        pixal3d_to_glb=pixal3d_to_glb.astype(np.float32),
    )
    return {
        "path": str(output_path),
        "render_resolution": int(render_resolution),
        "visible_pixels": int(len(xs)),
        "near": float(near),
        "far": float(far),
        "color_fields": ["base_color", "source_rgb", "source_alpha", "alpha", "metallic", "roughness"],
    }


def run_frame(
    pipeline,
    image_path: Path,
    output_path: Path,
    fov_x: float,
    seed: int,
    image_resolution: int,
    resolution: int,
    max_num_tokens: int,
    decimation_target: int,
    texture_size: int,
    remesh: bool,
    preprocessed_path: Path,
    correspondence_path: Path | None,
    correspondence_resolution: int,
    crop_bbox: tuple[float, float, float, float] | None,
) -> dict:
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)
    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    image_preprocessed.save(preprocessed_path)

    distance = distance_from_fov(
        fov_x,
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0, image_resolution - 1]),
        mesh_scale=1.0,
        image_resolution=image_resolution,
    )["distance_from_x"]
    camera_params = {"camera_angle_x": float(fov_x), "distance": float(distance), "mesh_scale": 1.0}

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    pipeline_type = f"{resolution}_cascade"
    mesh_list, (_, _, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        },
        shape_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        tex_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )
    mesh = mesh_list[0]
    correspondence = None
    if correspondence_path is not None:
        correspondence = export_visible_surface_correspondence(
            mesh=mesh,
            camera_params=camera_params,
            output_path=correspondence_path,
            render_resolution=correspondence_resolution,
            crop_bbox=crop_bbox,
            pixal3d_to_glb=ROT_PIXAL3D_TO_GLB,
            source_image=image_preprocessed,
        )

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    glb.apply_transform(ROT_PIXAL3D_TO_GLB)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(output_path, extension_webp=True)

    return {
        "camera_params": camera_params,
        "pipeline_type": pipeline_type,
        "actual_resolution": int(res),
        "preprocessed_path": str(preprocessed_path),
        "output_path": str(output_path),
        "correspondence": correspondence,
        "export": {
            "decimation_target": int(decimation_target),
            "texture_size": int(texture_size),
            "remesh": bool(remesh),
        },
    }


def main() -> None:
    args = parse_args()
    vggt_output_dir = Path(args.vggt_output_dir)
    output_dir = Path(args.output_dir) if args.output_dir else vggt_output_dir / "pixal3d" / "sequence"
    input_dir = output_dir / "inputs"
    preprocessed_dir = output_dir / "preprocessed"
    glb_dir = output_dir / "glb"
    correspondence_dir = output_dir / "correspondence_maps"
    metadata_dir = output_dir / "metadata"
    log_path = output_dir / "status.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    dynamic = load_dynamic(vggt_output_dir, args.dynamic_file)
    images = dynamic["images"].astype(np.uint8)
    masks = dynamic["masks"].astype(np.uint8)
    intrinsics = dynamic["intrinsic"].astype(np.float64)
    num_frames, height, width = masks.shape
    end = num_frames if args.end < 0 else min(args.end, num_frames)
    start = max(args.start, 0)
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, num_frames={num_frames}")

    run_config = {
        "vggt_output_dir": str(vggt_output_dir),
        "output_dir": str(output_dir),
        "dynamic_file": args.dynamic_file,
        "frame_range": [start, end],
        "num_frames": num_frames,
        "image_size_wh": [int(width), int(height)],
        "model_path": args.model_path,
        "low_vram": bool(args.low_vram),
        "resolution": int(args.resolution),
        "seed": int(args.seed),
        "crop_padding": float(args.crop_padding),
        "export_correspondence_maps": bool(args.export_correspondence_maps),
        "correspondence_resolution": int(args.correspondence_resolution),
        "image_cond_configs": IMAGE_COND_CONFIGS,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[Batch] Output dir: {output_dir}", flush=True)
    print(f"[Batch] Frames: {start}..{end - 1} ({end - start})", flush=True)
    pipeline = build_pipeline(args.model_path, args.low_vram)

    for frame_idx in range(start, end):
        frame_start = time.time()
        glb_path = glb_dir / f"{frame_idx:06d}.glb"
        correspondence_path = correspondence_dir / f"{frame_idx:06d}.npz"
        metadata_path = metadata_dir / f"{frame_idx:06d}.json"
        image_path = input_dir / f"{frame_idx:06d}_rgba.png"
        preprocessed_path = preprocessed_dir / f"{frame_idx:06d}_preprocessed.png"

        expected_outputs_exist = glb_path.is_file() and metadata_path.is_file()
        if args.export_correspondence_maps:
            expected_outputs_exist = expected_outputs_exist and correspondence_map_is_current(correspondence_path)
        if args.skip_existing and not args.overwrite and expected_outputs_exist:
            print(f"[Batch] Skip existing frame {frame_idx:06d}", flush=True)
            append_jsonl(
                log_path,
                {
                    "frame_idx": frame_idx,
                    "status": "skipped",
                    "output_path": str(glb_path),
                    "correspondence_path": str(correspondence_path) if args.export_correspondence_maps else None,
                },
            )
            continue

        try:
            mask = masks[frame_idx]
            bbox = alpha_bbox(mask, args.crop_padding)
            fov_x = crop_fov_x(intrinsics[frame_idx], bbox, width)
            make_rgba(images[frame_idx], mask, image_path)

            result = run_frame(
                pipeline=pipeline,
                image_path=image_path,
                output_path=glb_path,
                fov_x=fov_x,
                seed=args.seed,
                image_resolution=args.image_resolution,
                resolution=args.resolution,
                max_num_tokens=args.max_num_tokens,
                decimation_target=args.decimation_target,
                texture_size=args.texture_size,
                remesh=args.remesh,
                preprocessed_path=preprocessed_path,
                correspondence_path=correspondence_path if args.export_correspondence_maps else None,
                correspondence_resolution=args.correspondence_resolution,
                crop_bbox=bbox,
            )
            elapsed = time.time() - frame_start
            metadata = {
                "frame_idx": frame_idx,
                "status": "ok",
                "elapsed_sec": elapsed,
                "input_rgba_path": str(image_path),
                "mask_bbox_xyxy_processed": None if bbox is None else [float(v) for v in bbox],
                "intrinsic": intrinsics[frame_idx].tolist(),
                "fov_x_radians_after_crop": float(fov_x),
                "processed_image_size_wh": [int(width), int(height)],
                **result,
            }
            metadata_dir.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            append_jsonl(
                log_path,
                {
                    "frame_idx": frame_idx,
                    "status": "ok",
                    "elapsed_sec": elapsed,
                    "output_path": str(glb_path),
                    "correspondence_path": str(correspondence_path) if args.export_correspondence_maps else None,
                },
            )
            print(f"[Batch] Done frame {frame_idx:06d} in {elapsed:.1f}s -> {glb_path}", flush=True)
        except Exception as exc:
            elapsed = time.time() - frame_start
            error = {
                "frame_idx": frame_idx,
                "status": "error",
                "elapsed_sec": elapsed,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(log_path, error)
            print(f"[Batch] ERROR frame {frame_idx:06d}: {exc!r}", flush=True)
            if args.stop_on_error:
                raise
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Preprocess scenes for the Surflo evaluation benchmark.

Runs the VGGT-1B backbone once per image subset and caches everything the
evaluation harness needs to rebuild a batch **without** re-running VGGT:

  * used-layer aggregated tokens (sparse 24-entry list, ``fp16``),
  * VGGT + COLMAP cameras,
  * the COLMAP <-> VGGT alignment ``(L, T)``,
  * scene extent,
  * optionally VGGT world points / RGB images.

GT surface points/normals are consolidated separately into ``surface_data.npz``
(shuffled, re-chunked, uncompressed for fast lazy per-chunk loading) so that
fresh random surface samples can be drawn per scene at eval time and aligned
on-the-fly with ``(L, T)``.

The output layout matches :class:`surflo.data.preprocessed.PreprocessedSceneDataset`:

    output_dir/<scene_id>/
        sample_0000_views_016.pt
        ...
        surface_data.npz

Usage:
    python scripts/preprocess.py \
        --scene_list scene_list.txt \
        --data_dir /path/to/DL3DV \
        --output_dir /path/to/preprocessed \
        --n_samples 1 --n_images 16 --spatial_scale 20.0 \
        --save_vggt_world_points --save_rgb_images
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from surflo.data.utils import (
    find_affine_transform,
    find_robust_alignment_transform,
    load_and_preprocess_images,
    load_colmap_cameras,
    load_depth_data,
)
from surflo.structures.multi_cameras import (
    get_multi_cameras_from_intrinsics_and_extrinsics,
)
from surflo.utils.geometry import depths_to_points_parallel_batched
from surflo.nn.vggt.models.vggt import VGGT
from surflo.nn.vggt.utils.pose_enc import pose_encoding_to_extri_intri

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("preprocess")

TARGET_SHAPE = np.array((280, 518))  # (H, W), must match the eval dataloader
SIZE_DIVIDER = 14
TARGET_RATIO = 16.0 / 9.0
TOPK_RATIO = 0.25

# Must match SurfaceNet.intermediate_layer_idx + the camera-token layer (last=23).
USED_TOKEN_LAYER_INDICES = [4, 11, 17, 23]


# ---------------------------------------------------------------------------
# GT surface consolidation
# ---------------------------------------------------------------------------
def consolidate_surface_data(scene_dir: Path, output_dir: Path, load_normals: bool = True) -> bool:
    """Concatenate the per-scene surface point/normal chunks into one npz.

    Reads directly from a ``<scene>.tar`` archive when the directory itself is
    absent. The points are shuffled once and re-chunked so each chunk keeps
    spatial diversity; the archive is uncompressed so ``np.load`` can lazily
    map a single ~MB chunk instead of the whole cloud.
    """
    output_file = output_dir / "surface_data.npz"
    if output_file.exists():
        _log.debug(f"Surface data already consolidated: {output_file}")
        return True

    point_subdir = "gw_output/surface_point_labels"
    normal_subdir = "gw_output/surface_normal_labels"

    tar_path = scene_dir.with_suffix(".tar")
    use_tar = tar_path.is_file() and not scene_dir.is_dir()

    if use_tar:
        import io
        import tarfile

        all_points, all_normals = [], []
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            point_members = sorted(
                [m for m in members if point_subdir in m.name and m.isfile()],
                key=lambda m: m.name,
            )
            if not point_members:
                _log.warning(f"No surface point files in {tar_path}")
                return False
            for pm in point_members:
                buf = tf.extractfile(pm)
                all_points.append(torch.load(io.BytesIO(buf.read()), map_location="cpu"))
            if load_normals:
                normal_members = sorted(
                    [m for m in members if normal_subdir in m.name and m.isfile()],
                    key=lambda m: m.name,
                )
                for nm in normal_members:
                    buf = tf.extractfile(nm)
                    all_normals.append(torch.load(io.BytesIO(buf.read()), map_location="cpu"))
    else:
        point_dir = scene_dir / point_subdir
        if not point_dir.is_dir():
            _log.warning(f"Surface point dir not found: {point_dir}")
            return False
        all_points = [torch.load(f, map_location="cpu") for f in sorted(point_dir.iterdir())]
        all_normals = []
        if load_normals:
            normal_dir = scene_dir / normal_subdir
            if not normal_dir.is_dir():
                _log.warning(f"Surface normal dir not found: {normal_dir}")
                return False
            all_normals = [torch.load(f, map_location="cpu") for f in sorted(normal_dir.iterdir())]

    surface_points = torch.cat(all_points, dim=0).float()
    perm = torch.randperm(surface_points.shape[0])
    surface_points = surface_points[perm]
    if load_normals and all_normals:
        surface_normals = torch.cat(all_normals, dim=0).float()[perm]

    n_chunks = len(all_points)
    chunk_size = (surface_points.shape[0] + n_chunks - 1) // n_chunks
    arrays = {"n_chunks": np.array(n_chunks)}
    for i in range(n_chunks):
        s, e = i * chunk_size, min((i + 1) * chunk_size, surface_points.shape[0])
        arrays[f"points_{i:03d}"] = surface_points[s:e].numpy()
        if load_normals and all_normals:
            arrays[f"normals_{i:03d}"] = surface_normals[s:e].numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **arrays)
    _log.info(
        f"Consolidated {surface_points.shape[0]:,} surface points into "
        f"{n_chunks} chunks -> {output_file}"
    )
    return True


# ---------------------------------------------------------------------------
# Scene IO helpers
# ---------------------------------------------------------------------------
def get_scene_dir(data_dir: str, scene_relpath: str) -> Path:
    return Path(data_dir) / scene_relpath


@contextmanager
def open_scene_dir(scene_dir: Path, data_dir: Path):
    """Yield a usable scene dir, extracting ``<scene>.tar`` to a tmp dir if needed."""
    if scene_dir.is_dir():
        yield scene_dir
        return
    tar_path = scene_dir.with_suffix(".tar")
    if not tar_path.is_file():
        raise FileNotFoundError(
            f"Scene not found as directory ({scene_dir}) or tar ({tar_path})"
        )
    import shutil
    import tarfile

    rel = scene_dir.relative_to(data_dir)
    tmp_dir = data_dir.parent / "DL3DV-preprocessing-tmp" / rel
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(tmp_dir)
        extracted = tmp_dir / scene_dir.name
        if not extracted.is_dir():
            candidates = [c for c in tmp_dir.iterdir() if c.is_dir()]
            if len(candidates) == 1:
                extracted = candidates[0]
            else:
                raise RuntimeError(
                    f"Unexpected tar structure in {tar_path}: {[c.name for c in candidates]}"
                )
        yield extracted
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_scene_metadata(scene_dir: Path):
    cameras_file = scene_dir / "gw_output" / "cameras.json"
    with open(cameras_file, "r") as f:
        cameras_list = json.load(f)
    scene_extent = torch.load(
        scene_dir / "gw_output" / "scene_extent.pth", map_location="cpu"
    )
    return cameras_file, cameras_list, scene_extent


def get_valid_image_paths(scene_dir: Path, cameras_list: list) -> list:
    images_dir = scene_dir / "images"
    extension = os.listdir(images_dir)[0].split(".")[-1]
    all_image_paths = sorted(glob(str(images_dir / f"*.{extension}")))
    valid_names = {cam["img_name"] + "." + extension for cam in cameras_list}
    return [p for p in all_image_paths if Path(p).name in valid_names]


# ---------------------------------------------------------------------------
# VGGT run + COLMAP<->VGGT alignment
# ---------------------------------------------------------------------------
def sparsify_tokens(
    aggregated_tokens_list: List[torch.Tensor],
    used_indices: List[int],
    tokens_dtype: torch.dtype = torch.float16,
) -> List[Optional[torch.Tensor]]:
    used = set(used_indices)
    return [
        t.cpu().to(tokens_dtype) if i in used else None
        for i, t in enumerate(aggregated_tokens_list)
    ]


@torch.no_grad()
def run_vggt_and_align(
    vggt_model,
    images: torch.Tensor,
    colmap_intrinsics: torch.Tensor,
    colmap_extrinsics: torch.Tensor,
    depth_data: torch.Tensor,
    spatial_scale: float,
    save_vggt_world_points: bool,
    save_rgb_images: bool,
    dtype: torch.dtype,
    device: torch.device,
    tokens_dtype: torch.dtype = torch.float16,
):
    images_batched = images.unsqueeze(0).to(device)
    colmap_intrinsics_b = colmap_intrinsics.unsqueeze(0)
    colmap_extrinsics_b = colmap_extrinsics.unsqueeze(0)
    depth_data_b = depth_data.unsqueeze(0)

    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = vggt_model(images_batched, return_aggregated_tokens=True)

    vggt_extrinsics, vggt_intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images_batched.shape[-2:]
    )
    vggt_depth = predictions["depth"]

    colmap_world_points = depths_to_points_parallel_batched(
        colmap_intrinsics_b.to(device),
        colmap_extrinsics_b.to(device),
        rearrange(depth_data_b, "B N 1 H W -> B N H W 1").to(device),
        to_world=True,
    ).to(torch.float32)
    vggt_world_points = depths_to_points_parallel_batched(
        vggt_intrinsics, vggt_extrinsics, vggt_depth, to_world=True,
    )

    conf_map = predictions["depth_conf"].view(1, -1)
    n_pixels = conf_map.shape[-1]
    conf_index = torch.topk(
        conf_map, k=int(TOPK_RATIO * n_pixels), dim=-1, sorted=False
    ).indices
    colmap_pts_filtered = torch.gather(
        colmap_world_points.reshape(1, -1, 3), dim=-2,
        index=conf_index.view(1, -1, 1).repeat(1, 1, 3),
    )
    vggt_pts_filtered = torch.gather(
        vggt_world_points.reshape(1, -1, 3), dim=-2,
        index=conf_index.view(1, -1, 1).repeat(1, 1, 3),
    )

    all_colmap_cameras = get_multi_cameras_from_intrinsics_and_extrinsics(
        intrinsics=colmap_intrinsics_b.to(device),
        extrinsics=colmap_extrinsics_b.to(device),
        data_device=device,
    )
    all_vggt_cameras = get_multi_cameras_from_intrinsics_and_extrinsics(
        intrinsics=vggt_intrinsics, extrinsics=vggt_extrinsics, data_device=device,
    )

    try:
        _, L, T = find_robust_alignment_transform(
            colmap_world_points=colmap_pts_filtered,
            vggt_world_points=vggt_pts_filtered.to(torch.float32),
            colmap_cameras=all_colmap_cameras,
            vggt_cameras=all_vggt_cameras,
            main_cam_idx=0,
            weight_quantile_threshold=0.5,
            spatial_scale=spatial_scale,
        )
    except ValueError:
        _, L, T = find_affine_transform(
            X=colmap_pts_filtered, Y=vggt_pts_filtered.to(torch.float32),
        )

    if L.isnan().any() or T.isnan().any():
        _log.warning("Alignment transform contains NaN values")
        return None

    result = {
        "aggregated_tokens_list": sparsify_tokens(
            predictions["aggregated_tokens_list"], USED_TOKEN_LAYER_INDICES, tokens_dtype
        ),
        "patch_start_idx": predictions["patch_start_idx"],
        "vggt_extrinsics": vggt_extrinsics.cpu().float(),
        "vggt_intrinsics": vggt_intrinsics.cpu().float(),
        "alignment_L": L.cpu().float(),
        "alignment_T": T.cpu().float(),
    }
    if save_vggt_world_points:
        result["vggt_world_points"] = vggt_world_points.cpu().to(torch.float16)
    if save_rgb_images:
        result["rgb_images"] = predictions["images"].cpu().to(torch.float16)
    return result


def preprocess_scene(
    vggt_model,
    scene_dir: Path,
    output_dir: Path,
    n_samples: int,
    n_images_list: List[int],
    spatial_scale: float,
    save_vggt_world_points: bool,
    save_rgb_images: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    cameras_file, cameras_list, scene_extent = load_scene_metadata(scene_dir)
    all_image_paths = get_valid_image_paths(scene_dir, cameras_list)
    num_images = len(all_image_paths)

    if num_images < max(n_images_list):
        _log.warning(
            f"Scene {scene_dir.name} has only {num_images} images, "
            f"need {max(n_images_list)}. Skipping."
        )
        return False

    scene_radius = scene_extent["scene_radius"]
    scene_center = scene_extent["scene_center"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for n_images in n_images_list:
        if num_images < n_images:
            _log.warning(f"Scene {scene_dir.name}: only {num_images} images, skip n_images={n_images}.")
            continue
        for sample_idx in range(n_samples):
            output_file = output_dir / f"sample_{sample_idx:04d}_views_{n_images:03d}.pt"
            if output_file.exists():
                _log.debug(f"Skipping existing {output_file}")
                continue

            segment_size = num_images / n_images
            ids = np.array([
                np.random.randint(int(i * segment_size), int((i + 1) * segment_size))
                for i in range(n_images)
            ])
            np.random.shuffle(ids)
            image_paths = [all_image_paths[i] for i in ids]

            images = load_and_preprocess_images(
                image_paths, mode="no_stretch", target_ratio=TARGET_RATIO,
            )
            colmap_intrinsics, colmap_extrinsics, _ = load_colmap_cameras(
                camera_file=str(cameras_file),
                target_size=TARGET_SHAPE[1],
                image_names=[str(p) for p in image_paths],
                size_divider=SIZE_DIVIDER,
                target_ratio=TARGET_RATIO,
            )
            depth_data = load_depth_data(
                str(scene_dir / "gw_output" / "depth"),
                [str(p) for p in image_paths],
                target_size=TARGET_SHAPE[1],
                size_divider=SIZE_DIVIDER,
                target_ratio=TARGET_RATIO,
            )

            result = run_vggt_and_align(
                vggt_model=vggt_model, images=images,
                colmap_intrinsics=colmap_intrinsics, colmap_extrinsics=colmap_extrinsics,
                depth_data=depth_data, spatial_scale=spatial_scale,
                save_vggt_world_points=save_vggt_world_points,
                save_rgb_images=save_rgb_images, dtype=dtype, device=device,
            )
            if result is None:
                _log.warning(
                    f"Alignment failed for {scene_dir.name} sample {sample_idx} "
                    f"n_images={n_images}, skipping."
                )
                continue

            sample = {
                "ids": ids,
                "image_names": [Path(p).name for p in image_paths],
                "colmap_intrinsics": colmap_intrinsics,
                "colmap_extrinsics": colmap_extrinsics,
                "scene_radius": scene_radius,
                "scene_center": scene_center,
                **result,
            }
            torch.save(sample, output_file)
            _log.debug(f"Saved {output_file}.")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess scenes with VGGT for Surflo eval.")
    p.add_argument("--scene_list", type=str, required=True,
                   help="File with one scene relpath per line (e.g. '10K/SCENE_HASH').")
    p.add_argument("--data_dir", type=str, required=True, help="Dataset root.")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    p.add_argument("--n_samples", type=int, default=1, help="Image subsets sampled per (scene, n_images).")
    p.add_argument("--n_images", type=int, default=16, help="Images per sample.")
    p.add_argument("--min_images", type=int, default=None, help="If set, sweep n_images from min to max.")
    p.add_argument("--max_images", type=int, default=None, help="Upper bound for the n_images sweep.")
    p.add_argument("--spatial_scale", type=float, default=20.0, help="Robust alignment spatial scale.")
    p.add_argument("--save_vggt_world_points", action="store_true",
                   help="Also cache VGGT world points (required for per-scene cull + source sampling).")
    p.add_argument("--save_rgb_images", action="store_true",
                   help="Also cache RGB images (required for guided confidence recovery).")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    if args.min_images is not None:
        max_img = args.max_images if args.max_images is not None else args.n_images
        n_images_list = list(range(args.min_images, max_img + 1))
    else:
        n_images_list = [args.n_images]
    _log.info(f"n_images list: {n_images_list} ({args.n_samples} sample(s) each)")

    with open(args.scene_list, "r") as f:
        scene_list = [line.strip() for line in f if line.strip()]
    end_idx = args.end_idx if args.end_idx > 0 else len(scene_list)
    scene_list = scene_list[args.start_idx:end_idx]
    _log.info(f"Processing {len(scene_list)} scenes (indices {args.start_idx}..{end_idx - 1}).")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    _log.info("Loading VGGT-1B ...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval()
    for prm in vggt_model.parameters():
        prm.requires_grad = False
    _log.info("VGGT loaded.")

    successes = failures = 0
    for scene_relpath in tqdm(scene_list, desc="Preprocessing scenes"):
        scene_dir = get_scene_dir(args.data_dir, scene_relpath)
        output_subdir = Path(args.output_dir) / scene_dir.name
        try:
            consolidate_surface_data(scene_dir, output_subdir, load_normals=True)
            with open_scene_dir(scene_dir, Path(args.data_dir)) as resolved_dir:
                ok = preprocess_scene(
                    vggt_model=vggt_model, scene_dir=resolved_dir, output_dir=output_subdir,
                    n_samples=args.n_samples, n_images_list=n_images_list,
                    spatial_scale=args.spatial_scale,
                    save_vggt_world_points=args.save_vggt_world_points,
                    save_rgb_images=args.save_rgb_images, dtype=dtype, device=device,
                )
            successes += int(bool(ok))
            failures += int(not ok)
        except Exception as e:  # noqa: BLE001
            _log.error(f"Failed to process {scene_relpath}: {e}")
            failures += 1

    _log.info(f"Done: {successes} successes, {failures} failures.")


if __name__ == "__main__":
    main()

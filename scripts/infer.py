"""Surflo inference entry point.

Reconstruct a surface from a folder of RGB images:

    # Plain (unguided) flow-matching -> initial.ply + final.ply
    # (source.n_images defaults to null = every image in the folder)
    python scripts/infer.py mode=plain plain=default \
        ckpt=/path/to/surflo_v0.pt \
        source.image_folder=/path/to/images \
        output_dir=outputs/surflo_plain

    # Guided flow + 3DGS densification (+ optional texture)
    #   -> point_cloud_normals.ply + point_cloud_rgb.ply + mesh.ply
    python scripts/infer.py mode=guided guided=default mesh=default \
        ckpt=/path/to/surflo_v0.pt \
        source.image_folder=/path/to/images source.n_images=16 \
        num_query_points=100000 \
        expert.conf_threshold_for_normal_guidance=2.0 \
        texture.enabled=true texture.n_iterations=0 \
        output_dir=outputs/surflo_densify_default

Two modes (`mode=`):
  * plain  -- source sample -> ODE -> oriented points. No rendering.
  * guided -- rendering-guided flow with 3DGS-style densification ->
              Gaussians -> wrapping mesh.

Only the config block matching `mode` is consumed; the others are ignored.
"""
from __future__ import annotations

import gc
import inspect
import json
import logging
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

# Allow ``python scripts/infer.py`` from a source checkout (before an editable
# install) by putting the package root (the parent of scripts/) on sys.path.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from surflo.model.loader import load_model
from surflo.data.image_folder import build_image_folder_batch, list_images_in_folder
from surflo.inference.engine import plain_inference
from surflo.utils.io import (
    normals_to_rgb_uint8,
    uint8_colors_from_floats,
    write_mesh_ply,
    write_points_ply,
)
from surflo.utils.guided_result import (
    gaussians_from_guided_result,
    build_refined_cameras_from_guided_result,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cuda_sync() -> None:
    """Flush pending CUDA work so the wall-clock timings below are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Monodepth expert (lazily loaded, cached across scenes)
# ---------------------------------------------------------------------------
_MONO_EXPERT_CACHE = None


def _expert_required(guided_kwargs: Dict[str, Any]) -> Tuple[bool, bool]:
    """Return ``(needs_monodepth, needs_normals)`` for a kwargs dict.

    Each is True only when the corresponding ``use_*_guidance`` flag is set AND
    the tensor isn't already provided.
    """
    needs_mono = bool(guided_kwargs.get("use_monodepth_guidance", False))
    needs_norm = bool(guided_kwargs.get("use_normal_guidance", False))
    if needs_mono and guided_kwargs.get("monodepths") is not None:
        needs_mono = False
    if needs_norm and guided_kwargs.get("normal_guidances") is not None:
        needs_norm = False
    return needs_mono, needs_norm


def _maybe_inject_experts(
    *,
    batch: dict,
    scene_idx: int,
    guided_kwargs: Dict[str, Any],
    expert_cfg: DictConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Build monodepth-expert tensors when the preset requests them.

    Runs DA3 per scene, resizes the prediction back to input resolution in
    disparity space, derives per-view normals from the high-res monodepth
    (VGGT-depth aligned, confidence-gated), and puts ``monodepths`` /
    ``normal_guidances`` into a *copy* of ``guided_kwargs``.
    """
    global _MONO_EXPERT_CACHE

    needs_mono, needs_norm = _expert_required(guided_kwargs)
    if not (needs_mono or needs_norm):
        return guided_kwargs

    if not bool(expert_cfg.get("enabled", True)):
        raise RuntimeError(
            "The selected preset requests monodepth / normal guidance but "
            "`expert.enabled=false`. Either enable the expert or flip "
            "`use_monodepth_guidance` / `use_normal_guidance` off in the preset."
        )

    # Imported here so plain / no-expert runs never touch DepthAnything3.
    from surflo.inference.expert import MonodepthExpert, compute_guidance_tensors
    from surflo.structures.cameras import get_cameras_from_intrinsics_and_extrinsics

    if _MONO_EXPERT_CACHE is None:
        _log.info(
            f"[infer] loading monodepth expert ({expert_cfg.monodepth_id}) "
            f"on {device}..."
        )
        _MONO_EXPERT_CACHE = MonodepthExpert(
            model_id=str(expert_cfg.monodepth_id), device=device,
        )

    scene_imgs = batch.get("rgb_images")
    if scene_imgs is None:
        raise RuntimeError(
            "Monodepth expert needs `batch['rgb_images']` (the raw RGB pixels), "
            "but it is missing from the batch."
        )

    cameras = None
    if needs_norm:
        cameras = get_cameras_from_intrinsics_and_extrinsics(
            intrinsics=batch["vggt_intrinsics"][scene_idx],
            extrinsics=batch["vggt_extrinsics"][scene_idx],
            images=scene_imgs[scene_idx],
            data_device=str(device),
        )

    monodepths, normals = compute_guidance_tensors(
        batch=batch,
        scene_idx=scene_idx,
        expert=_MONO_EXPERT_CACHE,
        pred_res=int(expert_cfg.pred_res),
        process_res_method=str(expert_cfg.process_res_method),
        conf_threshold_for_normal_guidance=float(
            expert_cfg.conf_threshold_for_normal_guidance
        ),
        compute_monodepths=needs_mono,
        compute_normals=needs_norm,
        cameras=cameras,
        device=device,
    )

    out = dict(guided_kwargs)
    if needs_mono and monodepths is not None:
        out["monodepths"] = monodepths
        _log.info(f"[infer] injected monodepths {tuple(monodepths.shape)}.")
    if needs_norm and normals is not None:
        out["normal_guidances"] = normals
        _log.info(f"[infer] injected normal_guidances {tuple(normals.shape)}.")
    return out


# ---------------------------------------------------------------------------
# Plain inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_plain(
    model, batch: dict, *, scene_idx: int, plain_cfg: DictConfig,
    num_query_points: int, seed: int, device: torch.device,
) -> Dict[str, Any]:
    """Run plain (guidance-free) inference via the runner-based engine.

    Uses :func:`surflo.inference.engine.plain_inference`, which drives the same
    flow-matching Euler ODE as the guided variants (via ``run_plain_flow``). The
    runner manages its own bf16 autocast around the flow network and the token
    compression, so -- exactly like the guided path -- no outer ``model.dtype``
    autocast is needed here (source sampling / unlift run in fp32). This is
    truly unguided: any nonzero ``plain.guidance_scale`` is ignored.

    ``return_source`` mirrors ``save_initial`` so the world-space source cloud
    (for ``initial.ply``) is decoded and returned as ``source_points`` /
    ``source_normals`` -- keeping :func:`_save_plain_outputs` unchanged.

    ``num_query_points`` comes from the top-level config (shared with guided
    mode), not from ``plain_cfg``.
    """
    gscale = float(plain_cfg.get("guidance_scale", 0.0) or 0.0)
    if gscale != 0.0:
        _log.info(
            f"[infer] plain mode is unguided; ignoring plain.guidance_scale={gscale}."
        )
    return plain_inference(
        model, batch,
        scene_idx=scene_idx,
        num_steps=int(plain_cfg.num_steps),
        num_query_points=int(num_query_points),
        num_points_per_batch=int(plain_cfg.num_points_per_batch),
        seed=int(seed),
        return_source=bool(plain_cfg.get("save_initial", True)),
    )


def _save_plain_outputs(
    out_dir: Path, *, plain_result: Dict[str, Any],
    plain_cfg: DictConfig, ply_cfg: DictConfig,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    fallback_color = np.asarray(
        ply_cfg.get("initial_color", [128, 128, 255]), dtype=np.uint8,
    ).reshape(1, 3)

    if bool(plain_cfg.get("save_initial", True)):
        init_pts = plain_result["source_points"]
        init_nrm = plain_result.get("source_normals")
        if init_pts is not None:
            n_init = int(init_pts.shape[0])
            if init_nrm is not None:
                cols = normals_to_rgb_uint8(init_nrm)
                color_source = "source_noise_normals"
            else:
                cols = np.broadcast_to(fallback_color, (n_init, 3)).copy()
                color_source = "fallback_solid"
            n = write_points_ply(
                str(out_dir / "initial.ply"), init_pts,
                colors_uint8=cols, normals=init_nrm,
                binary=bool(ply_cfg.binary),
            )
            info["n_initial"] = n
            info["initial_color_source"] = color_source
            _log.info(f"[infer] wrote initial.ply with {n} points ({color_source}).")

    if bool(plain_cfg.get("save_final", True)):
        final_pts = plain_result["points"]
        final_nrm = plain_result.get("normals")
        cols = normals_to_rgb_uint8(final_nrm) if final_nrm is not None else None
        n = write_points_ply(
            str(out_dir / "final.ply"), final_pts,
            colors_uint8=cols, normals=final_nrm,
            binary=bool(ply_cfg.binary),
        )
        info["n_final"] = n
        _log.info(f"[infer] wrote final.ply with {n} points.")

    return info


# ---------------------------------------------------------------------------
# Guided inference
# ---------------------------------------------------------------------------
def _filter_kwargs_to_signature(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the target function does not accept, logging them once."""
    valid = set(inspect.signature(fn).parameters.keys())
    unknown = sorted(set(kwargs) - valid)
    if unknown:
        _log.info(
            f"[infer] dropping {len(unknown)} preset kwarg(s) not accepted by "
            f"{fn.__name__}: {unknown}"
        )
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return kwargs


def _run_guided(
    model, batch: dict, *, scene_idx: int, guided_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Call ``guided_inference`` with the (signature-filtered) kwargs.

    The rendering loss requires fp32 and its own backward pass, so this runs
    with autocast OFF and grads ON.
    """
    from surflo.inference.engine import guided_inference as fn

    guided_kwargs = _filter_kwargs_to_signature(fn, guided_kwargs)
    with torch.amp.autocast("cuda", enabled=False), torch.enable_grad():
        return fn(model=model, batch=batch, scene_idx=scene_idx, **guided_kwargs)


def _save_guided_outputs(
    out_dir: Path, *, batch: dict, scene_idx: int,
    guided_result: Dict[str, Any], mesh_cfg: DictConfig, ply_cfg: DictConfig,
    device: torch.device,
    filtered_centers_opacity_threshold: Optional[float] = None,
    texture_cfg: Optional[DictConfig] = None,
) -> Dict[str, Any]:
    """Write ``point_cloud_normals.ply`` + ``point_cloud_rgb.ply`` and ``mesh.ply``."""
    info: Dict[str, Any] = {}

    # ---- point_cloud_normals.ply (+ point_cloud_rgb.ply) ----
    # The reconstructed cloud is the guided Gaussian centers, optionally culled to
    # opacity >= filtered_centers_opacity_threshold. Mesh extraction (below) always
    # uses the FULL Gaussian set regardless of this cull.
    info.update(_write_point_clouds(
        out_dir, batch=batch, scene_idx=scene_idx,
        guided_result=guided_result, mesh_cfg=mesh_cfg, ply_cfg=ply_cfg,
        threshold=filtered_centers_opacity_threshold, device=device,
    ))

    # ---- mesh.ply (optional) ----
    if not bool(mesh_cfg.get("enabled", True)):
        info["mesh_enabled"] = False
        return info

    try:
        from surflo.extraction.occupancy.wrapping import (
            pivot_extraction_with_binary_search,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(f"[infer] wrapping-mesh extractor unavailable ({e}); skipping mesh.ply.")
        info["mesh_enabled"] = False
        info["mesh_error"] = str(e)
        return info

    gs = gaussians_from_guided_result(guided_result, device=device)
    cameras = build_refined_cameras_from_guided_result(
        batch=batch, scene_idx=scene_idx, guided_result=guided_result,
        apply_camera_correction=bool(mesh_cfg.get("use_refined_cameras", True)),
        data_device=str(device),
    )

    bg = torch.zeros(3, device=device, dtype=torch.float32)
    extract_kwargs = dict(
        sdf_mode="approximate",
        sdf_isosurface_value=float(mesh_cfg.sdf_isosurface_value),
        use_regular_pivots=bool(mesh_cfg.use_regular_pivots),
        std_factor=float(mesh_cfg.std_factor),
        n_pivots=int(mesh_cfg.n_pivots),
        use_smallest_axis_as_normal=bool(mesh_cfg.use_smallest_axis_as_normal),
        n_points_per_sdf_evaluation=int(mesh_cfg.n_points_per_sdf_evaluation),
        use_valid_mask=bool(mesh_cfg.use_valid_mask),
        filter_large_edges=bool(mesh_cfg.filter_large_edges),
        collapse_large_edges=bool(mesh_cfg.collapse_large_edges),
        mtet_on_cpu=bool(mesh_cfg.mtet_on_cpu),
        n_binary_steps=int(mesh_cfg.n_binary_steps),
        delaunay_method=str(mesh_cfg.get("delaunay_method", "geodel")),
    )
    _log.info(
        f"[infer] extracting wrapping mesh (N_gauss={int(gs.means.shape[0])}, "
        f"N_views={len(cameras)}); kwargs={extract_kwargs}"
    )
    # NOTE: meshing wall-clock is dominated by the CPU Delaunay triangulation of
    # the pivots inside the extractor, not by the GPU occupancy queries -- so
    # this step scales with the pivot count. `mesh.delaunay_method` selects the
    # backend; the default `geodel` is multi-threaded (~20x faster than the
    # single-threaded `scipy` fallback at 200k points on 32 cores).
    _cuda_sync()
    t_mesh = time.time()
    mesh = pivot_extraction_with_binary_search(
        views=cameras, gaussians=gs, background=bg,
        kernel_size=float(mesh_cfg.kernel_size), **extract_kwargs,
    )
    _cuda_sync()
    mesh_elapsed = time.time() - t_mesh
    n_v, n_f = write_mesh_ply(
        str(out_dir / "mesh.ply"), mesh.verts, mesh.faces,
        vertex_colors_uint8=(
            (mesh.verts_colors.detach().cpu().float().clamp(0, 1) * 255 + 0.5)
            .to(torch.uint8).numpy()
            if mesh.verts_colors is not None else None
        ),
        binary=bool(ply_cfg.binary),
    )
    info["mesh_enabled"] = True
    info["mesh_n_verts"] = n_v
    info["mesh_n_faces"] = n_f
    info["mesh_elapsed_s"] = float(mesh_elapsed)
    _log.info(
        f"[infer] wrote mesh.ply: V={n_v}, F={n_f} (meshing took "
        f"{mesh_elapsed:.1f}s; dominated by the Delaunay triangulation "
        f"of the pivots)."
    )

    # ---- Optional textured-mesh refinement ----
    if (
        texture_cfg is not None
        and bool(texture_cfg.get("enabled", False))
        and n_v > 0 and n_f > 0
    ):
        info.update(_maybe_texture_mesh(
            out_dir, batch=batch, scene_idx=scene_idx, mesh=mesh,
            cameras=cameras, texture_cfg=texture_cfg, ply_cfg=ply_cfg,
            device=device,
        ))
    return info


@torch.no_grad()
def _write_point_clouds(
    out_dir: Path, *, batch: dict, scene_idx: int, guided_result: Dict[str, Any],
    mesh_cfg: DictConfig, ply_cfg: DictConfig, threshold: Optional[float],
    device: torch.device,
) -> Dict[str, Any]:
    """Write the reconstructed cloud as ``point_cloud_normals.ply`` (+ RGB companion).

    The points are the guided Gaussian centers, optionally culled to
    ``opacity >= threshold`` (``None`` / ``<= 0`` keeps every point). Two files
    are written for the same (culled) points:

      * ``point_cloud_normals.ply`` -- colored by the surface normals.
      * ``point_cloud_rgb.ply``     -- colored by TSDF-fused input RGB.

    The RGB pass renders TSDF depth from *all* Gaussians (so occlusion reflects
    the full scene, not just the kept subset) and colors only the kept points.
    Mesh extraction is untouched and always uses the full Gaussian set.
    """
    info: Dict[str, Any] = {}
    centers_pts = guided_result["points"].detach().float()
    centers_nrm = guided_result.get("aux_normals")
    if centers_nrm is None:
        centers_nrm = guided_result.get("normals")

    # ---- Optional opacity cull ----
    pts = centers_pts
    nrm = centers_nrm
    keep_desc = f"{int(pts.shape[0])} points (all)"
    if threshold is not None and float(threshold) > 0.0:
        opacities = guided_result.get("aux_opacities")
        if opacities is None:
            _log.warning("[infer] opacity cull requested but no aux_opacities; keeping all points.")
        else:
            opacities = opacities.detach().float().reshape(-1)
            n_total = int(opacities.numel())
            if centers_pts.shape[0] != n_total:
                _log.warning("[infer] aux_opacities/points shape mismatch; keeping all points.")
            else:
                keep = opacities >= float(threshold)
                pts = centers_pts[keep]
                nrm = centers_nrm[keep] if centers_nrm is not None else None
                keep_desc = (
                    f"{int(keep.sum().item())}/{n_total} points "
                    f"(opacity >= {threshold})"
                )
                info["point_cloud_opacity_threshold"] = float(threshold)

    # ---- point_cloud_normals.ply ----
    cols = normals_to_rgb_uint8(nrm) if nrm is not None else None
    n_pts = write_points_ply(
        str(out_dir / "point_cloud_normals.ply"), pts,
        colors_uint8=cols, normals=nrm, binary=bool(ply_cfg.binary),
    )
    info["n_point_cloud"] = n_pts
    _log.info(f"[infer] wrote point_cloud_normals.ply with {keep_desc}.")

    # ---- point_cloud_rgb.ply (per-point TSDF fusion of the input RGB) ----
    if pts.shape[0] > 0:
        try:
            from surflo.extraction.texture.optimize import (
                evaluate_point_colors_via_gaussian_tsdf,
            )
            scene_imgs = batch.get("rgb_images")
            if scene_imgs is None:
                scene_imgs = batch.get("images")
            if scene_imgs is None:
                raise RuntimeError("no 'rgb_images'/'images' in batch")
            scene_images = scene_imgs[scene_idx].to(device)

            # Render the TSDF depth from ALL Gaussians so occlusion reflects the
            # full scene, then color only the kept points.
            gs_full = gaussians_from_guided_result(guided_result, device=device)
            cams_rgb = build_refined_cameras_from_guided_result(
                batch=batch, scene_idx=scene_idx, guided_result=guided_result,
                apply_camera_correction=bool(mesh_cfg.get("use_refined_cameras", True)),
                data_device=str(device),
            )
            rgb_colors = evaluate_point_colors_via_gaussian_tsdf(
                points=pts.to(device), gaussians=gs_full,
                cameras=cams_rgb, images=scene_images,
            )
            n_rgb = write_points_ply(
                str(out_dir / "point_cloud_rgb.ply"), pts,
                colors_uint8=uint8_colors_from_floats(rgb_colors),
                normals=nrm, binary=bool(ply_cfg.binary),
            )
            info["n_point_cloud_rgb"] = n_rgb
            _log.info(f"[infer] wrote point_cloud_rgb.ply with {n_rgb} points.")
        except Exception as e:  # noqa: BLE001
            _log.warning(f"[infer] failed to write point_cloud_rgb.ply: {e}")
            info["point_cloud_rgb_error"] = str(e)
    return info


def _maybe_texture_mesh(
    out_dir: Path, *, batch: dict, scene_idx: int, mesh, cameras,
    texture_cfg: DictConfig, ply_cfg: DictConfig, device: torch.device,
) -> Dict[str, Any]:
    """Dispatch vertex-color or UV-texture refinement of ``mesh.ply``."""
    scene_imgs = batch.get("rgb_images")
    if scene_imgs is None:
        scene_imgs = batch.get("images")
    if scene_imgs is None:
        _log.warning("[infer] texture requested but no images in batch; skipping.")
        return {"texture_enabled": False, "texture_error": "no_images_in_batch"}

    scene_images = scene_imgs[scene_idx]
    mode = str(texture_cfg.get("mode", "vertex_colors")).strip().lower()
    stem = Path(str(texture_cfg.get("output_filename", "mesh_textured.ply"))).stem or "mesh_textured"

    if mode == "uv_texture":
        from surflo.extraction.texture.optimize import optimize_mesh_texture_and_save
        return optimize_mesh_texture_and_save(
            str(out_dir / f"{stem}.glb"),
            mesh=mesh, cameras=cameras, images=scene_images,
            texture_resolution=int(texture_cfg.get("texture_resolution", 2048)),
            atlas_padding=int(texture_cfg.get("atlas_padding", 2)),
            atlas_max_chart_iterations=int(texture_cfg.get("atlas_max_chart_iterations", 0)),
            dilate_iterations=int(texture_cfg.get("dilate_iterations", 4)),
            n_iterations=int(texture_cfg.get("n_iterations", 0)),
            learning_rate=float(texture_cfg.get("uv_texture_learning_rate", 0.01)),
            lambda_dssim=float(texture_cfg.get("lambda_dssim", 0.2)),
            init_color=texture_cfg.get("init_color", "tsdf"),
            tsdf_trunc_margin_first_pass_factor=float(
                texture_cfg.get("tsdf_trunc_margin_first_pass_factor", 2e-3)),
            tsdf_trunc_margin_fallback_factor=float(
                texture_cfg.get("tsdf_trunc_margin_fallback_factor", 1.0)),
            use_scalable_renderer_for_tsdf=bool(texture_cfg.get("use_scalable_renderer", True)),
            use_mipmaps=bool(texture_cfg.get("use_mipmaps", True)),
            use_antialiasing=bool(texture_cfg.get("use_antialiasing", True)),
        )

    if mode != "vertex_colors":
        _log.warning(f"[infer] unknown texture.mode={mode!r}; using 'vertex_colors'.")
    from surflo.extraction.texture.optimize import optimize_mesh_vertex_colors_and_save
    return optimize_mesh_vertex_colors_and_save(
        str(out_dir / f"{stem}.ply"),
        mesh=mesh, cameras=cameras, images=scene_images,
        n_iterations=int(texture_cfg.get("n_iterations", 0)),
        learning_rate=float(texture_cfg.get("learning_rate", 0.0025)),
        lambda_dssim=float(texture_cfg.get("lambda_dssim", 0.2)),
        use_scalable_renderer=bool(texture_cfg.get("use_scalable_renderer", True)),
        init_color=texture_cfg.get("init_color", "tsdf"),
        tsdf_trunc_margin_first_pass_factor=float(
            texture_cfg.get("tsdf_trunc_margin_first_pass_factor", 2e-3)),
        tsdf_trunc_margin_fallback_factor=float(
            texture_cfg.get("tsdf_trunc_margin_fallback_factor", 1.0)),
        binary=bool(ply_cfg.binary),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="infer", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    mode = str(cfg.mode)
    if mode not in ("plain", "guided"):
        raise ValueError(f"Unknown mode={mode!r}; expected plain/guided.")

    seed = int(cfg.get("seed", 42))
    set_global_seeds(seed)
    device = torch.device(str(cfg.get("device", "cuda")))

    model = load_model(
        cfg.model, ckpt_path=cfg.get("ckpt"), device=device,
        use_ema=bool(cfg.get("use_ema", True)),
    )

    # ---- Build the (single-scene) batch from the image folder ----
    src = cfg.source
    folder = str(src.image_folder)
    # `n_images: null` means "use every image in the folder" (matching the API,
    # where `n_images=None` does the same). Sampling is then a no-op.
    n_images = src.get("n_images", None)
    n_images = int(n_images) if n_images is not None else len(list_images_in_folder(folder))
    batch, selected = build_image_folder_batch(
        model,
        folder=folder,
        n_images=n_images,
        sampling=str(src.get("sampling", "uniform")),
        seed=seed,
        target_size=int(src.get("target_size", 518)),
        cull_radius=src.get("cull_radius", None),
        device=device,
    )
    _log.info(f"[infer] using {n_images} image(s) from {folder}")
    scene_idx = 0
    scene_id = batch.get("seq_name", ["scene"])[0]

    out_root = Path(str(cfg.output_dir))
    scene_dir = out_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    info: Dict[str, Any] = {"scene_id": scene_id, "mode": mode}
    if mode == "plain":
        # Time only the ODE inference (no experts / meshing in plain mode).
        _cuda_sync()
        t_ode = time.time()
        res = _run_plain(
            model, batch, scene_idx=scene_idx, plain_cfg=cfg.plain,
            num_query_points=int(cfg.num_query_points),
            seed=seed, device=device,
        )
        _cuda_sync()
        info["ode_inference_s"] = float(time.time() - t_ode)
        _log.info(
            f"[infer] ODE inference finished in {info['ode_inference_s']:.1f}s."
        )
        info.update(_save_plain_outputs(
            scene_dir, plain_result=res, plain_cfg=cfg.plain, ply_cfg=cfg.ply,
        ))
    else:
        guided_kwargs = OmegaConf.to_container(cfg.get("guided"), resolve=True)
        if guided_kwargs is None:
            raise RuntimeError(
                f"mode={mode} but `guided` config block is missing.")
        # `num_query_points` is shared with plain mode and lives at the top level,
        # so inject it into the preset kwargs before they are splatted into
        # `guided_inference` (which would otherwise fall back to its own default).
        guided_kwargs["num_query_points"] = int(cfg.num_query_points)
        # Monodepth / normal priors are computed here and are deliberately NOT
        # included in the reported inference time.
        guided_kwargs = _maybe_inject_experts(
            batch=batch, scene_idx=scene_idx, guided_kwargs=guided_kwargs,
            expert_cfg=cfg.expert, device=device,
        )
        # Time ONLY the ODE inference (guidance flow + the extra polish
        # iterations): after the priors are ready, and before meshing.
        _cuda_sync()
        t_ode = time.time()
        result = _run_guided(
            model, batch, scene_idx=scene_idx, guided_kwargs=guided_kwargs,
        )
        _cuda_sync()
        info["ode_inference_s"] = float(time.time() - t_ode)
        _log.info(
            f"[infer] ODE inference finished in {info['ode_inference_s']:.1f}s "
            f"(after monodepth/normal priors, before meshing)."
        )
        # Mesh extraction always uses the full Gaussian set; the opacity cull
        # (filtered_centers_opacity_threshold) only trims the saved
        # point_cloud_{normals,rgb}.ply, never mesh.ply.
        info.update(_save_guided_outputs(
            scene_dir, batch=batch, scene_idx=scene_idx, guided_result=result,
            mesh_cfg=cfg.mesh, ply_cfg=cfg.ply, device=device,
            filtered_centers_opacity_threshold=cfg.get(
                "filtered_centers_opacity_threshold", None),
            texture_cfg=cfg.get("texture", None),
        ))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    summary = {
        "mode": mode,
        "ckpt": (str(cfg.ckpt) if cfg.get("ckpt") is not None else None),
        "source": OmegaConf.to_container(cfg.source, resolve=True),
        "selected_images": [str(p) for p in selected],
        "seed": seed,
        "scene": info,
        "host": socket.gethostname(),
        "output_dir": str(out_root),
    }
    summary_path = scene_dir / "_infer_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _log.info(f"[infer] done; outputs in {scene_dir}.")


if __name__ == "__main__":
    main()

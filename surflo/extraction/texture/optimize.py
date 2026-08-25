"""High-level mesh texturing orchestration (optional).

Two coexisting paths on top of the low-level texture primitives
(:mod:`surflo.extraction.texture.vertex_colors`,
:mod:`surflo.extraction.texture.uv_atlas`,
:mod:`surflo.extraction.texture.texture_map`):

  * :func:`optimize_mesh_vertex_colors_and_save` -- per-vertex RGB
    refinement, written as a PLY.
  * :func:`optimize_mesh_texture_and_save` -- xatlas UV unwrap + 2D
    texture optimisation, written as a GLB.

In both modes the seed colors come from per-vertex TSDF fusion of the input
RGB using the MESH'S OWN rasterised depth
(:func:`evaluate_mesh_colors_all_vertices`). External depths (VGGT, DA3, ...)
are deliberately NOT used: they carry geometric drift relative to the
extracted mesh and would silently invalidate the per-pixel TSDF check.

All soft failures (empty mesh, camera/image mismatch, missing optional deps
like ``xatlas`` / ``nvdiffrast``) log a warning and return
``{"texture_enabled": False, "texture_error": ...}`` rather than raising, so a
batch inference loop keeps going.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from surflo.utils.io import write_mesh_ply

_log = logging.getLogger(__name__)


def _parse_init_color(spec: Any, n_verts: int, device: torch.device) -> torch.Tensor:
    """Build an ``(n_verts, 3)`` float tensor of initial vertex colors.

    ``spec`` accepts ``"gray"`` (0.5, default), ``"white"`` (1.0),
    ``"black"`` (0.0), or any 3-iterable RGB triple in ``[0, 1]``.
    """
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "white":
            return torch.full((n_verts, 3), 1.0, device=device, dtype=torch.float32)
        if s == "black":
            return torch.zeros((n_verts, 3), device=device, dtype=torch.float32)
        if s in ("gray", "grey"):
            return torch.full((n_verts, 3), 0.5, device=device, dtype=torch.float32)
    else:
        try:
            arr = np.asarray(list(spec), dtype=np.float32).reshape(3)
            arr = np.clip(arr, 0.0, 1.0)
            base = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
            return base.unsqueeze(0).expand(n_verts, 3).contiguous()
        except Exception:
            pass
    _log.info(f"[texture] Unrecognised init_color={spec!r}; falling back to gray 0.5.")
    return torch.full((n_verts, 3), 0.5, device=device, dtype=torch.float32)


@torch.no_grad()
def evaluate_mesh_colors_via_tsdf(
    mesh,
    cameras: List[Any],
    images: torch.Tensor,
    *,
    points: Optional[torch.Tensor] = None,
    trunc_margin: Optional[float] = None,
    use_scalable_renderer: bool = True,
    return_hit_mask: bool = False,
) -> Any:
    """Per-vertex TSDF fusion of the input RGB using the MESH'S OWN
    rasterized depth (one rasterization per view).

    The depth source must be the mesh itself (rasterized through
    :class:`MeshRasterizer` / :class:`MeshRenderer`), not an external
    monocular/stereo prediction, so the TSDF surface aligns with the mesh
    by construction. Frustum-culls faces per view to keep nvdiffrast within
    budget; the per-vertex TSDF book-keeping indexes vertex positions (kept
    global), not faces.
    """
    from surflo.extraction.depth_fusion.tsdf import AdaptiveTSDF
    from surflo.structures.cameras import get_cameras_spatial_extent, is_in_view_frustum
    from surflo.structures.mesh import Meshes as _Meshes
    from surflo.rendering.mesh import MeshRasterizer, MeshRenderer, ScalableMeshRenderer

    device = mesh.verts.device

    pivots = mesh.verts if points is None else points
    pivots = pivots.detach().to(device=device, dtype=torch.float32)

    scene_radius = float(get_cameras_spatial_extent(cameras)["radius"].item())
    if not (scene_radius > 0):
        scene_radius = 1.0
    if trunc_margin is None:
        trunc_margin = 2e-3 * scene_radius

    vol = AdaptiveTSDF(
        points=pivots,
        trunc_margin=float(trunc_margin),
        znear=None,
        zfar=None,
        initial_sdf_value=-1.1,
        use_binary_opacity=False,
    )

    rasterizer = MeshRasterizer(cameras=cameras)
    renderer_cls = ScalableMeshRenderer if use_scalable_renderer else MeshRenderer
    renderer = renderer_cls(rasterizer=rasterizer)

    images_eff = images.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    for cam_id, view in enumerate(cameras):
        faces_mask = is_in_view_frustum(mesh.verts, view)[mesh.faces].any(dim=1)
        culled = _Meshes(verts=mesh.verts, faces=mesh.faces[faces_mask])
        if culled.faces.shape[0] == 0:
            continue
        render_pkg = renderer(
            culled,
            cam_idx=cam_id,
            return_depth=True,
            return_normals=False,
            use_antialiasing=False,
        )
        depth_img = render_pkg["depth"]  # (1, H, W, 1)
        vol.integrate(
            img=images_eff[cam_id],
            depth=depth_img,
            camera=view,
            obs_weight=1.0,
            interpolate_depth=False,
            interpolation_mode="bilinear",
            padding_mode="border",
            align_corners=True,
            weight_by_softmax=False,
            softmax_temperature=1.0,
        )

    field = vol.return_field_values()
    colors = field["colors"]
    if not return_hit_mask:
        return colors
    hit_mask = (field["tsdf"][..., 0] > -1.1)
    return colors, hit_mask


@torch.no_grad()
def evaluate_mesh_colors_all_vertices(
    mesh,
    cameras: List[Any],
    images: torch.Tensor,
    *,
    trunc_margin_first_pass: Optional[float] = None,
    trunc_margin_fallback_factor: float = 1.0,
    use_scalable_renderer: bool = True,
) -> torch.Tensor:
    """Two-pass per-vertex color TSDF fusion.

    First pass uses a tight truncation to recover clearly-visible vertices
    via mesh-rendered depth; the second pass re-runs the fusion on the
    remaining miss vertices with a much larger truncation so occluded /
    back-facing vertices still pick up the average color of the views they
    project into.
    """
    from surflo.structures.cameras import get_cameras_spatial_extent

    device = mesh.verts.device
    n_v = int(mesh.verts.shape[0])

    colors, hit_mask = evaluate_mesh_colors_via_tsdf(
        mesh, cameras, images,
        points=None,
        trunc_margin=trunc_margin_first_pass,
        use_scalable_renderer=use_scalable_renderer,
        return_hit_mask=True,
    )
    colors = colors.to(device=device, dtype=torch.float32)
    hit_mask = hit_mask.to(device=device, dtype=torch.bool)

    n_hit = int(hit_mask.sum().item())
    n_miss = int((~hit_mask).sum().item())
    _log.info(
        f"[texture/tsdf] first pass: {n_hit}/{n_v} vertices observed "
        f"({100.0 * n_hit / max(n_v, 1):.1f}%) at trunc={trunc_margin_first_pass}. "
        f"{n_miss} miss vertices will go through the relaxed second pass."
    )
    if n_miss == 0:
        return colors

    scene_radius = float(get_cameras_spatial_extent(cameras)["radius"].item())
    if not (scene_radius > 0):
        scene_radius = 1.0
    fallback_trunc = float(trunc_margin_fallback_factor) * scene_radius

    miss_idx = (~hit_mask).nonzero(as_tuple=False).view(-1)
    fallback_colors, fallback_hit_mask = evaluate_mesh_colors_via_tsdf(
        mesh, cameras, images,
        points=mesh.verts[miss_idx],
        trunc_margin=fallback_trunc,
        use_scalable_renderer=use_scalable_renderer,
        return_hit_mask=True,
    )
    fallback_colors = fallback_colors.to(device=device, dtype=torch.float32)
    fallback_hit_mask = fallback_hit_mask.to(device=device, dtype=torch.bool)

    n_recovered = int(fallback_hit_mask.sum().item())
    n_still_unobs = int((~fallback_hit_mask).sum().item())
    _log.info(
        f"[texture/tsdf] second pass: recovered {n_recovered}/{n_miss} miss vertices "
        f"at trunc={fallback_trunc} (factor={trunc_margin_fallback_factor}); "
        f"{n_still_unobs} remain unobserved (color stays at 0)."
    )

    colors = colors.clone()
    colors[miss_idx] = fallback_colors
    return colors


@torch.no_grad()
def evaluate_point_colors_via_gaussian_tsdf(
    points: torch.Tensor,
    gaussians: Any,
    cameras: List[Any],
    images: torch.Tensor,
    *,
    trunc_margin_first_pass: Optional[float] = None,
    trunc_margin_fallback_factor: float = 1.0,
) -> torch.Tensor:
    """Per-point TSDF color fusion of the input RGB (point-cloud analogue of
    :func:`evaluate_mesh_colors_all_vertices`).

    Same two-pass scheme; the only difference is the depth source: here we use
    the GAUSSIANS' own RaDe-GS median depth (rendered once per view, reused
    across both passes) instead of mesh-rasterized depth.
    """
    from surflo.extraction.depth_fusion.tsdf import AdaptiveTSDF
    from surflo.structures.cameras import get_cameras_spatial_extent
    from surflo.optimization.gaussian_optimization import render_surflo_default

    device = points.device
    pts = points.detach().to(device=device, dtype=torch.float32)
    n_p = int(pts.shape[0])

    scene_radius = float(get_cameras_spatial_extent(cameras)["radius"].item())
    if not (scene_radius > 0):
        scene_radius = 1.0
    trunc_first = (
        2e-3 * scene_radius if trunc_margin_first_pass is None
        else float(trunc_margin_first_pass)
    )

    images_eff = images.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    bg = torch.zeros(3, device=device, dtype=torch.float32)

    depths = [
        render_surflo_default(
            gaussians=gaussians, viewpoint_camera=view, bg_color=bg,
        )["median_depth"].detach()
        for view in cameras
    ]

    def _fuse(pivot_pts: torch.Tensor, trunc: float):
        vol = AdaptiveTSDF(
            points=pivot_pts,
            trunc_margin=float(trunc),
            znear=None,
            zfar=None,
            initial_sdf_value=-1.1,
            use_binary_opacity=False,
        )
        for cam_id, view in enumerate(cameras):
            vol.integrate(
                img=images_eff[cam_id],
                depth=depths[cam_id],
                camera=view,
                obs_weight=1.0,
                interpolate_depth=False,
                interpolation_mode="bilinear",
                padding_mode="border",
                align_corners=True,
                weight_by_softmax=False,
                softmax_temperature=1.0,
            )
        field = vol.return_field_values()
        return field["colors"], (field["tsdf"][..., 0] > -1.1)

    colors, hit_mask = _fuse(pts, trunc_first)
    colors = colors.to(device=device, dtype=torch.float32)
    hit_mask = hit_mask.to(device=device, dtype=torch.bool)
    n_miss = int((~hit_mask).sum().item())
    _log.info(
        f"[centers-rgb/tsdf] first pass: {n_p - n_miss}/{n_p} points observed "
        f"({100.0 * (n_p - n_miss) / max(n_p, 1):.1f}%) at trunc={trunc_first}."
    )
    if n_miss > 0:
        miss_idx = (~hit_mask).nonzero(as_tuple=False).view(-1)
        fb_trunc = float(trunc_margin_fallback_factor) * scene_radius
        fb_colors, fb_hit = _fuse(pts[miss_idx], fb_trunc)
        colors = colors.clone()
        colors[miss_idx] = fb_colors.to(device=device, dtype=torch.float32)
        _log.info(
            f"[centers-rgb/tsdf] second pass: recovered "
            f"{int(fb_hit.sum().item())}/{n_miss} miss points at trunc={fb_trunc} "
            f"(factor={trunc_margin_fallback_factor})."
        )
    return colors


def optimize_mesh_vertex_colors_and_save(
    out_path: str,
    *,
    mesh,
    cameras: List[Any],
    images: torch.Tensor,
    n_iterations: int = 0,
    learning_rate: float = 0.0025,
    lambda_dssim: float = 0.2,
    use_scalable_renderer: bool = True,
    init_color: Any = "tsdf",
    tsdf_trunc_margin_first_pass_factor: float = 2e-3,
    tsdf_trunc_margin_fallback_factor: float = 1.0,
    binary: bool = True,
) -> Dict[str, Any]:
    """Refine per-vertex colors of ``mesh`` against ``images`` then save as PLY."""
    info: Dict[str, Any] = {}

    n_v = int(mesh.verts.shape[0]) if mesh.verts is not None else 0
    n_f = int(mesh.faces.shape[0]) if mesh.faces is not None else 0
    if n_v == 0 or n_f == 0:
        _log.warning(f"[texture] Mesh is empty (V={n_v}, F={n_f}); skipping.")
        info["texture_enabled"] = False
        info["texture_error"] = "empty_mesh"
        return info

    if len(cameras) == 0:
        _log.warning("[texture] No cameras provided; skipping.")
        info["texture_enabled"] = False
        info["texture_error"] = "no_cameras"
        return info

    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4 or images.shape[1] != 3:
        _log.warning(
            f"[texture] Unexpected images shape {tuple(images.shape)}; "
            f"expected (V, 3, H, W). Skipping."
        )
        info["texture_enabled"] = False
        info["texture_error"] = "bad_images_shape"
        return info

    if int(images.shape[0]) != len(cameras):
        _log.warning(
            f"[texture] images has {int(images.shape[0])} views but cameras has "
            f"{len(cameras)}; skipping."
        )
        info["texture_enabled"] = False
        info["texture_error"] = "view_count_mismatch"
        return info

    cam0 = cameras[0]
    cam_h = int(getattr(cam0, "image_height"))
    cam_w = int(getattr(cam0, "image_width"))
    if (int(images.shape[-2]), int(images.shape[-1])) != (cam_h, cam_w):
        _log.warning(
            f"[texture] images resolution "
            f"{(int(images.shape[-2]), int(images.shape[-1]))} != camera resolution "
            f"{(cam_h, cam_w)}; skipping."
        )
        info["texture_enabled"] = False
        info["texture_error"] = "resolution_mismatch"
        return info

    device = mesh.verts.device
    images = images.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    try:
        from surflo.extraction.texture.vertex_colors import optimize_vertex_colors
        from surflo.structures.mesh import Meshes
    except Exception as e:  # noqa: BLE001
        _log.warning(f"[texture] vertex-color optimiser unavailable ({e}); skipping.")
        info["texture_enabled"] = False
        info["texture_error"] = str(e)
        return info

    init_color_str = init_color.strip().lower() if isinstance(init_color, str) else None
    existing = getattr(mesh, "verts_colors", None)

    if init_color_str == "preserve" and existing is not None and existing.shape == (n_v, 3):
        seed_colors = existing.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        seed_source = "preserve"
    elif init_color_str == "tsdf":
        try:
            from surflo.structures.cameras import get_cameras_spatial_extent
            scene_radius = float(get_cameras_spatial_extent(cameras)["radius"].item())
            if not (scene_radius > 0):
                scene_radius = 1.0
            trunc_first = float(tsdf_trunc_margin_first_pass_factor) * scene_radius
            seed_colors = evaluate_mesh_colors_all_vertices(
                mesh=mesh, cameras=cameras, images=images,
                trunc_margin_first_pass=trunc_first,
                trunc_margin_fallback_factor=float(tsdf_trunc_margin_fallback_factor),
                use_scalable_renderer=bool(use_scalable_renderer),
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        except Exception as e:  # noqa: BLE001
            _log.warning(
                f"[texture] TSDF init via mesh-rendered depth crashed "
                f"({type(e).__name__}: {e}); falling back to gray seed."
            )
            seed_colors = _parse_init_color("gray", n_verts=n_v, device=device)
            seed_source = f"tsdf->fallback(gray; err={type(e).__name__})"
        else:
            seed_source = (
                f"tsdf-mesh-depth(trunc_first={float(tsdf_trunc_margin_first_pass_factor)},"
                f"trunc_fallback={float(tsdf_trunc_margin_fallback_factor)})"
            )
    else:
        seed_colors = _parse_init_color(init_color, n_verts=n_v, device=device)
        seed_source = str(init_color)

    seeded_mesh = Meshes(
        verts=mesh.verts.detach().to(device=device, dtype=torch.float32),
        faces=mesh.faces.detach().to(device=device),
        verts_colors=seed_colors,
    )

    _log.info(
        f"[texture] Optimising vertex colors: V={n_v}, F={n_f}, views={len(cameras)}, "
        f"res={cam_h}x{cam_w}, iters={int(n_iterations)}, lr={float(learning_rate)}, "
        f"lambda_dssim={float(lambda_dssim)}, init={seed_source}."
    )

    t0 = time.time()
    try:
        textured = optimize_vertex_colors(
            mesh=seeded_mesh,
            cameras=cameras,
            images=images,
            num_iterations=int(n_iterations),
            learning_rate=float(learning_rate),
            lambda_dssim=float(lambda_dssim),
            use_scalable_renderer=bool(use_scalable_renderer),
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(
            f"[texture] Optimisation crashed ({type(e).__name__}: {e}); "
            f"skipping textured mesh dump."
        )
        info["texture_enabled"] = False
        info["texture_error"] = f"{type(e).__name__}: {e}"
        return info
    elapsed = time.time() - t0

    cols_uint8 = (
        (textured.verts_colors.detach().cpu().float().clamp(0.0, 1.0) * 255.0 + 0.5)
        .to(torch.uint8).numpy()
    )
    n_v_out, n_f_out = write_mesh_ply(
        out_path, textured.verts, textured.faces,
        vertex_colors_uint8=cols_uint8, binary=bool(binary),
    )
    info["texture_enabled"] = True
    info["texture_mode"] = "vertex_colors"
    info["texture_n_verts"] = int(n_v_out)
    info["texture_n_faces"] = int(n_f_out)
    info["texture_n_iterations"] = int(n_iterations)
    info["texture_init_color"] = seed_source
    info["texture_elapsed_s"] = float(elapsed)
    info["texture_output_path"] = str(out_path)
    _log.info(
        f"[texture] Wrote {out_path}: V={n_v_out}, F={n_f_out} "
        f"(optimisation took {elapsed:.1f}s)."
    )
    return info


def optimize_mesh_texture_and_save(
    out_path: str,
    *,
    mesh,
    cameras: List[Any],
    images: torch.Tensor,
    texture_resolution: int = 2048,
    atlas_padding: int = 2,
    atlas_max_chart_iterations: int = 0,
    dilate_iterations: int = 4,
    n_iterations: int = 0,
    learning_rate: float = 0.01,
    lambda_dssim: float = 0.2,
    init_color: Any = "tsdf",
    tsdf_trunc_margin_first_pass_factor: float = 2e-3,
    tsdf_trunc_margin_fallback_factor: float = 1.0,
    use_scalable_renderer_for_tsdf: bool = True,
    use_mipmaps: bool = True,
    use_antialiasing: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """UV-unwrap a mesh, bake an initial texture, optimize it, save GLB."""
    info: Dict[str, Any] = {}

    n_v = int(mesh.verts.shape[0]) if mesh.verts is not None else 0
    n_f = int(mesh.faces.shape[0]) if mesh.faces is not None else 0
    if n_v == 0 or n_f == 0:
        _log.warning(f"[texture-uv] Mesh is empty (V={n_v}, F={n_f}); skipping.")
        info["texture_enabled"] = False
        info["texture_error"] = "empty_mesh"
        return info

    if len(cameras) == 0:
        _log.warning("[texture-uv] No cameras provided; skipping.")
        info["texture_enabled"] = False
        info["texture_error"] = "no_cameras"
        return info

    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4 or images.shape[1] != 3:
        _log.warning(
            f"[texture-uv] Unexpected images shape {tuple(images.shape)}; "
            f"expected (V, 3, H, W). Skipping."
        )
        info["texture_enabled"] = False
        info["texture_error"] = "bad_images_shape"
        return info

    if int(images.shape[0]) != len(cameras):
        _log.warning(
            f"[texture-uv] images has {int(images.shape[0])} views but cameras has "
            f"{len(cameras)}; skipping."
        )
        info["texture_enabled"] = False
        info["texture_error"] = "view_count_mismatch"
        return info

    cam0 = cameras[0]
    cam_h = int(getattr(cam0, "image_height"))
    cam_w = int(getattr(cam0, "image_width"))
    if (int(images.shape[-2]), int(images.shape[-1])) != (cam_h, cam_w):
        _log.warning(
            f"[texture-uv] images resolution "
            f"{(int(images.shape[-2]), int(images.shape[-1]))} != camera resolution "
            f"{(cam_h, cam_w)}; skipping."
        )
        info["texture_enabled"] = False
        info["texture_error"] = "resolution_mismatch"
        return info

    device = mesh.verts.device
    images = images.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    out_path_p = Path(out_path)
    if out_path_p.suffix.lower() != ".glb":
        _log.warning(
            f"[texture-uv] out_path={out_path_p!r} does not end in .glb; "
            f"file will still be written as GLB."
        )

    try:
        from surflo.extraction.texture.uv_atlas import unwrap_mesh
        from surflo.extraction.texture.texture_map import (
            bake_texture_from_vertex_colors,
            dilate_texture,
            make_glctx,
            optimize_texture,
            save_textured_mesh_as_glb,
        )
        from surflo.structures.mesh import Meshes
    except Exception as e:  # noqa: BLE001
        _log.warning(
            f"[texture-uv] UV-texture stack unavailable ({type(e).__name__}: {e}); "
            f"skipping. (Install xatlas via `pip install xatlas`.)"
        )
        info["texture_enabled"] = False
        info["texture_error"] = f"{type(e).__name__}: {e}"
        return info

    init_color_str = init_color.strip().lower() if isinstance(init_color, str) else None
    existing = getattr(mesh, "verts_colors", None)
    if init_color_str == "preserve" and existing is not None and existing.shape == (n_v, 3):
        seed_colors = existing.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        seed_source = "preserve"
    elif init_color_str == "tsdf":
        try:
            from surflo.structures.cameras import get_cameras_spatial_extent
            scene_radius = float(get_cameras_spatial_extent(cameras)["radius"].item())
            if not (scene_radius > 0):
                scene_radius = 1.0
            trunc_first = float(tsdf_trunc_margin_first_pass_factor) * scene_radius
            seed_colors = evaluate_mesh_colors_all_vertices(
                mesh=mesh, cameras=cameras, images=images,
                trunc_margin_first_pass=trunc_first,
                trunc_margin_fallback_factor=float(tsdf_trunc_margin_fallback_factor),
                use_scalable_renderer=bool(use_scalable_renderer_for_tsdf),
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            seed_source = (
                f"tsdf-mesh-depth(trunc_first={float(tsdf_trunc_margin_first_pass_factor)},"
                f"trunc_fallback={float(tsdf_trunc_margin_fallback_factor)})"
            )
        except Exception as e:  # noqa: BLE001
            _log.warning(
                f"[texture-uv] TSDF init via mesh-rendered depth crashed "
                f"({type(e).__name__}: {e}); falling back to gray seed."
            )
            seed_colors = _parse_init_color("gray", n_verts=n_v, device=device)
            seed_source = f"tsdf->fallback(gray; err={type(e).__name__})"
    else:
        seed_colors = _parse_init_color(init_color, n_verts=n_v, device=device)
        seed_source = str(init_color)

    seeded_mesh = Meshes(
        verts=mesh.verts.detach().to(device=device, dtype=torch.float32),
        faces=mesh.faces.detach().to(device=device),
        verts_colors=seed_colors,
    )

    _log.info(
        f"[texture-uv] V={n_v}, F={n_f}, views={len(cameras)}, res={cam_h}x{cam_w}, "
        f"tex={texture_resolution}^2, iters={int(n_iterations)}, "
        f"lr={float(learning_rate)}, lambda_dssim={float(lambda_dssim)}, init={seed_source}."
    )

    t_unwrap = time.time()
    try:
        atlas = unwrap_mesh(
            seeded_mesh,
            resolution=int(texture_resolution),
            padding=int(atlas_padding),
            max_chart_iterations=int(atlas_max_chart_iterations),
        )
    except ImportError as e:
        _log.warning(
            f"[texture-uv] xatlas not installed ({e}); skipping. "
            f"Install with `pip install xatlas`."
        )
        info["texture_enabled"] = False
        info["texture_error"] = f"xatlas_missing: {e}"
        return info
    except Exception as e:  # noqa: BLE001
        _log.warning(f"[texture-uv] UV unwrap failed ({type(e).__name__}: {e}); skipping.")
        info["texture_enabled"] = False
        info["texture_error"] = f"unwrap_failed: {type(e).__name__}: {e}"
        return info
    elapsed_unwrap = time.time() - t_unwrap

    glctx = make_glctx(use_opengl=True)
    t_bake = time.time()
    try:
        tex_init, tex_mask = bake_texture_from_vertex_colors(
            mesh=atlas.mesh, uvs=atlas.uvs,
            texture_resolution=int(texture_resolution), glctx=glctx,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(
            f"[texture-uv] Bake failed ({type(e).__name__}: {e}); "
            f"falling back to a constant-gray texture init."
        )
        tex_init = torch.full(
            (int(texture_resolution), int(texture_resolution), 3),
            0.5, device=device, dtype=torch.float32,
        )
        tex_mask = torch.zeros(
            int(texture_resolution), int(texture_resolution),
            device=device, dtype=torch.float32,
        )
    elapsed_bake = time.time() - t_bake

    t_opt = time.time()
    try:
        tex_final, tex_mask = optimize_texture(
            mesh=atlas.mesh, uvs=atlas.uvs, cameras=cameras, images=images,
            texture_init=tex_init, init_mask=tex_mask,
            n_iterations=int(n_iterations), learning_rate=float(learning_rate),
            lambda_dssim=float(lambda_dssim), use_mipmaps=bool(use_mipmaps),
            use_antialiasing=bool(use_antialiasing), seed=int(seed), glctx=glctx,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(
            f"[texture-uv] Optimisation crashed ({type(e).__name__}: {e}); "
            f"falling back to the unrefined baked texture."
        )
        tex_final = tex_init
    elapsed_opt = time.time() - t_opt

    if int(dilate_iterations) > 0:
        try:
            tex_final, tex_mask = dilate_texture(
                tex_final, tex_mask, n_iterations=int(dilate_iterations),
            )
        except Exception as e:  # noqa: BLE001
            _log.warning(
                f"[texture-uv] Gutter dilation failed ({type(e).__name__}: {e}); "
                f"saving texture without dilation."
            )

    try:
        n_v_out, n_f_out = save_textured_mesh_as_glb(
            str(out_path_p), mesh=atlas.mesh, uvs=atlas.uvs, texture=tex_final,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(
            f"[texture-uv] GLB save failed ({type(e).__name__}: {e}); skipping."
        )
        info["texture_enabled"] = False
        info["texture_error"] = f"glb_save_failed: {type(e).__name__}: {e}"
        return info

    info["texture_enabled"] = True
    info["texture_mode"] = "uv_texture"
    info["texture_resolution"] = int(texture_resolution)
    info["texture_atlas_padding"] = int(atlas_padding)
    info["texture_n_verts"] = int(n_v_out)
    info["texture_n_faces"] = int(n_f_out)
    info["texture_n_seam_dupes"] = int(int(atlas.mesh.verts.shape[0]) - n_v)
    info["texture_n_iterations"] = int(n_iterations)
    info["texture_init_color"] = seed_source
    info["texture_unwrap_elapsed_s"] = float(elapsed_unwrap)
    info["texture_bake_elapsed_s"] = float(elapsed_bake)
    info["texture_optimize_elapsed_s"] = float(elapsed_opt)
    info["texture_output_path"] = str(out_path_p)
    _log.info(
        f"[texture-uv] Wrote {out_path_p}: V'={n_v_out}, F={n_f_out}, "
        f"tex={int(texture_resolution)}^2 (unwrap {elapsed_unwrap:.1f}s, "
        f"bake {elapsed_bake:.1f}s, opt {elapsed_opt:.1f}s)."
    )
    return info

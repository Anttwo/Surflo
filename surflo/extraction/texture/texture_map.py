"""UV-textured mesh refinement: bake + differentiable optimization + GLB export.

Companion to :mod:`surflo.extraction.texture.uv_atlas`. Where
:func:`~surflo.extraction.texture.vertex_colors.optimize_vertex_colors` learns
one RGB triple per mesh vertex, this module learns a full 2D texture map
sampled through per-vertex UVs, decoupling texture quality from vertex density:
1024^2 or 2048^2 texels stay sharp on a coarse mesh.

Three building blocks:

  * :func:`bake_texture_from_vertex_colors` -- one-shot, no optimization:
    rasterize the mesh IN UV SPACE (texel grid as "screen") and bilinearly
    interpolate per-vertex colors into every covered texel. Pairs naturally
    with TSDF-seeded vertex colors produced upstream.

  * :func:`optimize_texture` -- differentiable refinement against posed RGB
    views through nvdiffrast (``dr.rasterize`` + ``dr.interpolate`` +
    ``dr.texture`` with mipmap filtering + ``dr.antialias``), backpropping an
    L1 + (1 - SSIM) loss into the texture tensor. Same loss and LR schedule as
    :func:`optimize_vertex_colors`, so the two paths are interchangeable.

  * :func:`dilate_texture` -- gutter fill, run after optimization. The
    optimizer only touches texels covered by a chart; the gutter keeps its
    baked value, and minified samples blend the two into visible seams. A few
    iterations of max-radius-1 dilation push chart colors outward to fill it.

GLB export goes through :class:`trimesh.visual.TextureVisuals`, which packs the
UVs and a PIL texture image into the GLB binary, so the file is self-contained.
"""

from __future__ import annotations

import logging
from pathlib import Path
from random import Random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr

from surflo.structures.cameras import Camera
from surflo.structures.mesh import Meshes


__all__ = [
    "make_glctx",
    "bake_texture_from_vertex_colors",
    "dilate_texture",
    "optimize_texture",
    "save_textured_mesh_as_glb",
]


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# nvdiffrast context helper.
# ---------------------------------------------------------------------------

def make_glctx(use_opengl: bool = True):
    """Return an nvdiffrast rasterization context.

    Like :class:`surflo.rendering.mesh.MeshRasterizer`, prefers the GL backend
    (faster on Linux/EGL setups) and falls back to CUDA if GL is unavailable.
    One context is allocated per call rather than cached at module level:
    contexts hold GPU resources and texture optimization is short-lived.
    """
    if use_opengl:
        try:
            return dr.RasterizeGLContext()
        except Exception as e:  # noqa: BLE001
            _log.info(
                f"[texture-map] dr.RasterizeGLContext() failed ({type(e).__name__}: {e}); "
                f"falling back to CUDA backend."
            )
    return dr.RasterizeCudaContext()


# ---------------------------------------------------------------------------
# Bake initial texture from per-vertex colors via UV-space rasterization.
# ---------------------------------------------------------------------------

@torch.no_grad()
def bake_texture_from_vertex_colors(
    *,
    mesh: Meshes,
    uvs: torch.Tensor,
    texture_resolution: int = 2048,
    glctx=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bake per-vertex colors into a 2D texture by rasterizing in UV space.

    Treats the texel grid as the rasterization target: each face of
    the mesh is "drawn" at the location of its 3 vertex UVs (scaled
    to NDC). For every texel that lands inside a face, we
    barycentrically interpolate the per-vertex colors and write the
    result. Texels outside any chart are left at zero with a zero
    mask -- :func:`dilate_texture` is the standard follow-up to fill
    that gutter.

    This is what nvdiffrec's ``render/texture.py`` does for its
    "warm start" texture: it lets the subsequent differentiable
    optimization start from a sharp image instead of constant gray,
    which converges in noticeably fewer iterations.

    Args:
        mesh: must carry ``verts_colors`` of shape ``(V', 3)`` in
            ``[0, 1]``. The seam-duplicated mesh returned by
            :func:`surflo.extraction.texture.uv_atlas.unwrap_mesh`
            with a scattered per-vertex color tensor is the intended
            input.
        uvs: ``(V', 2)`` per-vertex UV coordinates in ``[0, 1]``
            (origin bottom-left, xatlas / nvdiffrast convention).
        texture_resolution: side length of the square texture.
        glctx: optional nvdiffrast context (re-use it across baking
            and the subsequent :func:`optimize_texture` call to avoid
            repeated GL allocations).

    Returns:
        ``(texture, mask)`` of shapes ``(H_t, W_t, 3)`` and
        ``(H_t, W_t)`` (both float on the mesh's device). ``mask``
        is 1.0 where the texel was covered by a rasterized face.
    """
    if mesh.verts_colors is None:
        raise ValueError(
            "bake_texture_from_vertex_colors: mesh.verts_colors is None; "
            "cannot bake without a per-vertex color source."
        )
    if int(mesh.verts_colors.shape[0]) != int(mesh.verts.shape[0]):
        raise ValueError(
            f"bake_texture_from_vertex_colors: verts_colors has "
            f"{int(mesh.verts_colors.shape[0])} rows but mesh has "
            f"{int(mesh.verts.shape[0])} vertices."
        )
    if int(uvs.shape[0]) != int(mesh.verts.shape[0]):
        raise ValueError(
            f"bake_texture_from_vertex_colors: uvs has {int(uvs.shape[0])} rows "
            f"but mesh has {int(mesh.verts.shape[0])} vertices."
        )

    device = mesh.verts.device
    H = W = int(texture_resolution)
    if glctx is None:
        glctx = make_glctx()

    # UV[0,1] -> NDC[-1,1]; z=0, w=1 keeps the rasterizer happy without
    # affecting the 2D layout. (1, V', 4) shape required by dr.rasterize.
    uv_ndc = uvs.to(device=device, dtype=torch.float32) * 2.0 - 1.0
    pos = torch.cat(
        [
            uv_ndc,
            torch.zeros(uv_ndc.shape[0], 1, device=device, dtype=torch.float32),
            torch.ones(uv_ndc.shape[0], 1, device=device, dtype=torch.float32),
        ],
        dim=-1,
    )[None]  # (1, V', 4)
    faces = mesh.faces.to(torch.int32)
    rast, _ = dr.rasterize(glctx, pos, faces, resolution=[H, W])

    # Interpolate per-vertex colors at each rasterized fragment.
    colors_img, _ = dr.interpolate(
        mesh.verts_colors.to(device=device, dtype=torch.float32)[None],
        rast,
        faces,
    )  # (1, H, W, 3)
    mask = (rast[..., 3:4] > 0).float()  # (1, H, W, 1); rast[...,3] = tri_id+1

    texture = (colors_img * mask)[0].clamp(0.0, 1.0)  # (H, W, 3)
    mask = mask[0, ..., 0]  # (H, W)

    cov = float(mask.mean().item())
    _log.info(
        f"[texture-map] baked init texture: {H}x{W}, "
        f"chart coverage={100.0 * cov:.1f}%."
    )
    return texture, mask


# ---------------------------------------------------------------------------
# Texture dilation / gutter fill.
# ---------------------------------------------------------------------------

@torch.no_grad()
def dilate_texture(
    texture: torch.Tensor,
    mask: torch.Tensor,
    n_iterations: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propagate chart colors outward into the gutter, in place-like.

    Iteratively averages each invalid texel with the valid 3x3
    neighbours (weighted by their mask). Equivalent to the gutter-fill
    pass nvdiffrec uses before saving. ``n_iterations`` should be at
    least equal to the atlas padding (in texels) for the gutter to
    fully fill, but more is harmless aside from a tiny constant cost.

    Args:
        texture: ``(H, W, 3)``, modified-by-return (not in place).
        mask: ``(H, W)``, 1.0 where the texel is "valid" (covered by
            a chart or already filled by a previous iteration).
        n_iterations: number of dilation passes.

    Returns:
        ``(texture_dilated, mask_dilated)`` of the same shapes.
    """
    if n_iterations <= 0:
        return texture, mask

    H, W = int(texture.shape[0]), int(texture.shape[1])
    tex = texture.permute(2, 0, 1)[None].float()  # (1, 3, H, W)
    msk = mask[None, None].float()                # (1, 1, H, W)

    # Box kernel; we accumulate weighted color and mask, then divide.
    kernel_color = torch.ones(3, 1, 3, 3, device=texture.device, dtype=torch.float32)
    kernel_mask = torch.ones(1, 1, 3, 3, device=texture.device, dtype=torch.float32)

    for _ in range(int(n_iterations)):
        weighted = tex * msk  # (1, 3, H, W)
        num = F.conv2d(weighted, kernel_color, padding=1, groups=3)
        den = F.conv2d(msk, kernel_mask, padding=1)
        # Update only the previously-invalid texels (keep optimized
        # values intact where mask was already 1).
        new_mask = (den > 0).float()
        fill = num / den.clamp_min(1.0)
        tex = torch.where(msk > 0, tex, fill)
        msk = torch.maximum(msk, new_mask)

    return tex[0].permute(1, 2, 0).clamp(0.0, 1.0), msk[0, 0]


# ---------------------------------------------------------------------------
# Differentiable texture optimization.
# ---------------------------------------------------------------------------

def _rasterize_view(
    glctx,
    camera: Camera,
    verts: torch.Tensor,
    faces: torch.Tensor,
):
    """Run nvdiffrast for one camera, returning ``(rast, rast_db, pos)``.

    Same projection convention as
    :func:`surflo.rendering.mesh.nvdiff_rasterization` (homogenize verts then
    ``pos @ camera.full_proj_transform``), but additionally returns ``rast_db``
    and the clip-space positions needed for UV derivatives + antialiasing.
    """
    H = int(camera.image_height)
    W = int(camera.image_width)
    device = verts.device

    pos_homog = torch.cat(
        [verts, torch.ones(verts.shape[0], 1, device=device, dtype=verts.dtype)],
        dim=-1,
    )
    pos = (pos_homog @ camera.full_proj_transform)[None]  # (1, V', 4)
    rast, rast_db = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[H, W])
    return rast, rast_db, pos


def _l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - gt).mean()


def optimize_texture(
    *,
    mesh: Meshes,
    uvs: torch.Tensor,
    cameras: List[Camera],
    images: torch.Tensor,
    texture_init: torch.Tensor,
    init_mask: Optional[torch.Tensor] = None,
    n_iterations: int = 1000,
    learning_rate: float = 0.01,
    lambda_dssim: float = 0.2,
    use_opengl: bool = True,
    use_mipmaps: bool = True,
    use_antialiasing: bool = True,
    seed: int = 0,
    glctx=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimize a 2D texture against posed RGB views.

    Single-mesh, fixed-topology, fixed-camera-pose optimization
    (analogous to :func:`surflo.extraction.texture.vertex_colors.optimize_vertex_colors`
    but for texture maps). Each iteration:

      1. Sample a random view (without replacement within an epoch).
      2. Rasterize the mesh with nvdiffrast at the camera's resolution.
      3. Interpolate UVs at every covered fragment, with derivatives.
      4. Sample the (learnable) texture via :func:`dr.texture` with
         mipmap filtering (so distant geometry uses lower-frequency
         texels and doesn't over-sample the texture).
      5. Antialias the rendered image (propagates gradients across
         silhouette edges).
      6. L1 + (1 - SSIM) loss vs the GT image; Adam step.

    Args:
        mesh: unwrapped mesh (output of :func:`unwrap_mesh.mesh`).
        uvs: ``(V', 2)`` per-vertex UVs in ``[0, 1]``.
        cameras: per-view :class:`Camera` list (any length).
        images: ``(N, 3, H, W)`` GT RGB in ``[0, 1]`` matching the
            cameras' resolution.
        texture_init: ``(H_t, W_t, 3)`` initial texture (e.g. baked by
            :func:`bake_texture_from_vertex_colors`).
        init_mask: optional ``(H_t, W_t)`` validity mask for
            ``texture_init`` -- currently used only for the post-opt
            dilation pass owned by the caller. The optimizer itself
            doesn't constrain updates by mask (gutter texels can drift,
            and that's fine because dilation overwrites them).
        n_iterations: optimization steps. ~1000 is a good default;
            scale with view count for slower convergence.
        learning_rate: Adam LR. ``0.01`` is a reasonable default for
            texel logits in ``[0, 1]`` (10x bigger than the
            vertex-color path because each texel sees far fewer
            gradient updates per epoch than each vertex).
        lambda_dssim: ``(1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)``.
        use_opengl: forwarded to :func:`make_glctx`.
        use_mipmaps: pass ``filter_mode='linear-mipmap-linear'`` to
            :func:`dr.texture`. Required to avoid aliasing on distant
            geometry; disable only for debugging.
        use_antialiasing: wrap the rendered image in :func:`dr.antialias`.
            Disable to debug pure-rasterization issues.
        seed: PRNG seed for view shuffling.
        glctx: optional shared nvdiffrast context.

    Returns:
        ``(texture, mask)`` -- ``texture`` of shape ``(H_t, W_t, 3)``
        in ``[0, 1]``, detached. ``mask`` is the post-optimization
        validity (==``init_mask`` if provided, else all-ones).
    """
    device = mesh.verts.device
    n_v = int(mesh.verts.shape[0])
    n_f = int(mesh.faces.shape[0])
    n_views = int(images.shape[0])
    if n_views != len(cameras):
        raise ValueError(
            f"optimize_texture: images has {n_views} views but cameras has "
            f"{len(cameras)}."
        )
    if uvs.shape != (n_v, 2):
        raise ValueError(
            f"optimize_texture: uvs shape {tuple(uvs.shape)} does not match "
            f"(V'={n_v}, 2)."
        )
    if texture_init.ndim != 3 or texture_init.shape[-1] != 3:
        raise ValueError(
            f"optimize_texture: texture_init shape {tuple(texture_init.shape)} "
            f"must be (H_t, W_t, 3)."
        )

    if glctx is None:
        glctx = make_glctx(use_opengl=use_opengl)

    H_t, W_t = int(texture_init.shape[0]), int(texture_init.shape[1])
    if init_mask is not None and tuple(init_mask.shape) != (H_t, W_t):
        raise ValueError(
            f"optimize_texture: init_mask shape {tuple(init_mask.shape)} "
            f"does not match texture {(H_t, W_t)}."
        )

    # Fused SSIM is only loaded when actually needed -- keeps the import
    # surface aligned with optimize_vertex_colors.
    try:
        from fused_ssim import fused_ssim
    except ImportError as e:
        raise ImportError(
            "optimize_texture requires `fused_ssim`. Install the project's "
            "rendering extras or pip install fused-ssim."
        ) from e

    tex = torch.nn.Parameter(
        texture_init.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    )
    optimizer = torch.optim.Adam([{"params": [tex], "lr": float(learning_rate)}], eps=1e-15)

    verts = mesh.verts.detach().to(device=device, dtype=torch.float32)
    faces = mesh.faces.detach().to(device=device, dtype=torch.int32)
    uvs_dev = uvs.detach().to(device=device, dtype=torch.float32)
    images_dev = images.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    # Pull images into (N, 3, H, W) once.
    if images_dev.ndim != 4 or images_dev.shape[1] != 3:
        raise ValueError(
            f"optimize_texture: images expected (N, 3, H, W), got {tuple(images_dev.shape)}."
        )

    rng = Random(int(seed))
    view_stack: List[int] = []

    ema_loss = 0.0
    log_every = max(1, int(n_iterations) // 50)
    _log.info(
        f"[texture-map] optimizing UV texture: V'={n_v}, F={n_f}, "
        f"views={n_views}, tex={H_t}x{W_t}, iters={int(n_iterations)}, "
        f"lr={float(learning_rate)}, lambda_dssim={float(lambda_dssim)}, "
        f"mipmaps={bool(use_mipmaps)}, antialias={bool(use_antialiasing)}."
    )

    for it in range(int(n_iterations)):
        if not view_stack:
            view_stack = list(range(n_views))
            rng.shuffle(view_stack)
        view_idx = view_stack.pop()
        cam = cameras[view_idx]

        rast, rast_db, pos = _rasterize_view(glctx, cam, verts, faces)

        uv_pix, uv_db = dr.interpolate(
            uvs_dev[None], rast, faces,
            rast_db=rast_db, diff_attrs="all",
        )  # uv_pix: (1, H, W, 2); uv_db: (1, H, W, 4)

        if use_mipmaps:
            rendered = dr.texture(
                tex[None], uv_pix,
                uv_da=uv_db,
                filter_mode="linear-mipmap-linear",
                boundary_mode="clamp",
            )
        else:
            rendered = dr.texture(
                tex[None], uv_pix,
                filter_mode="linear",
                boundary_mode="clamp",
            )

        if use_antialiasing:
            rendered = dr.antialias(rendered, rast, pos, faces)

        rendered_chw = rendered.squeeze(0).permute(2, 0, 1)  # (3, H, W)
        gt = images_dev[view_idx]

        l1 = _l1_loss(rendered_chw, gt)
        ssim_val = fused_ssim(rendered_chw.unsqueeze(0), gt.unsqueeze(0))
        loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_val)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # Texel-space clamp keeps the learnt logits in the legal RGB
        # range so the antialiasing/SSIM pass doesn't have to handle
        # out-of-[0, 1] inputs (and the saved PNG is bit-exact).
        with torch.no_grad():
            tex.clamp_(0.0, 1.0)

        ema_loss = 0.4 * float(loss.detach().item()) + 0.6 * ema_loss
        if (it + 1) % log_every == 0 or it == int(n_iterations) - 1:
            _log.info(
                f"[texture-map] iter {it + 1}/{int(n_iterations)}: loss_ema={ema_loss:.5f}"
            )

    final_tex = tex.detach().clamp(0.0, 1.0)
    final_mask = (
        init_mask.to(device=device, dtype=torch.float32)
        if init_mask is not None
        else torch.ones(H_t, W_t, device=device, dtype=torch.float32)
    )
    return final_tex, final_mask


# ---------------------------------------------------------------------------
# GLB export.
# ---------------------------------------------------------------------------

def save_textured_mesh_as_glb(
    out_path: str,
    *,
    mesh: Meshes,
    uvs: torch.Tensor,
    texture: torch.Tensor,
) -> Tuple[int, int]:
    """Write a self-contained GLB with embedded texture.

    glTF/GLB uses a top-left UV origin while xatlas / nvdiffrast use
    bottom-left, so we flip V on export and leave the texture image
    as-is. This matches the convention used by every GLB viewer
    (Blender, three.js, model-viewer).

    Args:
        out_path: ``.glb`` filename (parents are created).
        mesh: unwrapped mesh.
        uvs: ``(V', 2)`` UVs in ``[0, 1]``, bottom-left origin.
        texture: ``(H_t, W_t, 3)`` final texture in ``[0, 1]``.

    Returns:
        ``(n_verts_written, n_faces_written)``.
    """
    try:
        import trimesh  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - both are project deps already
        raise ImportError(
            "save_textured_mesh_as_glb requires trimesh and Pillow."
        ) from e

    verts_np = mesh.verts.detach().cpu().numpy().astype(np.float64)
    faces_np = mesh.faces.detach().cpu().numpy().astype(np.int64)
    uvs_np = uvs.detach().cpu().numpy().astype(np.float32).copy()
    # glTF V-flip: convert from bottom-left to top-left origin.
    uvs_np[:, 1] = 1.0 - uvs_np[:, 1]

    tex_uint8 = (
        (texture.detach().cpu().float().clamp(0.0, 1.0) * 255.0 + 0.5)
        .to(torch.uint8).numpy()
    )
    pil_img = Image.fromarray(tex_uint8, mode="RGB")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mesh_tm = trimesh.Trimesh(
        vertices=verts_np,
        faces=faces_np,
        visual=trimesh.visual.TextureVisuals(
            uv=uvs_np,
            image=pil_img,
        ),
        process=False,
    )
    mesh_tm.export(str(out))
    return int(verts_np.shape[0]), int(faces_np.shape[0])

"""Generic per-scene Gaussian Splatting optimization (no model, no flow).

A simple Gaussian-Splatting *baseline* that takes:

  * a set of camera poses + RGB images (typically VGGT camera predictions),
  * an initial 3D point cloud (typically subsampled VGGT pointmap),

and optimizes Gaussian parameters (means, log-scales, quaternions, colors,
logit-opacities, optional learned-normal features) with Adam + a rendering
loss for a fixed number of iterations.

Deliberately minimal: no flow-matching or aux model, no camera pose
optimization, no exposure compensation, no scene rescaling, no SH bands beyond
the diffuse DC term, and **no densification or pruning** -- the Gaussian count
is fixed for the whole optimization.

Losses:

  * RGB = ``(1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)``, weighted by
    ``lambda_rgb``.
  * Depth-Normal consistency (``lambda_dn``), from the RaDe-GS median +
    expected depth. Silently disabled if the render output lacks those keys.
  * Learned-normal (``lambda_normal``, OFF by default). Requires a renderer
    that accepts ``colors_precomp``; RaDe-GS does.

Pass ``render_fn`` (type ``RenderFn``) to swap RaDe-GS for another rasterizer
exposing the same output dict; the default is :func:`render_surflo_default`.
A render function MUST populate ``"render"`` ``(3, H, W)``; ``median_depth`` /
``expected_depth`` / ``normal`` / ``mask`` are optional and enable the
corresponding losses when present.
"""

from __future__ import annotations
import logging

import math
from random import randint
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from torch.optim import Adam

from surflo.rendering.gaussians import Gaussians
from surflo.rendering.surflo import render_surflo
from surflo.structures.cameras import (
    Camera,
    get_cameras_from_intrinsics_and_extrinsics,
    get_cameras_spatial_extent,
)
from surflo.utils.geometry import get_knn_index, matrix_to_quaternion

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Render function contract & default RaDe-GS implementation.
# ---------------------------------------------------------------------------

# A render function takes a Gaussians object + a Camera + an optional
# bg_color and returns a dict with at least {"render": (3, H, W)}. It MAY
# also return "median_depth" / "expected_depth" / "normal" / "mask" for
# the depth-normal / mask losses, and MUST accept a ``colors_precomp``
# kwarg if used with the optional normal loss (the user is expected to
# wrap any non-RaDe-GS renderer to adapt its signature).
RenderFn = Callable[..., Dict[str, torch.Tensor]]


def render_surflo_default(
    gaussians: Gaussians,
    viewpoint_camera: Camera,
    bg_color: Optional[torch.Tensor] = None,
    *,
    colors_precomp: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """Default :data:`RenderFn` -- thin wrapper around :func:`render_surflo`.

    Adds:
      * a default black background when ``bg_color`` is None,
      * a guard that ensures the Gaussians instance exposes the
        ``colors_sh`` attribute (RaDe-GS reads it directly, but
        :class:`surflo.rendering.gaussians.Gaussians` does not define
        it, so it is set to ``None`` here when missing).
    """
    if bg_color is None:
        bg_color = torch.zeros(3, device=gaussians.means.device, dtype=gaussians.means.dtype)
    return render_surflo(
        viewpoint_camera,
        gaussians=gaussians,
        bg_color=bg_color,
        kernel_size=0.0,
        scaling_modifier=1.0,
        require_coord=False,
        require_depth=True,
        colors_precomp=colors_precomp,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Loss helpers, re-implemented locally rather than imported from
# surflo.inference so this module stays decoupled from the flow-matching
# machinery and the renderer can be swapped freely.
# ---------------------------------------------------------------------------

def _l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (network_output - gt).abs().mean()


def rgb_loss(
    image: torch.Tensor,
    gt_image: torch.Tensor,
    *,
    lambda_dssim: float = 0.2,
    lambda_rgb: float = 1.0,
) -> torch.Tensor:
    """L1 + DSSIM, scaled by ``lambda_rgb``. Inputs are ``(3, H, W)``."""
    l1 = _l1_loss(image, gt_image)
    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), padding="valid")
    loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_value)
    return loss * lambda_rgb


def _depth_to_normal_with_mask(
    view: Camera,
    depth: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute view-space normals from a single depth map ``(1, H, W)``.

    Same scheme as :func:`surflo.inference.expert.depth_to_normal_with_mask`:
    reproject pixels into view space, take central differences, cross
    product, normalize, pad. ``valid_points`` masks out the borders and
    pixels with non-positive depth.
    """
    Fx = view.image_width / (2.0 * math.tan(view.FoVx / 2.0))
    Fy = view.image_height / (2.0 * math.tan(view.FoVy / 2.0))
    Cx = float(view.image_width - 1) / 2.0
    Cy = float(view.image_height - 1) / 2.0

    W, H = view.image_width, view.image_height
    device = depth.device
    x = (torch.arange(W, device=device, dtype=torch.float32) - Cx) / Fx
    y = (torch.arange(H, device=device, dtype=torch.float32) - Cy) / Fy
    points = torch.cat(
        [depth * x[None, None], depth * y[None, :, None], depth], dim=0,
    )
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normal_map = F.normalize(torch.cross(dy, dx, dim=0), dim=0)
    output = F.pad(normal_map, (1, 1, 1, 1))

    valid_depths = depth > 0
    valid_depths = (
        valid_depths[:, 2:, 1:-1]
        & valid_depths[:, :-2, 1:-1]
        & valid_depths[:, 1:-1, 2:]
        & valid_depths[:, 1:-1, :-2]
        & valid_depths[:, 1:-1, 1:-1]
    )
    valid_points = torch.zeros_like(depth, dtype=torch.bool)
    valid_points[:, 1:-1, 1:-1] = valid_depths
    return output, valid_points


def dn_loss(
    render_pkg: Dict[str, torch.Tensor],
    viewpoint_cam: Camera,
    *,
    reg_depth_ratio: float = 0.6,
    lambda_dn: float = 0.05,
) -> torch.Tensor:
    """Depth-normal consistency: penalize disagreement between rendered
    normals and normals derived from the rendered depth blend.

    Returns a 0-tensor (no gradient) if the render dict is missing any of
    ``median_depth`` / ``expected_depth`` / ``normal`` -- this lets the
    optimizer be used with renderers that don't expose those, with the
    user simply leaving ``lambda_dn=0`` (or relying on the silent skip).
    """
    keys_needed = ("median_depth", "expected_depth", "normal")
    if not all(k in render_pkg and render_pkg[k] is not None for k in keys_needed):
        return torch.zeros((), device=viewpoint_cam.world_view_transform.device)

    depth_blend = torch.where(
        render_pkg["median_depth"] > 0,
        (1.0 - reg_depth_ratio) * render_pkg["expected_depth"]
        + reg_depth_ratio * render_pkg["median_depth"],
        render_pkg["median_depth"],
    )
    depth_normal, valid_points = _depth_to_normal_with_mask(viewpoint_cam, depth_blend)
    err = 1.0 - torch.linalg.vecdot(render_pkg["normal"], depth_normal, dim=0)
    err = torch.where(valid_points.squeeze(0), err, torch.zeros_like(err)).mean()
    return lambda_dn * err


def _convert_features_to_normals(features: torch.Tensor) -> torch.Tensor:
    """Convert per-Gaussian 4D features to a (signed, unit) normal vector.

    Same layout as :func:`surflo.inference.losses.convert_features_to_normals`:
      * ``features[:, :3]`` -- direction (normalized).
      * ``features[:, 3:4]`` -- ``tanh``-bounded sign multiplier.
    """
    direction = F.normalize(features[:, :3], dim=-1)
    sign = torch.tanh(features[:, -1:])
    return direction * sign


def normal_loss(
    *,
    gs: Gaussians,
    normals: torch.Tensor,
    bg_color: torch.Tensor,
    viewpoint_cam: Camera,
    render_fn: RenderFn,
    lambda_normal: float = 0.05,
    depth_ratio_for_alignment: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penalize disagreement between rendered learned-normals and normals
    derived from rendered median depth.

    Disabled by default in :class:`GSOptimConfig` (``use_normal_loss=False``);
    only useful for a follow-up baseline that learns explicit normals.
    """
    pkg = render_fn(gs, viewpoint_cam, bg_color, colors_precomp=normals)
    rendered_normals = pkg["render"]
    if "median_depth" not in pkg:
        # Renderer doesn't expose median depth -- can't compute the
        # alignment target. Return a zero loss but propagate the rendered
        # normals so callers can still log / visualize them.
        zero = torch.zeros((), device=rendered_normals.device)
        return zero, rendered_normals

    median_depth = pkg["median_depth"]
    view_to_world = viewpoint_cam.world_view_transform[:3, :3].permute(-1, -2)

    median_depth_normal, valid = _depth_to_normal_with_mask(viewpoint_cam, median_depth)
    median_depth_normal = (
        median_depth_normal.permute(1, 2, 0) @ view_to_world
    ).permute(2, 0, 1)

    err = 1.0 - (rendered_normals * median_depth_normal).sum(dim=0)
    loss = depth_ratio_for_alignment * torch.where(
        valid.squeeze(0), err, torch.zeros_like(err),
    ).mean()
    return lambda_normal * loss, rendered_normals


# ---------------------------------------------------------------------------
# Learning rate functions.
# ---------------------------------------------------------------------------

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


# ---------------------------------------------------------------------------
# Gaussian initialization from a point cloud.
# ---------------------------------------------------------------------------

@torch.no_grad()
def gaussian_params_from_points(
    points: torch.Tensor,
    *,
    k_neighbors: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Isotropic Gaussian init from local point density.

    Each Gaussian is centred on its input point, scaled **isotropically** by the
    mean distance to its ``k_neighbors`` nearest neighbours, and given identity
    rotation. Returns ``(means, scaling, rotation_quat)``.

    Note this is *not* the anisotropic SVD-of-local-neighbourhood init used by
    :func:`surflo.inference.gaussians.get_gaussian_parameters_from_point_cloud`
    -- the scale is a single value per Gaussian, floored at 1e-6.
    """
    N = int(points.shape[0])
    device = points.device
    if N == 0:
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 4, device=device),
        )
    knn_idx = get_knn_index(points=points, k=k_neighbors, include_self=False)  # (N, K)
    p = points[knn_idx]                                                       # (N, K, 3)
    
    # Compute the mean distance to the K nearest neighbors for scaling
    mean_dist = (p - points.unsqueeze(1)).norm(dim=-1)  # (N, K)
    mean_dist = mean_dist.mean(dim=-1)  # (N,)
    scaling = mean_dist.clamp_min(0.000001)
    scaling = scaling.reshape(N, 1).repeat(1, 3)
    
    # Use identity rotation
    rotation = torch.zeros(N, 4, device=device)
    rotation[:, 0] = 1.0
    
    return points, scaling, rotation


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# Public dataclass with all knobs.
# ---------------------------------------------------------------------------

@dataclass
class GSOptimConfig:
    """All knobs for :func:`optimize_gaussians`. Defaults mirror the
    aux-Gaussian regime used in ``guided_inference`` (with the FM-related
    bits stripped).

    Notes on losses:
      * ``use_rgb_loss`` (RGB L1 + DSSIM): ON by default.
      * ``use_dn_loss`` (depth-normal consistency from RaDe-GS): ON.
      * ``use_normal_loss`` (rendered learned-normals vs depth normals):
        OFF by default. Activates ``learn_normals``.

    Notes on learning rates:
      * Position LR is ``position_lr_init * spatial_lr_scale`` where
        ``spatial_lr_scale`` is the camera-extent radius (matches
        guided_inference). All other LRs are scaled by
        ``aux_lr_multiplier`` (default 5.0).
    """
    # ---- Optimization length -------------------------------------------
    num_iterations: int = 7000
    log_interval: int = 500
    seed: Optional[int] = None  # set to make the per-scene optim deterministic

    # ---- Debug visualization -------------------------------------------
    # When set to a directory path, every ``log_interval`` steps the
    # optimizer dumps a single ``tmp.png`` panel showing the latest
    # rendered RGB / median depth / normal map (the same dict keys
    # produced by RaDe-GS or any compatible renderer). The image
    # is overwritten in place -- no per-step accumulation -- so this is
    # safe to leave on for long runs. ``None`` (default) disables it.
    log_image_dir: Optional[str] = None

    # ---- Camera / view sampling ----------------------------------------
    # If True, every iteration evaluates the loss on a single random view.
    # If False, every iteration evaluates the loss on ALL views (slower
    # per step but more stable). Single-view is the standard 3DGS recipe.
    use_random_view: bool = True

    # ---- RGB loss (L1 + DSSIM) -----------------------------------------
    use_rgb_loss: bool = True
    lambda_rgb: float = 1.0
    lambda_dssim: float = 0.2

    # ---- Depth-Normal consistency loss (RaDe-GS) -----------------------
    use_dn_loss: bool = True
    start_dn_loss_at_step: int = 3500
    lambda_dn: float = 0.05
    dn_reg_depth_ratio: float = 0.6

    # ---- Optional learned-normal loss ----------------------------------
    use_normal_loss: bool = False
    learn_normals: bool = False
    start_normal_loss_at_step: int = int(num_iterations * 2./3.)
    lambda_normal: float = 0.05
    decouple_normals: bool = True
    normal_depth_ratio_for_alignment: float = 0.6

    # ---- Background sampling -------------------------------------------
    use_random_bg: bool = True

    # ---- Gaussian init -------------------------------------------------
    initial_opacity: float = 0.1
    k_neighbors: int = 3
    init_color_default: float = 0.5      # used only when init_colors is None

    # ---- Spherical harmonics (view-dependent appearance) ---------------
    # ``sh_degree`` controls the number of SH bands BEYOND the diffuse DC
    # term. The DC term itself stays in ``aux_colors`` (RGB, in [0, 1])
    # and is converted to SH inside RaDe-GS via ``RGB2SH``.
    #   * 0 -> diffuse only (no view-dependence). Default; cheapest.
    #   * 1 -> +3 non-DC bands  (4 SH coeffs per Gaussian total)
    #   * 2 -> +8 non-DC bands  (9 SH coeffs)
    #   * 3 -> +15 non-DC bands (16 SH coeffs)  -- standard 3DGS recipe.
    # Higher SH bands are typically learned with a much smaller LR than
    # the DC term (here driven by ``feature_dc_lr``) so that they only
    # capture the residual view-dependent appearance once the diffuse
    # color has settled. Default mirrors guided_inference: 1/20 of the
    # DC LR.
    sh_degree: int = 3
    colors_sh_lr: float = 0.0025 / 20.0

    # ---- Adam learning rates -------------------------------------------
    aux_lr_multiplier: float = 1.0
    position_lr_init: float  = 0.00016    # multiplied by spatial_lr_scale
    position_lr_final: float = 0.0000016  # multiplied by spatial_lr_scale
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_dc_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    gaussian_features_lr: float = 0.025  # only used when learn_normals=True


# ---------------------------------------------------------------------------
# Result holder.
# ---------------------------------------------------------------------------

@dataclass
class GSOptimResult:
    """Bundle of artifacts produced by :func:`optimize_gaussians`.

    ``loss_history`` is a per-step total-loss list (or ``None`` when the
    step was not evaluated; in single-view mode each step has exactly one
    entry). Useful for sanity-checking convergence in the dump pipeline.
    """
    gaussians: Gaussians                 # final optimized Gaussians (CUDA, float32)
    cameras: List[Camera]                # cameras used during optimization
    learned_normals: Optional[torch.Tensor]  # (P, 3) world-space, or None
    loss_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Debug visualization (optional, used when cfg.log_image_dir is set).
# ---------------------------------------------------------------------------

@torch.no_grad()
def _save_debug_panel(
    pkg: Dict[str, torch.Tensor],
    *,
    out_dir: str,
    filename: str = "tmp.png",
    render_fn_name: str = "render_fn",
    step_idx: int = 0,
    total_steps: Optional[int] = None,
) -> None:
    """Write a 1xN matplotlib panel with the latest RGB / median-depth /
    normal renders to ``out_dir / filename``, overwriting any existing
    file.

    Robust by design:
      * Lazy-imports matplotlib with the Agg backend so it works in
        headless environments (no DISPLAY required).
      * Plots only the panels for which the corresponding key is
        present in ``pkg`` (``"render"`` / ``"median_depth"`` /
        ``"normal"``); silently skips missing ones.
      * Wraps the I/O in a try/except so a plotting failure never
        crashes the optimization loop -- it just prints a warning.

    Display conventions:
      * RGB:    expects ``(3, H, W)`` in [0, 1]; clipped before display.
      * Depth:  expects ``(1, H, W)`` or ``(H, W)``; rendered with the
                ``magma`` colormap, with vmin/vmax clipped to the valid
                (>0) range to avoid the alpha-cutoff zero-pixels
                dominating the colormap.
      * Normal: expects ``(3, H, W)`` of unit vectors in [-1, 1];
                visualized as ``(n + 1) / 2``, the usual normal-map
                convention.
    """
    import os
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)  # only the very first call sticks
        import matplotlib.pyplot as plt
    except ImportError as exc:
        _log.info(f"[gsopt][debug-viz] matplotlib import failed: {exc}; skipping.")
        return

    try:
        os.makedirs(out_dir, exist_ok=True)

        panels: List[Tuple[str, np.ndarray, Optional[str]]] = []

        if "render" in pkg:
            rgb = pkg["render"].detach().float().clamp(0.0, 1.0).cpu().numpy()
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = np.transpose(rgb, (1, 2, 0))    # (H, W, 3)
            panels.append(("RGB", rgb, None))

        if "median_depth" in pkg and pkg["median_depth"] is not None:
            d = pkg["median_depth"].detach().float().cpu().numpy()
            if d.ndim == 3 and d.shape[0] == 1:
                d = d[0]                              # (H, W)
            panels.append(("Median depth", d, "magma"))

        if "normal" in pkg and pkg["normal"] is not None:
            n = pkg["normal"].detach().float().cpu().numpy()
            if n.ndim == 3 and n.shape[0] == 3:
                n = np.transpose(n, (1, 2, 0))        # (H, W, 3)
            n = np.clip((n + 1.0) * 0.5, 0.0, 1.0)
            panels.append(("Normal", n, None))

        if not panels:
            return  # nothing to render -- silently skip

        n_panels = len(panels)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(4.5 * n_panels, 4.0), squeeze=False,
        )
        for ax, (title, arr, cmap) in zip(axes[0], panels):
            if cmap is not None:
                # vmin/vmax based on the valid (positive) range so the
                # cutoff zero-pixels at the alpha border don't squash
                # the dynamic range of the colormap.
                positive = arr[arr > 0]
                if positive.size > 0:
                    vmin, vmax = float(positive.min()), float(positive.max())
                    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                else:
                    ax.imshow(arr, cmap=cmap)
            else:
                ax.imshow(arr)
            ax.set_title(title)
            ax.set_axis_off()

        if total_steps is not None:
            suptitle = (
                f"render_fn={render_fn_name}    "
                f"step {step_idx + 1}/{total_steps}"
            )
        else:
            suptitle = f"render_fn={render_fn_name}    step {step_idx + 1}"
        fig.suptitle(suptitle)
        fig.tight_layout()

        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 -- never crash optim on viz failure
        _log.info(f"[gsopt][debug-viz] failed to save debug panel: {exc}")


# ---------------------------------------------------------------------------
# The main entry point: optimize Gaussians on a single scene.
# ---------------------------------------------------------------------------

def optimize_gaussians(
    *,
    init_points: torch.Tensor,             # (P, 3) initial Gaussian centers
    extrinsics: torch.Tensor,              # (N, 3, 4) world-to-cam
    intrinsics: torch.Tensor,              # (N, 3, 3)
    rgb_images: torch.Tensor,              # (N, 3, H, W) in [0, 1]
    cfg: Optional[GSOptimConfig] = None,
    init_colors: Optional[torch.Tensor] = None,    # (P, 3) in [0, 1]; defaults to grey
    init_normals: Optional[torch.Tensor] = None,   # (P, 3); used as init for learned normals if available
    render_fn: RenderFn = render_surflo_default,
    render_fn_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    progress_callback: Optional[
        Callable[[int, float, Dict[str, float]], None]
    ] = None,
) -> GSOptimResult:
    """Optimize Gaussians to fit a set of posed RGB images.

    The function only optimizes the Gaussian parameters -- camera poses
    are frozen at the input ``extrinsics`` / ``intrinsics``. The "render
    function" abstraction lets you pass in any rasterizer producing
    ``{"render": (3, H, W), ...}``; the default is RaDe-GS.

    Args:
        init_points: ``(P, 3)`` initial Gaussian centers in world space.
        extrinsics, intrinsics, rgb_images: per-camera poses + GT images.
        cfg: hyperparameters; defaults to :class:`GSOptimConfig`.
        init_colors: optional per-point initial colors in ``[0, 1]``.
            Falls back to ``cfg.init_color_default`` for every point.
        init_normals: optional per-point initial normals (used to seed
            the learnable-normal features when ``cfg.learn_normals``).
        render_fn: rasterizer adapter; see :class:`RenderFn`.
        render_fn_name: human-readable label for ``render_fn`` (e.g.
            ``"surflo"``). Used as the title in the
            optional debug visualization (``cfg.log_image_dir``).
            Falls back to ``render_fn.__name__`` when ``None``.
        device: target CUDA device. Defaults to ``init_points.device``.
        progress_callback: optional ``(step_idx, total_loss, partial_losses)``
            hook called after each Adam step. ``partial_losses`` is a dict
            of the per-loss scalars actually evaluated this step.

    Returns:
        :class:`GSOptimResult` -- final Gaussians + cameras (+ normals,
        if ``learn_normals``).
    """
    if cfg is None:
        cfg = GSOptimConfig()

    if device is None:
        device = init_points.device
    init_points = init_points.to(device=device, dtype=torch.float32)
    extrinsics = extrinsics.to(device=device, dtype=torch.float32)
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    rgb_images = rgb_images.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        # CUDA RNG matters for the random-view / random-bg picks.
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

    # ---- Build cameras ----
    cameras = get_cameras_from_intrinsics_and_extrinsics(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        images=rgb_images,
        data_device=str(device),
    )
    n_cams = len(cameras)
    spatial_lr_scale = float(get_cameras_spatial_extent(cameras=cameras)["radius"])

    # ---- Initialize Gaussian parameters from the input point cloud ----
    init_means, init_scaling, init_rotation = gaussian_params_from_points(
        init_points,
        k_neighbors=int(cfg.k_neighbors),
    )
    P = int(init_means.shape[0])
    if P == 0:
        # Edge case: no surviving init points (e.g. cull was too aggressive).
        # Return an empty Gaussians + matching cameras so the caller can
        # decide what to do (typically: skip this scene).
        empty_gs = Gaussians(
            means=init_means,
            rotations=torch.zeros(0, 4, device=device),
            scales=torch.zeros(0, 3, device=device),
            opacities=torch.zeros(0, device=device),
            colors=torch.zeros(0, 3, device=device),
        )
        return GSOptimResult(gaussians=empty_gs, cameras=cameras, learned_normals=None)

    # Init colors: per-point if provided, otherwise constant grey.
    if init_colors is None:
        colors = torch.full(
            (P, 3), float(cfg.init_color_default),
            device=device, dtype=torch.float32,
        )
    else:
        colors = init_colors.to(device=device, dtype=torch.float32)
        if colors.shape != (P, 3):
            raise ValueError(
                f"init_colors must have shape ({P}, 3); got {tuple(colors.shape)}."
            )

    # Learnable Gaussian parameters. We use the same activations as
    # ``guided_inference``: log-scales -> exp, logit-opacities -> sigmoid,
    # quaternions are normalized at render time.
    aux_xyz = init_means.clone().detach().requires_grad_(True)
    aux_log_scales = (
        init_scaling.clone().detach().log().requires_grad_(True)
    )
    aux_quats = (
        F.normalize(init_rotation, dim=-1).clone().detach().requires_grad_(True)
    )
    aux_colors = colors.clone().detach().requires_grad_(True)
    init_opacities = torch.full(
        (P,), float(cfg.initial_opacity), device=device, dtype=torch.float32,
    )
    aux_logit_opacities = (
        _inverse_sigmoid(init_opacities).clone().detach().requires_grad_(True)
    )

    # Learned-normal features (P, 4): direction (3) + tanh-sign (1).
    aux_normal_features: Optional[torch.Tensor] = None
    if cfg.learn_normals:
        feats = torch.zeros(P, 4, device=device, dtype=torch.float32)
        if init_normals is not None:
            feats[:, :3] = init_normals.to(device=device, dtype=torch.float32)
        else:
            # Random unit directions if no init available, so the SVD-derived
            # init isn't biased toward a degenerate direction. Sign stays at 0.
            feats[:, :3] = F.normalize(
                torch.randn(P, 3, device=device, dtype=torch.float32), dim=-1,
            )
        aux_normal_features = feats.detach().requires_grad_(True)

    # Spherical-harmonic non-DC bands (P, num_sh, 3) where
    # ``num_sh = (sh_degree + 1)^2 - 1``. Init at zero so the model
    # starts as fully diffuse and only learns view-dependent residuals.
    sh_degree = int(cfg.sh_degree)
    if sh_degree < 0 or sh_degree > 3:
        raise ValueError(
            f"sh_degree must be in {{0, 1, 2, 3}}; got {sh_degree}."
        )
    aux_colors_sh: Optional[torch.Tensor] = None
    num_sh_bands = 0
    if sh_degree > 0:
        num_sh_bands = (sh_degree + 1) ** 2 - 1
        aux_colors_sh = torch.zeros(
            P, num_sh_bands, 3, device=device, dtype=torch.float32,
        ).detach().requires_grad_(True)

    # ---- Build optimizer ----
    lr_mul = float(cfg.aux_lr_multiplier)
    groups = [
        {"params": [aux_xyz],
         "lr": cfg.position_lr_init * spatial_lr_scale * lr_mul, "name": "xyz"},
        {"params": [aux_colors],
         "lr": cfg.feature_dc_lr * lr_mul, "name": "f_dc"},
        {"params": [aux_logit_opacities],
         "lr": cfg.opacity_lr * lr_mul, "name": "opacity"},
        {"params": [aux_log_scales],
         "lr": cfg.scaling_lr * lr_mul, "name": "scaling"},
        {"params": [aux_quats],
         "lr": cfg.rotation_lr * lr_mul, "name": "rotation"},
    ]
    if aux_normal_features is not None:
        groups.append({
            "params": [aux_normal_features],
            "lr": cfg.gaussian_features_lr * lr_mul,
            "name": "gaussian_features",
        })
    if aux_colors_sh is not None:
        groups.append({
            "params": [aux_colors_sh],
            "lr": float(cfg.colors_sh_lr) * lr_mul,
            "name": "f_rest",
        })
    optimizer = Adam(groups, lr=0.0, eps=1e-15)
    position_lr_scheduler = get_expon_lr_func(
        lr_init=cfg.position_lr_init * spatial_lr_scale * lr_mul,
        lr_final=cfg.position_lr_final * spatial_lr_scale * lr_mul,
        lr_delay_mult=cfg.position_lr_delay_mult,
        max_steps=cfg.position_lr_max_steps,
    )

    def _materialize_gaussians(detach_for_render: bool = False) -> Gaussians:
        """Build a :class:`Gaussians` from the live aux tensors.

        Detaching the aux tensors is useful for the "decouple normals"
        path so the rendered-normal loss doesn't backprop into the main
        appearance params.

        SH plumbing: when ``cfg.sh_degree > 0`` the live ``aux_colors_sh``
        tensor is attached as ``gs.colors_sh`` and ``gs.active_sh_degree``
        is bumped so RaDe-GS picks up the higher bands -- it reads both
        attributes directly off the Gaussians instance (see
        :func:`surflo.rendering.surflo.render_surflo`).
        """
        means = aux_xyz.detach() if detach_for_render else aux_xyz
        scales = aux_log_scales.exp()
        scales = scales.detach() if detach_for_render else scales
        rotations = F.normalize(aux_quats, dim=-1)
        rotations = rotations.detach() if detach_for_render else rotations
        opacities = aux_logit_opacities.sigmoid()
        opacities = opacities.detach() if detach_for_render else opacities
        colors_t = aux_colors.detach() if detach_for_render else aux_colors
        sh_t = None
        if aux_colors_sh is not None:
            sh_t = aux_colors_sh.detach() if detach_for_render else aux_colors_sh
        return Gaussians(
            means=means,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors_t,
            colors_sh=sh_t,
            active_sh_degree=(sh_degree if sh_t is not None else 0),
        )

    # ---- Optimization loop ----
    loss_history: List[float] = []

    log_freq = max(int(cfg.log_interval), 1)
    
    if cfg.use_random_view:
        cam_indices_stack = []
    else:
        cam_indices_all = list(range(n_cams))

    for step_idx in range(int(cfg.num_iterations)):
        # Update position LR
        for param_group in optimizer.param_groups:
            if param_group["name"] == "xyz":
                updated_lr = position_lr_scheduler(step_idx)
                param_group["lr"] = updated_lr
        
        # Pick view
        if cfg.use_random_view:
            # Rebuild stack if empty
            if not cam_indices_stack:
                cam_indices_stack = list(range(n_cams))

            # Pick random camera index from stack
            _random_view_idx = randint(0, len(cam_indices_stack)-1)
            cam_idx = cam_indices_stack.pop(_random_view_idx)
            
            cam_iter = [int(cam_idx)]
        else:
            cam_iter = cam_indices_all

        optimizer.zero_grad(set_to_none=True)

        # Materialize Gaussians
        gs = _materialize_gaussians(detach_for_render=False)
        gs_nograd = (
            _materialize_gaussians(detach_for_render=True)
            if (cfg.use_normal_loss and cfg.decouple_normals) else None
        )

        # Resolve learned normals *once* per step so we don't re-rotate
        # them per camera (they're world-space, view-independent).
        normals_world: Optional[torch.Tensor] = None
        if cfg.use_normal_loss and aux_normal_features is not None:
            normals_world = _convert_features_to_normals(aux_normal_features)

        total_loss = torch.zeros((), device=device)
        partial_losses: Dict[str, float] = {}

        for cam_idx in cam_iter:
            cam = cameras[cam_idx]
            gt_img = cam.original_image.float()                       # (3, H, W)

            if cfg.use_random_bg:
                bg_color = torch.rand(3, device=device, dtype=torch.float32)
            else:
                bg_color = None

            pkg = render_fn(gs, cam, bg_color)
            rendered = pkg["render"]                                  # (3, H, W)

            if cfg.use_rgb_loss:
                _rgb = rgb_loss(
                    rendered, gt_img,
                    lambda_dssim=cfg.lambda_dssim, lambda_rgb=cfg.lambda_rgb,
                )
                total_loss = total_loss + _rgb
                partial_losses["rgb"] = float(_rgb.detach().item())

            if cfg.use_dn_loss and step_idx >= int(cfg.start_dn_loss_at_step):
                _dn = dn_loss(
                    pkg, cam,
                    reg_depth_ratio=cfg.dn_reg_depth_ratio,
                    lambda_dn=cfg.lambda_dn,
                )
                total_loss = total_loss + _dn
                partial_losses["dn"] = float(_dn.detach().item())

            if (
                cfg.use_normal_loss
                and aux_normal_features is not None
                and normals_world is not None
                and step_idx >= int(cfg.start_normal_loss_at_step)
            ):
                nrm_bg = (
                    torch.zeros(3, device=device, dtype=torch.float32)
                    if bg_color is None else torch.zeros_like(bg_color)
                )
                _nrm, _ = normal_loss(
                    gs=gs_nograd if (gs_nograd is not None) else gs,
                    normals=normals_world,
                    bg_color=nrm_bg,
                    viewpoint_cam=cam,
                    render_fn=render_fn,
                    lambda_normal=cfg.lambda_normal,
                    depth_ratio_for_alignment=cfg.normal_depth_ratio_for_alignment,
                )
                total_loss = total_loss + _nrm
                partial_losses["normal"] = float(_nrm.detach().item())

        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()

        step_total = float(total_loss.detach().item())
        loss_history.append(step_total)

        if progress_callback is not None:
            progress_callback(step_idx, step_total, partial_losses)

        if (step_idx % log_freq) == 0 or step_idx == int(cfg.num_iterations) - 1:
            partials_str = ", ".join(f"{k}={v:.4f}" for k, v in partial_losses.items())
            _log.info(
                f"[gsopt] step {step_idx + 1:>5}/{int(cfg.num_iterations)}  "
                f"loss={step_total:.4f}  ({partials_str})"
            )
            # Optional debug visualization. We use the most recent
            # render package from this step's view (single-view mode)
            # or the last cam in the iteration (all-views mode); either
            # way ``pkg`` reflects the latest render with the up-to-
            # date Gaussian state. The image is overwritten in place
            # each log_interval, so disk cost stays bounded.
            if cfg.log_image_dir is not None:
                _save_debug_panel(
                    pkg,
                    out_dir=str(cfg.log_image_dir),
                    filename="tmp.png",
                    render_fn_name=(
                        render_fn_name
                        if render_fn_name is not None
                        else getattr(render_fn, "__name__", "render_fn")
                    ),
                    step_idx=step_idx,
                    total_steps=int(cfg.num_iterations),
                )

    # ---- Materialize the final Gaussians (detached, no grads) ----
    final_gs = _materialize_gaussians(detach_for_render=True)
    final_normals = (
        _convert_features_to_normals(aux_normal_features.detach())
        if aux_normal_features is not None else None
    )
    return GSOptimResult(
        gaussians=final_gs,
        cameras=cameras,
        learned_normals=final_normals,
        loss_history=loss_history,
    )


# ---------------------------------------------------------------------------
# Convenience: render per-view depth maps from optimized Gaussians.
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_depth_per_view(
    *,
    gaussians: Gaussians,
    cameras: List[Camera],
    render_fn: RenderFn = render_surflo_default,
    depth_kind: str = "median",
) -> torch.Tensor:
    """Render one depth map per camera from an optimized Gaussian set.

    Used by the TSDF output mode of the dump pipeline. Returns a tensor
    of shape ``(N, H, W)`` (one map per camera). The depth kind defaults
    to ``"median"`` because that's what TSDF fusion expects (the median
    depth is robust to the long-tail of expected depth from semi-trans-
    parent Gaussians).
    """
    if depth_kind not in ("median", "expected"):
        raise ValueError(
            f"depth_kind must be 'median' or 'expected'; got {depth_kind!r}."
        )
    key = f"{depth_kind}_depth"
    depth_maps: List[torch.Tensor] = []
    for cam in cameras:
        pkg = render_fn(gaussians, cam, None)
        if key not in pkg:
            raise RuntimeError(
                f"render_fn output is missing '{key}' -- the renderer must "
                f"expose median/expected depths to be used in TSDF mode."
            )
        d = pkg[key]
        if d.ndim == 3 and d.shape[0] == 1:
            d = d.squeeze(0)
        depth_maps.append(d)
    return torch.stack(depth_maps, dim=0)

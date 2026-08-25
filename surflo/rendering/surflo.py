#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""The Surflo renderer: RaDe-GS-style rendering + occupancy integration.

Both entry points go through the single merged ``diff_gaussian_rasterization_surflo``
CUDA extension.

* :func:`render_surflo` renders RGB via SH-on-CUDA and, when ``normals`` is
  given, renders the learned normals in the **same** forward/backward pass as
  channels 3:5 (6-channel path). That is equivalent to two separate passes (one
  RGB, one with ``colors_precomp=normals``) because the per-channel accumulation
  ``C[ch] += feature[ch] * alpha * T`` shares identical geometry and opacity
  weights across channels. With ``normals=None`` it is a plain RGB render.
* :func:`integrate_surflo` wraps the isolated ``dgr_occ`` occupancy pipeline,
  used by the wrapping-mesh extraction path.

Equivalence with the upstream RaDe-GS / occupancy rasterizers is covered by
``submodules/diff-gaussian-rasterization-surflo/tests/test_parity.py``.
"""

import math
from typing import Optional

import torch
from diff_gaussian_rasterization_surflo import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from surflo.structures.cameras import Camera
from surflo.rendering.gaussians import Gaussians
from surflo.rendering.sh_utils import RGB2SH


def render_surflo(
    viewpoint_camera,
    gaussians: Gaussians,
    bg_color: torch.Tensor,
    kernel_size: float = 0.0,
    scaling_modifier: float = 1.0,
    require_coord: bool = False,
    require_depth: bool = True,
    colors_precomp: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
    normal_bg_color: Optional[torch.Tensor] = None,
    stop_normal_geometry_grad: bool = False,
    override_means: Optional[torch.Tensor] = None,
    override_opacity: Optional[torch.Tensor] = None,
    override_scaling: Optional[torch.Tensor] = None,
    override_rotation: Optional[torch.Tensor] = None,
):
    """Render the scene through the merged surflo rasterizer.

    Background tensor (``bg_color``) must be on GPU.

    When ``normals`` (shape ``(N, 3)``) is provided, the learned normals are
    rendered as extra channels in a single 6-channel pass; the returned dict then
    contains ``normal_image`` (the rendered learned normals), while ``render``
    stays the 3-channel RGB image. When ``normals`` is ``None``, no extra
    channels are rendered and ``normal_image`` is absent from the result.

    When ``stop_normal_geometry_grad`` is ``True`` (only meaningful with
    ``normals`` set), the learned-normal channels backpropagate gradient ONLY to
    the ``normals`` input, never to geometry/opacity. Together with detaching the
    depth used as the normal-alignment target, this reproduces the
    ``decouple_normals`` two-pass behavior (RGB from grad-enabled Gaussians,
    normals from detached-geometry Gaussians) in a single fused render.
    """

    means3D = gaussians.means if override_means is None else override_means

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # NOTE(surflo): when rendering normals too, the rasterizer accumulates 6
    # channels, so the background must have 6 entries. The extra 3 entries are
    # the background for the normal channels (default: zeros == black bg, matching
    # a standalone black-background normal render).
    if normals is not None:
        if normal_bg_color is None:
            normal_bg_color = torch.zeros(3, dtype=bg_color.dtype, device=bg_color.device)
        bg = torch.cat([bg_color, normal_bg_color], dim=0)
    else:
        bg = bg_color

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_coord=require_coord,
        require_depth=require_depth,
        debug=False,
        stop_normal_geometry_grad=stop_normal_geometry_grad,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    scales = gaussians.scales if override_scaling is None else override_scaling
    opacity = gaussians.opacities.view(-1, 1) if override_opacity is None else override_opacity.view(-1, 1)
    rotations = gaussians.rotations if override_rotation is None else override_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    if colors_precomp is None:
        # If SHs are provided, use them. Otherwise, just use the colors.
        if gaussians.colors_sh is not None:
            shs_dc = RGB2SH(gaussians.colors).view(-1, 1, 3)  # (N, 1, 3)
            shs_rest = gaussians.colors_sh  # (N, num_sh, 3)
            shs = torch.cat([shs_dc, shs_rest], dim=1)  # (N, num_sh + 1, 3)
        else:
            colors_precomp = gaussians.colors
            shs = None
    else:
        shs = None

    rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        normals=normals,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Split the (possibly 6-channel) color buffer into RGB + learned normals.
    if normals is not None:
        render_rgb = rendered_image[0:3]
        normal_image = rendered_image[3:6]
    else:
        render_rgb = rendered_image
        normal_image = None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "render": render_rgb,
        "mask": rendered_alpha,
        "expected_coord": rendered_expected_coord,
        "median_coord": rendered_median_coord,
        "expected_depth": rendered_expected_depth,
        "median_depth": rendered_median_depth,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "normal": rendered_normal,
    }
    if normal_image is not None:
        out["normal_image"] = normal_image
    return out


def integrate_surflo(
    points3D: torch.Tensor,
    viewpoint_camera: Camera,
    gaussians: Gaussians,
    kernel_size: float = 0.0,
    scaling_modifier: float = 1.0,
):
    """Integrate Gaussians onto the query points (occupancy).

    Uses the isolated ``dgr_occ`` occupancy pipeline. Returns ``alpha_integrated``
    (``1 - transmittance``) and ``inside``.
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=None,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_coord=False,
        require_depth=True,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.means
    opacity = gaussians.opacities.view(-1, 1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    scales = gaussians.scales
    rotations = gaussians.rotations

    depth_plane_precomp = None

    alpha_integrated, inside = rasterizer.integrate(
        points3D=points3D,
        means3D=means3D,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        view2gaussian_precomp=depth_plane_precomp,
    )

    return {"alpha_integrated": alpha_integrated, "inside": inside}

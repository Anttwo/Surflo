"""Render-guidance losses plus the camera-geometry / depth-normal helpers.

Photometric (L1 + D-SSIM), depth, depth-normal, learned-normal, curvature and
depth-order losses used to couple the flowing points, together with the cached
per-camera geometry (:class:`CameraGeometryCache`) and depth->normal helpers
they build on, and a few small math/time helpers.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
from fused_ssim import fused_ssim

from surflo.structures.cameras import Camera


# ---------------------------------------------------------------------------
# Math / activation helpers
# ---------------------------------------------------------------------------
def get_guidance_weight(time: Union[float, torch.Tensor]):
    if isinstance(time, torch.Tensor):
        return (time / 0.9).clamp_max(1.0) ** 5
    return min(1.0, time / 0.9) ** 5


def inverse_sigmoid(x):
    return torch.log(x / (1.0 - x))


def get_biphase_time_grid(
    num_steps_phase_1: int = 100,
    num_steps_phase_2: int = 100,
    phase_switch_frac: float = 0.9,
) -> torch.Tensor:
    return torch.cat([
        torch.linspace(0.0, phase_switch_frac, num_steps_phase_1 + 1),
        torch.linspace(phase_switch_frac, 1.0, num_steps_phase_2 + 1)[1:],
    ])


def convert_features_to_normals(
    features: torch.Tensor,
    normalize: bool = True,
    use_smallest_axis: Optional[bool] = None,
):
    n_gaussian_features = features.shape[1]
    if use_smallest_axis is None:
        assert n_gaussian_features in [1, 4]
        use_smallest_axis = n_gaussian_features == 1
    if use_smallest_axis:
        raise NotImplementedError("Not implemented")
    assert n_gaussian_features == 4

    normal_directions = features[:, :3]
    if normalize:
        normal_directions = torch.nn.functional.normalize(normal_directions, dim=-1)
    normal_signs = torch.tanh(features[:, -1:])
    return normal_directions * normal_signs


# ---------------------------------------------------------------------------
# Camera-geometry cache (per-camera intrinsics-derived rays / view→world)
# ---------------------------------------------------------------------------
class CameraGeometryCache:
    """Pre-compute per-camera tensors that the inner loop reuses on every step.

    For each camera we cache:
      * ``x``, ``y``: 1D tensors of shape ``(W,)`` / ``(H,)`` containing
        ``(arange - cx) / fx`` and ``(arange - cy) / fy``. These drive
        :func:`depth_to_normal_with_mask` (one of the hottest helpers in
        the inner loop).
      * ``fx``, ``fy``, ``cx``, ``cy``, ``H``, ``W``.
      * ``view_to_world``: ``(3, 3)`` rotation used by :func:`normal_loss`.
      * ``ray_grid``: ``(3, H, W)`` cached camera-frame ray directions used
        by :func:`depths_to_points` / :func:`depth_to_normal`.

    Build the cache **after** ``cameras`` has been finalised (i.e. after any
    optional scene rescaling that mutates ``image_width`` / ``image_height``).
    """

    def __init__(self, cameras: List[Camera], device: torch.device):
        self.cameras = cameras
        self.device = device
        self._cache: List[Dict[str, Any]] = []
        for cam in cameras:
            W = int(cam.image_width)
            H = int(cam.image_height)
            fx = W / (2.0 * math.tan(cam.FoVx / 2.0))
            fy = H / (2.0 * math.tan(cam.FoVy / 2.0))
            cx = float(W - 1) / 2.0
            cy = float(H - 1) / 2.0

            x = (torch.arange(W, device=device, dtype=torch.float32) - cx) / fx
            y = (torch.arange(H, device=device, dtype=torch.float32) - cy) / fy

            # Camera-frame ray grid (matches ``depths_to_points``).
            cx_grid = float(W) / 2.0
            cy_grid = float(H) / 2.0
            grid_x, grid_y = torch.meshgrid(
                torch.arange(W, device=device, dtype=torch.float32) + 0.5,
                torch.arange(H, device=device, dtype=torch.float32) + 0.5,
                indexing="xy",
            )
            pts = torch.stack(
                [grid_x, grid_y, torch.ones_like(grid_x)], dim=0,
            ).reshape(3, -1)
            intrins_inv = torch.tensor(
                [
                    [1.0 / fx, 0.0, -cx_grid / fx],
                    [0.0, 1.0 / fy, -cy_grid / fy],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )
            ray_grid = (intrins_inv @ pts).reshape(3, H, W)

            view_to_world = cam.world_view_transform[:3, :3].permute(-1, -2)

            self._cache.append({
                "H": H, "W": W,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "x": x, "y": y,
                "ray_grid": ray_grid,
                "view_to_world": view_to_world,
            })

    def __getitem__(self, ci: int) -> Dict[str, Any]:
        return self._cache[ci]

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Geometry helpers (depth ↔ points ↔ normals) — cached versions
# ---------------------------------------------------------------------------
def depths_to_points_cached(geo: Dict[str, Any], depthmap1: torch.Tensor,
                            depthmap2: Optional[torch.Tensor] = None):
    H, W = geo["H"], geo["W"]
    rays_d = geo["ray_grid"].reshape(3, -1)
    points1 = depthmap1.reshape(1, -1) * rays_d
    if depthmap2 is not None:
        points2 = depthmap2.reshape(1, -1) * rays_d
        return points1.reshape(3, H, W), points2.reshape(3, H, W)
    return points1.reshape(3, H, W)


def point_to_normal(points1, points2=None):
    points = points1[None] if points2 is None else torch.stack([points1, points2], dim=0)
    output = torch.zeros_like(points)
    dx = points[..., 2:, 1:-1] - points[..., :-2, 1:-1]
    dy = points[..., 1:-1, 2:] - points[..., 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    output[..., 1:-1, 1:-1] = normal_map
    return output[0] if points2 is None else output


def depth_to_normal_cached(geo: Dict[str, Any], depth1, depth2=None):
    pts = depths_to_points_cached(geo, depth1, depth2)
    pts = pts[None] if depth2 is None else pts
    return point_to_normal(*pts)


def depth_to_normal_with_mask_cached(
    geo: Dict[str, Any], depth: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """View-space normals from a depth map, using cached per-camera rays.

    Reuses the cached ``x``/``y`` rays instead of allocating fresh
    ``torch.arange``-derived tensors on every call.
    """
    x = geo["x"]                                  # (W,)
    y = geo["y"]                                  # (H,)
    points = torch.cat(
        [depth * x[None, None], depth * y[None, :, None], depth],
        dim=0,
    )
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dy, dx, dim=0), dim=0)
    output = torch.nn.functional.pad(normal_map, (1, 1, 1, 1))

    valid_depths = depth > 0
    valid_depths = (
        valid_depths[:, 2:, 1:-1] & valid_depths[:, :-2, 1:-1]
        & valid_depths[:, 1:-1, 2:] & valid_depths[:, 1:-1, :-2]
        & valid_depths[:, 1:-1, 1:-1]
    )
    valid_points = torch.zeros_like(depth, dtype=torch.bool)
    valid_points[:, 1:-1, 1:-1] = valid_depths
    return output, valid_points


# ---------------------------------------------------------------------------
# Loss helpers (RGB / depth / DN / normal / curvature / depth-order)
# ---------------------------------------------------------------------------
def l1_loss(network_output, gt):
    return torch.abs(network_output - gt).mean()


def rgb_loss(image, gt_image, lambda_dssim=0.2, lambda_rgb=1.0,
             exposure_coeff: Optional[torch.Tensor] = None):
    if exposure_coeff is not None:
        assert exposure_coeff.shape == (2,)
        transformed_image = torch.addcmul(
            exposure_coeff[1], torch.exp(exposure_coeff[0]), image,
        )
        Ll1 = l1_loss(transformed_image, gt_image)
    else:
        Ll1 = l1_loss(image, gt_image)
    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), padding="valid")
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)
    return loss * lambda_rgb


def dn_loss_cached(render_pkg, geo, reg_depth_ratio=0.6, lambda_depth_normal=0.05):
    depth_blend = torch.where(
        render_pkg["median_depth"] > 0,
        (1.0 - reg_depth_ratio) * render_pkg["expected_depth"]
        + reg_depth_ratio * render_pkg["median_depth"],
        render_pkg["median_depth"],
    )
    depth_normal, valid_points = depth_to_normal_with_mask_cached(geo, depth_blend)
    normal_error_map = 1.0 - torch.linalg.vecdot(render_pkg["normal"], depth_normal, dim=0)
    return lambda_depth_normal * torch.where(
        valid_points.squeeze(), normal_error_map, torch.zeros_like(normal_error_map),
    ).mean()


def depth_loss(render_pkg, gt_depth, reg_depth_ratio=0.6, lambda_depth=0.5,
               depth_weight_mask: Optional[torch.Tensor] = None,
               scene_scale: float = 1.0):
    depth_blend = torch.where(
        render_pkg["median_depth"] > 0,
        (1.0 - reg_depth_ratio) * render_pkg["expected_depth"]
        + reg_depth_ratio * render_pkg["median_depth"],
        render_pkg["median_depth"],
    )
    d_loss = (depth_blend - gt_depth).abs() / scene_scale
    if depth_weight_mask is not None:
        d_loss = d_loss * depth_weight_mask
    d_loss = torch.log(1.0 + d_loss)
    return lambda_depth * d_loss.mean()


def normal_alignment_loss_cached(rendered_normals, median_depth, geo,
                                 mask_depth_normal=True,
                                 depth_ratio_for_alignment=0.6):
    """Alignment loss between rendered (learned) normals and the normals derived
    from ``median_depth``.

    Pure computation, so the fused single-pass path in ``guided_inference``
    can use it with a precomputed
    ``rendered_normals`` (channels 3:5 of the 6-channel render) and a
    ``median_depth`` (detached under ``decouple_normals``).
    """
    view_to_world_transform = geo["view_to_world"]

    if mask_depth_normal:
        median_depth_normal, valid_depth_points = depth_to_normal_with_mask_cached(
            geo, median_depth,
        )
    else:
        median_depth_normal = depth_to_normal_cached(geo, median_depth, None)

    median_depth_normal = (
        median_depth_normal.permute(1, 2, 0) @ view_to_world_transform
    ).permute(2, 0, 1)

    if mask_depth_normal:
        normal_error_map = 1.0 - (rendered_normals * median_depth_normal).sum(dim=0)
        normal_field_alignment_loss = depth_ratio_for_alignment * (
            torch.where(
                valid_depth_points.squeeze(),
                normal_error_map,
                torch.zeros_like(normal_error_map),
            ).mean()
        )
    else:
        normal_field_alignment_loss = (
            (1.0 - (rendered_normals * median_depth_normal).sum(dim=0)).mean()
            * depth_ratio_for_alignment
        )

    return normal_field_alignment_loss, rendered_normals


def normal_to_curvature(normal: torch.Tensor) -> torch.Tensor:
    """5-stencil Laplacian of a unit-normal field; used by the curvature loss."""
    n = normal.permute([1, 2, 0])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode="replicate")
    n_c = n[:, 1:-1, 1:-1, :]
    n_u = n[:,  :-2, 1:-1, :] - n_c
    n_l = n[:, 1:-1,  :-2, :] - n_c
    n_b = n[:, 2:,   1:-1, :] - n_c
    n_r = n[:, 1:-1, 2:,   :] - n_c
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1])
    return curv.norm(1, 0, True)


# Module-level pixel-grid cache for compute_depth_order_loss.
_PIXEL_GRID_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}


def pixel_grid_for(height: int, width: int, device: torch.device) -> torch.Tensor:
    key = (int(height), int(width), str(device))
    g = _PIXEL_GRID_CACHE.get(key)
    if g is None:
        g = torch.stack(
            torch.meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.long, device=device),
                torch.linspace(0, width - 1, width, dtype=torch.long, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        _PIXEL_GRID_CACHE[key] = g
    return g


def compute_depth_order_loss(
    depth: torch.Tensor,
    prior_depth: torch.Tensor,
    scene_extent: float = 1.0,
    max_pixel_shift_ratio: float = 0.05,
    normalize_loss: bool = True,
    log_space: bool = True,
    log_scale: float = 20.0,
    reduction: str = "mean",
    debug: bool = False,
):
    height, width = depth.squeeze().shape
    device = depth.device
    pixel_coords = pixel_grid_for(height, width, device)

    max_pixel_shift = max(round(max_pixel_shift_ratio * max(height, width)), 1)
    pixel_shifts = torch.randint(
        -max_pixel_shift, max_pixel_shift + 1, pixel_coords.shape, device=device,
    )
    shifted = (pixel_coords + pixel_shifts)
    shifted[..., 0].clamp_(0, height - 1)
    shifted[..., 1].clamp_(0, width - 1)

    shifted_depth = depth.squeeze()[shifted[:, 0], shifted[:, 1]].reshape(depth.shape)
    shifted_prior_depth = prior_depth.squeeze()[
        shifted[:, 0], shifted[:, 1],
    ].reshape(depth.shape)

    diff = (depth - shifted_depth) / scene_extent
    prior_diff = (prior_depth - shifted_prior_depth) / scene_extent
    if normalize_loss:
        prior_diff = prior_diff / prior_diff.detach().abs().clamp(min=1e-8)
    depth_order_loss = -(diff * prior_diff).clamp(max=0)
    if log_space:
        depth_order_loss = torch.log(1.0 + log_scale * depth_order_loss)

    if reduction == "mean":
        depth_order_loss = depth_order_loss.mean()
    elif reduction == "sum":
        depth_order_loss = depth_order_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    if debug:
        return {
            "depth_order_loss": depth_order_loss,
            "diff": diff,
            "prior_diff": prior_diff,
            "shifted_depth": shifted_depth,
            "shifted_prior_depth": shifted_prior_depth,
        }
    return depth_order_loss




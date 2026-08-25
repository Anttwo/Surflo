"""3D Gaussian container and point-cloud -> Gaussian initialisation.

Turns the independently-flowing points into splattable 3D Gaussians (via a
local-neighbourhood SVD or isotropic init) for the rendering-guidance loop.
Kept free of any CUDA-rasterizer import so it stays cheap to load.
"""
from typing import Optional, Tuple

import numpy as np
import torch

# Canonical container lives in the lower rendering layer; re-exported here so
# `from surflo.inference.gaussians import Gaussians` keeps working.
from surflo.rendering.gaussians import Gaussians  # noqa: F401
from surflo.utils.geometry import (
    get_knn_index,
    matrix_to_quaternion,
    quaternion_multiply,
)


# ---------------------------------------------------------------------------
# Initial-Gaussian helpers (PCA / isotropic init from a point cloud)
# ---------------------------------------------------------------------------
def get_gaussian_parameters_from_point_cloud(
    points: torch.Tensor,
    k_neighbors: int,
    min_scale: float = 1e-6,
    max_scale: Optional[float] = None,
    means: Optional[torch.Tensor] = None,
    knn_idx: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD-from-local-neighborhood Gaussian init."""
    N, _ = points.shape

    if knn_idx is None:
        knn_idx = get_knn_index(points=points, k=k_neighbors, include_self=True)

    p = points[knn_idx]                                   # (N, K, 3)

    if means is None:
        p_bar = p.mean(dim=1, keepdim=True)               # (N, 1, 3)
    else:
        p_bar = means.unsqueeze(1)                        # (N, 1, 3)

    if normals is not None:
        n = normals.unsqueeze(1)
        p = p - (n * (p - p_bar)).sum(dim=-1, keepdim=True) * n

    P = (1.0 / np.sqrt(k_neighbors)) * (p - p_bar)        # (N, K, 3)

    _, S, Vt = torch.linalg.svd(P, full_matrices=False)
    V = Vt.transpose(-1, -2)                              # (N, 3, 3)

    scaling = S.clamp_min(min_scale)
    if max_scale is not None:
        scaling = scaling.clamp_max(max_scale)

    V[..., -1] = torch.linalg.det(V).view(N, 1) * V[..., -1]

    rotation = matrix_to_quaternion(V)
    return p_bar.squeeze(1), scaling, rotation


@torch.no_grad()
def get_isotropic_gaussian_parameters_from_point_cloud(
    points: torch.Tensor,
    k_neighbors: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = int(points.shape[0])
    device = points.device
    if N == 0:
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 4, device=device),
        )
    knn_idx = get_knn_index(points=points, k=k_neighbors, include_self=False)
    p = points[knn_idx]
    mean_dist = (p - points.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
    scaling = mean_dist.clamp_min(1e-6).reshape(N, 1).repeat(1, 3)
    rotation = torch.zeros(N, 4, device=device)
    rotation[:, 0] = 1.0
    return points, scaling, rotation


# ---------------------------------------------------------------------------
# Quaternion / camera-pose helpers
# ---------------------------------------------------------------------------
def quaternion_rotate_vectors(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by quaternion ``q`` (real part first), Rodrigues form."""
    q_w = q[..., 0:1]
    q_xyz = q[..., 1:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def apply_camera_pose_to_gaussians(
    gs: "Gaussians", q_cam: torch.Tensor, t_cam: torch.Tensor,
) -> "Gaussians":
    """Apply a single per-camera (R, t) correction to ``gs`` (renders only).

    Only the pose changes; every appearance attribute -- including
    ``active_sh_degree`` -- is carried over. Dropping the degree here silently
    rendered at the maximum SH band and defeated the warmup schedule.
    """
    q_unit = torch.nn.functional.normalize(q_cam, dim=-1)
    q_broadcast = q_unit.unsqueeze(0).expand(gs.means.shape[0], 4)

    rotated_means = quaternion_rotate_vectors(q_broadcast, gs.means)
    new_means = rotated_means + t_cam
    new_rotations = quaternion_multiply(q_broadcast, gs.rotations)

    return Gaussians(
        means=new_means,
        rotations=new_rotations,
        scales=gs.scales,
        opacities=gs.opacities,
        colors=gs.colors,
        colors_sh=gs.colors_sh,
        active_sh_degree=gs.active_sh_degree,
    )




import torch
from einops import rearrange, einsum
from torch_geometric.nn import knn
from typing import Union, Optional, Tuple
import numpy as np

_pixel_grid_cache: dict[tuple, torch.Tensor] = {}


def _get_pixel_grid(H: int, W: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a cached (1, 3, H*W) homogeneous pixel-center grid."""
    key = (H, W, dtype, device)
    grid = _pixel_grid_cache.get(key)
    if grid is None:
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, dtype=dtype, device=device) + 0.5,
            torch.arange(H, dtype=dtype, device=device) + 0.5,
            indexing="xy",
        )
        grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(1, 3, -1)
        _pixel_grid_cache[key] = grid
    return grid


def inverse_se3(M: torch.Tensor) -> torch.Tensor:
    """
    Inverse of a SE(3) matrix.

    Args:
        M (torch.Tensor): SE(3) matrix of shape (..., 4, 4).

    Returns:
        torch.Tensor: Inverse of the SE(3) matrix of shape (..., 4, 4).
    """
    M_inv = torch.zeros_like(M)  # (..., 4, 4)
    
    R = M[..., :3, :3]  # (..., 3, 3)
    t = M[..., :3, 3]  # (..., 3)
    
    R_inv = R.transpose(-1, -2)  # (..., 3, 3)
    t_inv = - einsum(R_inv, t, "... i j, ... j -> ... i")  # (..., 3)
    
    M_inv[..., :3, :3] = R_inv
    M_inv[..., :3, 3] = t_inv
    M_inv[..., 3, 3] = 1.0
    return M_inv


def depths_to_points_parallel_batched(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    depth: torch.Tensor,
    to_world: bool = True,
) -> torch.Tensor:
    """
    Converts depth maps to point maps, where they are batched.

    Args:
        intrinsics (torch.Tensor): Camera intrinsic matrices of shape (B, N, 3, 3).
        extrinsics (torch.Tensor): Camera extrinsic matrices of shape (B, N, 3, 4).
        depth (torch.Tensor): Depth maps of shape (B, N, H, W, 1).
        to_world (bool): Whether to transform the points to world coordinates.

    Returns:
        torch.Tensor: Point maps of shape (B, N, H, W, 3).
    """
    B, N = intrinsics.shape[:2]
    device = depth.device

    intrinsics = rearrange(intrinsics, "B N D1 D2 -> (B N) D1 D2", D1=3, D2=3)
    extrinsics = rearrange(extrinsics, "B N D1 D2 -> (B N) D1 D2", D1=3, D2=4)
    depth = rearrange(depth, "B N H W 1 -> (B N) H W")

    BN = depth.shape[0]
    H, W = depth.shape[-2:]

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    intrins_inv = torch.zeros_like(intrinsics)  # (BN, 3, 3)
    intrins_inv[:, 0, 0] = 1 / fx
    intrins_inv[:, 1, 1] = 1 / fy
    intrins_inv[:, 0, 2] = -cx / fx
    intrins_inv[:, 1, 2] = -cy / fy
    intrins_inv[:, 2, 2] = 1.0

    points = _get_pixel_grid(H, W, depth.dtype, device)  # (1, 3, H*W)
    rays_d = intrins_inv @ points.expand(BN, -1, -1)  # (BN, 3, H*W)
    points = depth.reshape(BN, 1, -1) * rays_d  # (BN, 3, H*W)

    if to_world:
        R = extrinsics[..., :3, :3]  # (BN, 3, 3)
        T = extrinsics[..., :3, 3:]  # (BN, 3, 1)
        R_cam_to_world = R.transpose(1, 2)  # (BN, 3, 3)
        t_cam_to_world = -torch.bmm(R_cam_to_world, T)  # (BN, 3, 1)
        points = torch.bmm(R_cam_to_world, points) + t_cam_to_world  # (BN, 3, H*W)

    return rearrange(points, "(B N) C (H W) -> B N H W C", B=B, N=N, H=H, W=W)


def depths_to_points_parallel(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    depth: torch.Tensor,
    to_world: bool = True,
) -> torch.Tensor:
    """
    Converts depth maps to point maps.

    Args:
        intrinsics (torch.Tensor): Camera intrinsic matrices of shape (N, 3, 3).
        extrinsics (torch.Tensor): Camera extrinsic matrices of shape (N, 3, 4).
        depth (torch.Tensor): Depth maps of shape (N, H, W) or (N, 1, H, W) or (N, H, W, 1).
        to_world (bool): Whether to transform the points to world coordinates.

    Returns:
        torch.Tensor: Point maps of shape (N, H, W, 3).
    """
    N = depth.shape[0]
    device = depth.device
    H, W = depth.squeeze().shape[-2:]

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    intrins_inv = torch.zeros_like(intrinsics)  # (N, 3, 3)
    intrins_inv[:, 0, 0] = 1 / fx
    intrins_inv[:, 1, 1] = 1 / fy
    intrins_inv[:, 0, 2] = -cx / fx
    intrins_inv[:, 1, 2] = -cy / fy
    intrins_inv[:, 2, 2] = 1.0

    points = _get_pixel_grid(H, W, depth.dtype, device)  # (1, 3, H*W)
    rays_d = intrins_inv @ points.expand(N, -1, -1)  # (N, 3, H*W)
    points = depth.reshape(N, 1, -1) * rays_d  # (N, 3, H*W)

    if to_world:
        R = extrinsics[..., :3, :3]  # (N, 3, 3)
        T = extrinsics[..., :3, 3:]  # (N, 3, 1)
        R_cam_to_world = R.transpose(1, 2)  # (N, 3, 3)
        t_cam_to_world = -torch.bmm(R_cam_to_world, T)  # (N, 3, 1)
        points = torch.bmm(R_cam_to_world, points) + t_cam_to_world  # (N, 3, H*W)

    return points.reshape(N, 3, H, W).permute(0, 2, 3, 1)  # (N, H, W, 3)


def get_knn_index(
    points:torch.Tensor, 
    k:int, 
    points2:Union[torch.Tensor, None]=None, 
    include_self:bool=False,
) -> torch.Tensor:
    """
    Return the k nearest neighbor indices of points in points2.
    If points2 is None, return the k nearest neighbor indices of points in points.
    If include_self is True, return the k nearest neighbor indices of points in points including the point itself.
    If include_self is False, return the k nearest neighbor indices of points in points excluding the point itself.

    Args:
        points (torch.Tensor): (n_pts, d)
        points2 (torch.Tensor, optional): (n_pts2, d). Defaults to None.
        k (int): number of nearest neighbors
        include_self (bool, optional): include the point itself in the nearest neighbors. Defaults to False.
            If points2 is not None, include_self is ignored.

    Returns:
        torch.Tensor: (n_pts, k)
    """
    if points2 is None:
        _k = k if include_self else k + 1
        knn_index = knn(points, points, k=_k)[1].view(len(points), _k)
        if not include_self:
            knn_index = knn_index[:, 1:]
    else:
        knn_index = knn(points2, points, k=k)[1].view(len(points), k)

    return knn_index


def get_point_cloud_main_axes(
    points:torch.Tensor,
    k_neighbors:int,
    knn_idx:Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a set of points, compute the main axes of the point cloud neighborhood.

    Args:
        points (torch.Tensor): 3D points, shape (N, 3)
        k_neighbors (int): number of neighbors to use for each point.
        knn_idx (torch.Tensor, optional): precomputed knn indices. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            singular_values (torch.Tensor): singular values of the covariance matrix, shape (N, 3)
            singular_vectors (torch.Tensor): singular vectors of the covariance matrix, shape (N, 3, 3)
    """
    N, _ = points.shape
    
    # Compute the knn indices if not provided
    if knn_idx is None:
        knn_idx = get_knn_index(points=points, k=k_neighbors, include_self=True)  # (N, K)
    
    # Get local neighborhoods and the barycenters
    p = points[knn_idx]  # (N, K, 3)
    
    # Compute the barycenter of the local neighborhood
    p_bar = p.mean(dim=1, keepdim=True)  # (N, 1, 3)
    
    # Get the local shifts
    P = (1. / np.sqrt(k_neighbors)) * (p - p_bar)  # (N, K, 3)
    
    # Compute the SVD of the local shift matrix
    # This returns the square roots and axes of the covariance matrix Q,
    # with Q = P^T @ P = 1/k * Sum_k (p_k - p_bar) @ (p_k - p_bar)^T
    _, S, Vt = torch.linalg.svd(P, full_matrices=False)
    V = Vt.transpose(-1, -2)  # (N, 3, 3)
    
    return S, V


def get_point_cloud_covariance(
    points:torch.Tensor,
    k_neighbors:int,
    knn_idx:Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a set of points, compute the covariance matrix of the point cloud.

    Args:
        points (torch.Tensor): 3D points, shape (N, 3)
        k_neighbors (int): number of neighbors to use for each point.
        knn_idx (torch.Tensor, optional): precomputed knn indices. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            means (torch.Tensor): mean of the Gaussian, shape (N, 3)
            scaling (torch.Tensor): scaling of the Gaussian, shape (N, 3)
            rotation (torch.Tensor): rotation of the Gaussian, shape (N, 4)
    """
    N, _ = points.shape
    
    # Compute the knn indices if not provided
    if knn_idx is None:
        knn_idx = get_knn_index(points=points, k=k_neighbors, include_self=True)  # (N, K)
    
    # Get local neighborhood
    p = points[knn_idx]  # (N, K, 3)
    
    # Compute the barycenter of the local neighborhood
    p_bar = p.mean(dim=1, keepdim=True)  # (N, 1, 3)
    
    # Get the local shifts
    P = (1. / np.sqrt(k_neighbors)) * (p - p_bar)  # (N, K, 3)
    
    # Compute the covariance matrix C, with 
    # C = P^T @ P 
    #   = 1/k * Sum_k (p_k - p_bar) @ (p_k - p_bar)^T
    C = torch.bmm(P.transpose(-1, -2), P)  # (N, 3, 3)
    
    return C


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


# =============================================================================
# FUNCTIONS BELOW ARE FROM pytorch3d
# =============================================================================

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)

"""Shared utilities for evaluating predicted point clouds against GT.

The metric pipeline lives in three pieces:

1. Closed-form **Umeyama** similarity alignment from camera centers to
   bring the predicted cloud into the GT frame (7-DoF: scale + rigid).
2. **Robust ICP refinement** (``robust_icp_v2``) that locks the scale,
   uses voxel-downsampled clouds, trimmed correspondences, a geometric
   truncation-distance schedule and Huber weighting.
3. Symmetric **Chamfer + F-score** on voxel-downsampled clouds.

Everything stays GPU-friendly via chunked NN search
(``torch_cluster.knn`` through :func:`surflo.utils.geometry.get_knn_index`)
so it can scale to ~1M-point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from surflo.utils.geometry import get_knn_index


# ---------------------------------------------------------------------------
# Umeyama similarity alignment (closed-form, fully batched)
# ---------------------------------------------------------------------------

def umeyama_alignment(
    src: Tensor,           # (N, 3) source points (e.g. predicted camera centers)
    dst: Tensor,           # (N, 3) target points (e.g. GT camera centers)
    estimate_scale: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Closed-form similarity alignment from ``src`` to ``dst`` (Umeyama 1991).

    Returns ``(s, R, t)`` such that ``dst ~= s * (R @ src.T).T + t``.
    Differentiable; runs on whatever device the inputs are on.
    """
    assert src.shape == dst.shape and src.dim() == 2 and src.shape[1] == 3
    n = src.shape[0]
    assert n >= 3, "Need at least 3 correspondences for similarity alignment."

    src_mean = src.mean(dim=0)
    dst_mean = dst.mean(dim=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = (dst_c.T @ src_c) / n
    U, S, Vt = torch.linalg.svd(H)

    d = torch.sign(torch.det(U @ Vt))
    one = torch.ones((), device=src.device, dtype=src.dtype)
    diag_vec = torch.stack([one, one, d])
    D = torch.diag(diag_vec)
    R = U @ D @ Vt

    if estimate_scale:
        var_src = (src_c ** 2).sum() / n
        s = (S * diag_vec).sum() / var_src
    else:
        s = torch.tensor(1.0, device=src.device, dtype=src.dtype)

    t = dst_mean - s * (R @ src_mean)
    return s, R, t


def apply_similarity(points: Tensor, s: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """Apply ``(s, R, t)`` to a ``(..., 3)`` tensor of points."""
    return s * (points @ R.T) + t


# ---------------------------------------------------------------------------
# Camera-center extraction
# ---------------------------------------------------------------------------

def camera_centers_from_extrinsics(extrinsics: Tensor) -> Tensor:
    """Compute world-space camera centers from world-to-camera extrinsics.

    Args:
        extrinsics: ``(N, 3, 4)`` or ``(N, 4, 4)`` world-to-camera matrices
            ``[R | t]``. The camera center in world coords is ``-R^T @ t``.

    Returns:
        ``(N, 3)`` tensor of camera centers.
    """
    assert extrinsics.dim() == 3 and extrinsics.shape[-1] in (4,) and extrinsics.shape[-2] in (3, 4)
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    centers = -torch.einsum("nji,nj->ni", R, t)
    return centers


# ---------------------------------------------------------------------------
# Memory-light nearest-neighbor distances
# ---------------------------------------------------------------------------
#
# We delegate the actual KNN search to :func:`get_knn_index` (a thin wrapper
# around ``torch_cluster``'s native CUDA kernel via
# ``torch_geometric.nn.knn``). That kernel iterates over the reference set
# without ever materializing the full ``(N, M)`` distance matrix, which is
# essential when N, M ~ 1e5+. Same idea as
# :func:`surflo.metrics.chamfer.compute_chamfer_distance`.


@torch.no_grad()
def nn_dists_and_indices(
    queries: Tensor,                # (N, 3)
    refs: Tensor,                   # (M, 3)
) -> tuple[Tensor, Tensor]:
    """Per-query nearest-neighbor distance + index in ``refs``.

    Returns ``(dists, idx)`` with ``dists.shape == idx.shape == (N,)`` and
    ``idx`` indexing into ``refs``. Memory is O(N + M) thanks to
    ``torch_cluster.knn``.
    """
    if queries.numel() == 0 or refs.numel() == 0:
        z = torch.zeros(queries.shape[0], device=queries.device, dtype=queries.dtype)
        zi = torch.zeros(queries.shape[0], device=queries.device, dtype=torch.long)
        return z, zi
    idx = get_knn_index(points=queries, points2=refs, k=1).squeeze(-1)
    dists = (queries - refs[idx]).norm(dim=-1)
    return dists, idx


@torch.no_grad()
def nn_distances(queries: Tensor, refs: Tensor) -> Tensor:
    """Per-query nearest-neighbor L2 distance to ``refs``."""
    dists, _ = nn_dists_and_indices(queries, refs)
    return dists


# ---------------------------------------------------------------------------
# Voxel downsampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def voxel_downsample(points: Tensor, voxel_size: float) -> Tensor:
    """Keep one centroid per occupied voxel. Output is unordered.

    Empty input is forwarded as-is so callers don't need to special-case
    scenes with zero predicted points (``keys.min(dim=0)`` would otherwise
    raise ``IndexError: Expected reduction dim 0 to have non-zero size``).
    """
    if voxel_size <= 0:
        return points
    if points.shape[0] == 0:
        return points
    keys = torch.floor(points / voxel_size).to(torch.int64)
    keys = keys - keys.min(dim=0).values
    base = keys.max(dim=0).values + 1
    flat = keys[:, 0] * (base[1] * base[2]) + keys[:, 1] * base[2] + keys[:, 2]
    uniq, inv = torch.unique(flat, return_inverse=True)
    sums = torch.zeros((uniq.numel(), 3), device=points.device, dtype=points.dtype)
    counts = torch.zeros(uniq.numel(), device=points.device, dtype=points.dtype)
    sums.index_add_(0, inv, points)
    counts.index_add_(0, inv, torch.ones_like(inv, dtype=points.dtype))
    return sums / counts[:, None]


# ---------------------------------------------------------------------------
# Robust ICP v2: scale-locked, voxel-downsampled, trimmed, Huber, scheduled
# ---------------------------------------------------------------------------

@torch.no_grad()
def robust_icp_v2(
    src: Tensor,                    # (N, 3) source cloud (already roughly aligned)
    dst: Tensor,                    # (M, 3) target cloud
    *,
    max_iters: int = 30,
    init_max_dist: Optional[float] = None,   # truncation distance, scene units
    final_max_dist: Optional[float] = None,  # truncation at last iter
    voxel_size: Optional[float] = None,      # downsample both clouds for correspondences
    trim_frac: float = 0.3,                  # drop the worst trim_frac of correspondences
    huber: bool = True,                      # Huber weighting on residuals
    huber_delta_frac: float = 0.5,           # Huber delta = huber_delta_frac * current max_dist
    convergence_eps: float = 1e-7,
    return_history: bool = False,
) -> tuple[Tensor, Tensor]:
    """Robust scale-locked rigid ICP (point-to-point).

    Improvements over the textbook version:

    - **Voxel downsampling** of both clouds before correspondence search to
      limit the influence of dense regions / floaters.
    - **Trimmed correspondences**: every iteration sorts the per-pair
      distances and uses only the lowest ``(1 - trim_frac)`` fraction
      (least-trimmed squares flavor).
    - **Geometric truncation schedule**: ``init_max_dist`` shrinks
      geometrically toward ``final_max_dist`` so early iters are
      forgiving and late iters strict. When ``final_max_dist`` is
      ``None`` it defaults to ``0.2 * init_max_dist``.
    - **Huber weighting** with ``delta = huber_delta_frac * max_dist``.
    - **Scale-locked Procrustes**: only rotation + translation are
      estimated (Umeyama owns the scale).

    Returns ``(R_total, t_total)`` such that
    ``src_aligned = src @ R_total.T + t_total``.
    """
    device, dtype = src.device, src.dtype
    R_total = torch.eye(3, device=device, dtype=dtype)
    t_total = torch.zeros(3, device=device, dtype=dtype)

    # Empty inputs => no refinement possible. Returning identity lets
    # ``align_pred_to_gt`` and the metrics layer carry on (Chamfer / F1
    # against an empty prediction will surface as a separate, more
    # informative number further downstream).
    if src.shape[0] == 0 or dst.shape[0] == 0:
        if return_history:
            return R_total, t_total, []  # type: ignore[return-value]
        return R_total, t_total

    # Default truncation distance: 5% of GT diameter.
    if init_max_dist is None:
        diag = (dst.max(dim=0).values - dst.min(dim=0).values).norm()
        init_max_dist = float(0.05 * diag)
    if final_max_dist is None:
        final_max_dist = 0.2 * init_max_dist

    # Pre-downsample once for correspondence search; we still apply the
    # final transform to the original ``src`` so the caller can index back.
    if voxel_size is not None and voxel_size > 0.0:
        src_ds = voxel_downsample(src, voxel_size)
        dst_ds = voxel_downsample(dst, voxel_size)
    else:
        src_ds, dst_ds = src, dst

    src_curr = src_ds.clone()
    prev_err = float("inf")
    history: list[float] = []

    if max_iters <= 0:
        return R_total, t_total

    for it in range(max_iters):
        # Geometric schedule of the truncation distance.
        if max_iters > 1:
            frac = it / (max_iters - 1)
        else:
            frac = 1.0
        max_dist = init_max_dist * (final_max_dist / init_max_dist) ** frac

        # 1) Correspondences.
        dists, idx = nn_dists_and_indices(src_curr, dst_ds)

        # 2) Truncate by max_dist + trimming.
        valid = dists < max_dist
        if valid.sum() < 10:
            break
        a = src_curr[valid]
        b = dst_ds[idx[valid]]
        r = (a - b).norm(dim=1)

        if trim_frac > 0.0:
            n_keep = max(10, int((1.0 - trim_frac) * r.numel()))
            keep_vals, keep_idx = torch.topk(r, n_keep, largest=False)
            a = a[keep_idx]
            b = b[keep_idx]
            r = keep_vals

        # 3) Huber weighting.
        if huber:
            delta = max(huber_delta_frac * max_dist, 1e-9)
            w = torch.where(
                r < delta,
                torch.ones_like(r),
                delta / r.clamp(min=1e-9),
            )
        else:
            w = torch.ones_like(r)
        w = w / w.sum().clamp(min=1e-9)

        # 4) Weighted Procrustes -> rigid (R_step, t_step).
        a_mean = (w[:, None] * a).sum(dim=0)
        b_mean = (w[:, None] * b).sum(dim=0)
        a_c = a - a_mean
        b_c = b - b_mean
        H = (b_c * w[:, None]).T @ a_c
        U, _, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.det(U @ Vt))
        one = torch.ones((), device=device, dtype=dtype)
        D = torch.diag(torch.stack([one, one, d]))
        R_step = U @ D @ Vt
        t_step = b_mean - R_step @ a_mean

        # 5) Apply step.
        src_curr = src_curr @ R_step.T + t_step
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step

        # 6) Convergence check on weighted RMS residual.
        with torch.no_grad():
            err = (w * ((a @ R_step.T + t_step - b) ** 2).sum(dim=1)).sum().sqrt().item()
        history.append(err)
        if abs(prev_err - err) < convergence_eps:
            break
        prev_err = err

    if return_history:
        return R_total, t_total, history  # type: ignore[return-value]
    return R_total, t_total


# ---------------------------------------------------------------------------
# Metrics: Chamfer (mean / median / trimmed) and F-score
# ---------------------------------------------------------------------------

@dataclass
class SurfaceMetrics:
    chamfer_mean: float
    chamfer_median: float
    chamfer_trimmed: float
    f_score: float
    precision: float
    recall: float
    tau: float


@torch.no_grad()
def chamfer_and_fscore(
    pred: Tensor,                # (N, 3)
    gt: Tensor,                  # (M, 3)
    tau: float,
    trim: float = 0.05,
) -> SurfaceMetrics:
    """Symmetric Chamfer (in distance, not squared) plus F-score@tau.

    Conventions:
      * ``precision`` = fraction of pred points within ``tau`` of gt.
      * ``recall`` = fraction of gt points within ``tau`` of pred.
      * ``f_score`` = 2 * P * R / (P + R).
    """
    d_pred = nn_distances(pred, gt)
    d_gt = nn_distances(gt, pred)

    cd_mean = 0.5 * (d_pred.mean() + d_gt.mean())
    cd_median = 0.5 * (d_pred.median() + d_gt.median())

    def _trim(d: Tensor) -> Tensor:
        k = int((1.0 - trim) * d.numel())
        if k <= 0:
            return d.mean()
        return torch.topk(d, k, largest=False).values.mean()

    cd_trim = 0.5 * (_trim(d_pred) + _trim(d_gt))

    precision = (d_pred < tau).float().mean()
    recall = (d_gt < tau).float().mean()
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-9)

    return SurfaceMetrics(
        chamfer_mean=cd_mean.item(),
        chamfer_median=cd_median.item(),
        chamfer_trimmed=cd_trim.item(),
        f_score=f1.item(),
        precision=precision.item(),
        recall=recall.item(),
        tau=float(tau),
    )


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Output of :func:`align_pred_to_gt`.

    All transforms are expressed so that
    ``pred_final = (s * (pred @ R_um.T) + t_um) @ R_icp.T + t_icp``.
    """
    pred_aligned: Tensor          # (P, 3) prediction after Umeyama (+ optional ICP)
    s: Tensor                     # Umeyama scale
    R_um: Tensor                  # Umeyama rotation
    t_um: Tensor                  # Umeyama translation
    R_icp: Tensor                 # ICP rotation (identity if skipped)
    t_icp: Tensor                 # ICP translation (zero if skipped)


@torch.no_grad()
def align_pred_to_gt(
    pred_points: Tensor,         # (P, 3)
    gt_points: Tensor,           # (M, 3)
    pred_cam_centers: Optional[Tensor],   # (V, 3) or None
    gt_cam_centers: Optional[Tensor],     # (V, 3) or None
    *,
    icp_iters: int = 30,
    icp_trim_frac: float = 0.3,
    icp_init_max_dist_frac: float = 0.05,
    icp_final_max_dist_frac: float = 0.01,
    icp_huber: bool = True,
    voxel_frac: float = 0.001,
) -> AlignmentResult:
    """Two-step prediction-to-GT alignment.

    Step A: Umeyama on camera centers if both ``pred_cam_centers`` and
    ``gt_cam_centers`` are provided; otherwise the closed-form alignment is
    skipped and the pipeline falls back to identity (the caller may want to
    pre-center / pre-scale the prediction in that case).

    Step B: Optional robust ICP refinement (scale-locked) on voxel-downsampled
    clouds. ``icp_iters=0`` skips it.
    """
    device, dtype = pred_points.device, pred_points.dtype
    diag = (gt_points.max(dim=0).values - gt_points.min(dim=0).values).norm().item()

    # ---- Step A: Umeyama on camera centers --------------------------------
    if (
        pred_cam_centers is not None
        and gt_cam_centers is not None
        and pred_cam_centers.shape == gt_cam_centers.shape
        and pred_cam_centers.shape[0] >= 3
    ):
        s, R_um, t_um = umeyama_alignment(
            pred_cam_centers.to(device=device, dtype=dtype),
            gt_cam_centers.to(device=device, dtype=dtype),
            estimate_scale=True,
        )
    else:
        s = torch.tensor(1.0, device=device, dtype=dtype)
        R_um = torch.eye(3, device=device, dtype=dtype)
        t_um = torch.zeros(3, device=device, dtype=dtype)

    pred_after_um = apply_similarity(pred_points, s, R_um, t_um)

    # ---- Step B: Robust ICP -----------------------------------------------
    R_icp = torch.eye(3, device=device, dtype=dtype)
    t_icp = torch.zeros(3, device=device, dtype=dtype)
    if icp_iters > 0:
        R_icp, t_icp = robust_icp_v2(
            src=pred_after_um,
            dst=gt_points,
            max_iters=icp_iters,
            init_max_dist=icp_init_max_dist_frac * diag,
            final_max_dist=icp_final_max_dist_frac * diag,
            voxel_size=voxel_frac * diag if voxel_frac > 0 else None,
            trim_frac=icp_trim_frac,
            huber=icp_huber,
        )
        pred_final = pred_after_um @ R_icp.T + t_icp
    else:
        pred_final = pred_after_um

    return AlignmentResult(
        pred_aligned=pred_final,
        s=s, R_um=R_um, t_um=t_um,
        R_icp=R_icp, t_icp=t_icp,
    )


@torch.no_grad()
def evaluate_alignment(
    pred_points: Tensor,
    gt_points: Tensor,
    *,
    voxel_frac: float = 0.001,
    tau_frac: float = 0.01,
    trim: float = 0.05,
) -> SurfaceMetrics:
    """Voxel-downsample both clouds to a common density, then compute
    Chamfer + F1 on the downsampled clouds.

    Empty input is handled by returning NaN metrics so an offending scene
    doesn't take down a whole eval run. The caller can detect the NaN and
    skip / blacklist the scene.
    """
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        nan = float("nan")
        diag_empty = (
            (gt_points.max(dim=0).values - gt_points.min(dim=0).values).norm().item()
            if gt_points.shape[0] > 0 else 0.0
        )
        return SurfaceMetrics(
            chamfer_mean=nan,
            chamfer_median=nan,
            chamfer_trimmed=nan,
            f_score=nan,
            precision=nan,
            recall=nan,
            tau=tau_frac * diag_empty,
        )
    diag = (gt_points.max(dim=0).values - gt_points.min(dim=0).values).norm().item()
    voxel_size = voxel_frac * diag if voxel_frac > 0 else 0.0
    if voxel_size > 0:
        pred_ds = voxel_downsample(pred_points, voxel_size)
        gt_ds = voxel_downsample(gt_points, voxel_size)
    else:
        pred_ds, gt_ds = pred_points, gt_points
    tau = tau_frac * diag
    return chamfer_and_fscore(pred_ds, gt_ds, tau=tau, trim=trim)

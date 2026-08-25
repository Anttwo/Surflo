"""Convert a guided-inference result dict into Gaussians + refined cameras.

The rendering-guided inference (:mod:`surflo.inference.guided`) returns a
flat result dict (``points``,
``aux_scales``, ``aux_quats``, ``aux_opacities``, ``aux_colors``,
``aux_cam_quats``, ``aux_cam_trans`` ...). Mesh extraction
(:func:`surflo.extraction.occupancy.wrapping.pivot_extraction_with_binary_search`)
consumes a :class:`~surflo.inference.gaussians.Gaussians` set and a list of
:class:`~surflo.structures.cameras.Camera`. These two helpers bridge the two.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from surflo.inference.gaussians import Gaussians
from surflo.structures.cameras import (
    Camera,
    get_cameras_from_intrinsics_and_extrinsics,
)

_log = logging.getLogger(__name__)


def quat_wxyz_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """Convert a ``(4,)`` quaternion in ``wxyz`` (real-first) order to a
    ``(3, 3)`` rotation matrix (input is auto-normalized).

    Hamilton convention, matching
    :func:`surflo.inference.gaussians.quaternion_rotate_vectors` /
    :func:`~surflo.inference.gaussians.apply_camera_pose_to_gaussians`: a
    quaternion ``aux_cam_quats[i]`` produced by the guided inference rotates
    the Gaussian centers via ``X' = R(q) @ X + t``.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = (q / n).tolist()
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def gaussians_from_guided_result(
    guided_result: dict, *, device: torch.device,
) -> Gaussians:
    """Build a :class:`Gaussians` instance from a guided-inference result.

    The returned Gaussians live in original (un-rescaled) world space: the
    guided inference already divides ``aux_scales`` and ``aux_means`` by
    ``scene_scale`` before returning them. SH coefficients are not part of
    the returned dict, so :attr:`Gaussians.colors_sh` is left unset -- that
    is fine for mesh extraction because
    :func:`pivot_extraction_with_binary_search` only calls
    ``integrate_surflo``, whose occupancy test does not read color.

    Means come from ``aux_means`` (the full, never-culled Gaussian centers),
    falling back to ``points`` for older result dicts that predate that key.
    """
    means_src = guided_result.get("aux_means")
    if means_src is None:
        means_src = guided_result["points"]
    means = means_src.to(device).float().reshape(-1, 3)
    scales = guided_result["aux_scales"].to(device).float().reshape(-1, 3)
    quats = guided_result["aux_quats"].to(device).float().reshape(-1, 4)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    opacities = guided_result["aux_opacities"].to(device).float().reshape(-1)
    colors = guided_result["aux_colors"].to(device).float().reshape(-1, 3)
    return Gaussians(
        means=means,
        rotations=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
    )


def build_refined_cameras_from_guided_result(
    batch: dict,
    scene_idx: int,
    guided_result: dict,
    *,
    apply_camera_correction: bool,
    data_device: str,
) -> List[Camera]:
    """Reconstruct the cameras the guided inference effectively rendered against.

    The inference applies the per-camera correction by *transforming the
    Gaussians* (see
    :func:`~surflo.inference.gaussians.apply_camera_pose_to_gaussians`)
    rather than the camera parameters, because the gsplat backward does not
    flow through camera intrinsics/extrinsics. For mesh extraction we want the
    opposite convention -- a single fixed Gaussian set and one corrected
    camera per view -- so we invert the transformation here.

    Rendering ``gs_corrected = R_q @ X + t_q`` with the original camera
    (``R_cw``, ``T_wc``) is identical to rendering the uncorrected,
    world-space ``gs`` with a refined camera::

        R_cw_new = R_q^T @ R_cw
        T_wc_new = R_cw^T @ t_q + T_wc

    where ``R_cw = Camera.R`` (cam-to-world rotation) and ``T_wc = Camera.T``
    (world-to-cam translation). ``aux_cam_trans`` is already divided back by
    ``scene_scale`` so it lives in the same world space as ``points``. When no
    camera correction was learned (or the caller disables it), the original
    cameras are returned unchanged.
    """
    intrinsics = batch["vggt_intrinsics"][scene_idx]
    extrinsics = batch["vggt_extrinsics"][scene_idx]
    scene_imgs = batch.get("rgb_images")
    if scene_imgs is None:
        scene_imgs = batch.get("images")
    images = scene_imgs[scene_idx] if scene_imgs is not None else None
    cameras = get_cameras_from_intrinsics_and_extrinsics(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        images=images,
        data_device=data_device,
    )

    aux_q = guided_result.get("aux_cam_quats")
    aux_t = guided_result.get("aux_cam_trans")
    if not apply_camera_correction or aux_q is None or aux_t is None:
        return cameras

    aux_q_np = aux_q.detach().cpu().float().numpy()
    aux_t_np = aux_t.detach().cpu().float().numpy()
    if aux_q_np.shape[0] != len(cameras):
        _log.warning(
            f"Refined-camera reconstruction: aux_cam_quats has "
            f"{aux_q_np.shape[0]} entries but the scene has {len(cameras)} "
            f"cameras. Falling back to the original (uncorrected) cameras."
        )
        return cameras

    refined: List[Camera] = []
    for i, cam in enumerate(cameras):
        R_cw = np.asarray(cam.R, dtype=np.float64)             # (3, 3)
        T_wc = np.asarray(cam.T, dtype=np.float64).reshape(3)  # (3,)
        R_q = quat_wxyz_to_rotmat_np(aux_q_np[i])              # (3, 3)
        t_q = aux_t_np[i].astype(np.float64).reshape(3)         # (3,)

        R_cw_new = R_q.T @ R_cw
        # R_cw^T == R_wc, so this becomes T_wc + R_wc @ t_q.
        T_wc_new = R_cw.T @ t_q + T_wc

        H = int(cam.image_height)
        W = int(cam.image_width)
        if cam.original_image is not None:
            cam_image = cam.original_image.detach().to(data_device)
        else:
            cam_image = torch.empty((3, H, W), device=data_device, dtype=torch.float32)
        refined.append(
            Camera(
                colmap_id=getattr(cam, "colmap_id", -1),
                R=R_cw_new,
                T=T_wc_new,
                FoVx=float(cam.FoVx),
                FoVy=float(cam.FoVy),
                image=cam_image,
                gt_alpha_mask=None,
                image_name=getattr(cam, "image_name", f"refined_{i}.png"),
                uid=i,
                data_device=data_device,
            )
        )
    return refined

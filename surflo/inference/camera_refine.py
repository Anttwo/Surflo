"""Per-view supervision helpers for the guided inference loop.

The ``build_*`` helpers assemble the per-view VGGT depths and the confidence /
cull masks consumed by the guided loop.
"""
import logging
from typing import List, Optional, Tuple

import torch

from surflo.structures.cameras import Camera, transform_points_world_to_view
from surflo.inference.losses import inverse_sigmoid

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lambda-schedule precomputation
# ---------------------------------------------------------------------------
def precompute_lambda_schedule(
    base_value: float,
    schedule_values: Optional[List[float]],
    schedule_steps: Optional[List[int]],
    total_steps: int,
) -> torch.Tensor:
    """Flatten a ``(steps, values)`` schedule into a per-step lookup table.

    Resolves the value for every step once at startup so the inner loop (run
    once per camera per inner step) only indexes into the result.

    The returned tensor has length ``total_steps + 1`` so callers can
    index by ``min(step_idx, total_steps)`` defensively.
    """
    n = int(total_steps) + 1
    out = torch.full((n,), float(base_value), dtype=torch.float64)
    if schedule_values is not None and schedule_steps is not None:
        for step, value in sorted(zip(schedule_steps, schedule_values)):
            s = int(step)
            if s < n:
                out[s:] = float(value)
    return out


# ---------------------------------------------------------------------------
# Pipeline-side scene preparation (cull masks, depthmaps, hi-res upsample)
# ---------------------------------------------------------------------------
def build_vggt_depths(
    batch: dict,
    scene_idx: int,
    cameras: List[Camera],
    camera_indices: List[int],
    scene_scale: float,
) -> torch.Tensor:
    """Build ``(N, 1, H, W)`` view-space depths from VGGT world points.

    Transforms all selected cameras' world points in a single vectorised pass.
    """
    vggt_wp_scene = batch["vggt_world_points"][scene_idx]   # (Nv, H, W, 3)
    H, W = vggt_wp_scene.shape[1], vggt_wp_scene.shape[2]

    selected_cams = [cameras[ci] for ci in camera_indices]
    selected_wp = vggt_wp_scene[camera_indices]              # (N, H, W, 3)
    if scene_scale != 1.0:
        selected_wp = selected_wp * scene_scale

    out_depths = []
    for cam, wp in zip(selected_cams, selected_wp):
        view_pts = transform_points_world_to_view(
            points=wp.view(1, -1, 3),
            cameras=[cam],
        ).view(H, W, 3)
        out_depths.append(view_pts[..., 2].view(1, 1, H, W))
    return torch.cat(out_depths, dim=0)


def build_confidence_masks(
    batch: dict, scene_idx: int, *,
    confidence_threshold: float,
    use_smooth_confidence_mask: bool,
    smooth_confidence_mask_min_value: float,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Returns ``(conf_mask, inv_conf_mask, binary_conf_mask, inv_binary)``
    or ``None`` if ``vggt_depth_conf`` is missing from the batch."""
    if "vggt_depth_conf" not in batch:
        _log.warning("No depth confidence map found in batch. Disabling confidence.")
        return None
    vggt_depth_conf = batch["vggt_depth_conf"][scene_idx]   # (N, H, W)
    if use_smooth_confidence_mask:
        assert confidence_threshold > 1.0
        assert smooth_confidence_mask_min_value < 0.5
        centered = vggt_depth_conf - confidence_threshold
        centered_min = 1.0 - confidence_threshold
        min_val_t = torch.tensor(smooth_confidence_mask_min_value, device=device)
        rescaled_min = inverse_sigmoid(min_val_t)
        scaling_factor = rescaled_min / centered_min
        rescaled = centered * scaling_factor

        conf_mask = torch.sigmoid(rescaled)
        inv_conf_mask = 1.0 - conf_mask
        binary_conf_mask = conf_mask > 0.5
        inv_binary_conf_mask = ~binary_conf_mask
    else:
        conf_mask = vggt_depth_conf > confidence_threshold
        inv_conf_mask = ~conf_mask
        binary_conf_mask = conf_mask
        inv_binary_conf_mask = inv_conf_mask
    return conf_mask, inv_conf_mask, binary_conf_mask, inv_binary_conf_mask


def build_cull_masks(
    batch: dict, scene_idx: int, *,
    cull_radius: torch.Tensor,
    img_mask_mean: torch.Tensor,
    img_mask_std: torch.Tensor,
    camera_indices: List[int],
    gt_images: List[torch.Tensor],
    vggt_depths: Optional[torch.Tensor],
) -> List[torch.Tensor]:
    """Build per-view cull masks and zero-out culled regions in ``gt_images``
    / ``vggt_depths`` **in place**."""
    img_cull_masks: List[torch.Tensor] = []
    for i_iter, camera_idx in enumerate(camera_indices):
        img_pointmap = batch["vggt_world_points"][scene_idx][camera_idx]
        m = (img_pointmap - img_mask_mean).norm(dim=-1, keepdim=True) > img_mask_std * cull_radius
        m = m.permute(2, 0, 1)
        img_cull_masks.append(m)
        gt_images[i_iter][m.expand_as(gt_images[i_iter])] = 0.0
        if vggt_depths is not None:
            vggt_depths[i_iter][m[0:1]] = 0.0
    return img_cull_masks


def upsample_supervision(
    *,
    guidance_images: torch.Tensor,
    cameras: List[Camera],
    camera_indices: List[int],
    gt_images: List[torch.Tensor],
    vggt_depths: Optional[torch.Tensor],
    img_cull_masks: Optional[List[torch.Tensor]],
    use_confidence: bool,
    use_smooth_confidence_mask: bool,
    conf_quad: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor],
           Optional[List[torch.Tensor]],
           Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Hi-res supervision: override per-camera image dims and upsample the GT
    image / depth / confidence / cull tensors to match."""
    if guidance_images.dim() != 4 or guidance_images.shape[1] != 3:
        raise ValueError(
            f"guidance_images must have shape (N, 3, H, W); got {tuple(guidance_images.shape)}"
        )
    if guidance_images.shape[0] != len(cameras):
        raise ValueError(
            f"guidance_images batch dim ({guidance_images.shape[0]}) must match "
            f"the number of cameras ({len(cameras)})."
        )

    guidance_images = guidance_images.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    H_hi, W_hi = int(guidance_images.shape[2]), int(guidance_images.shape[3])

    H_vggt = int(cameras[0].image_height)
    W_vggt = int(cameras[0].image_width)
    ratio_hi = W_hi / H_hi
    ratio_vggt = W_vggt / H_vggt
    if abs(ratio_hi - ratio_vggt) / max(ratio_vggt, 1e-6) > 0.02:
        _log.warning(
            f"guidance_images aspect ratio {ratio_hi:.4f} differs from "
            f"VGGT input aspect ratio {ratio_vggt:.4f} by more than 2%."
        )

    _log.info(f"guidance_images: rendering at {H_hi}x{W_hi} "
          f"(VGGT inputs at {H_vggt}x{W_vggt}).")

    for ci in range(len(cameras)):
        cameras[ci].original_image = guidance_images[ci]
        cameras[ci].image_height = H_hi
        cameras[ci].image_width = W_hi

    gt_images = [cameras[ci].original_image.float() for ci in camera_indices]

    if vggt_depths is not None:
        vggt_depths = torch.nn.functional.interpolate(
            vggt_depths, size=(H_hi, W_hi), mode="nearest",
        )

    new_conf_quad = conf_quad
    if use_confidence and conf_quad is not None:
        conf_mask, inv_conf_mask, binary_conf_mask, inv_binary_conf_mask = conf_quad
        if use_smooth_confidence_mask:
            conf_mask = torch.nn.functional.interpolate(
                conf_mask.unsqueeze(1), size=(H_hi, W_hi),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            inv_conf_mask = 1.0 - conf_mask
            binary_conf_mask = conf_mask > 0.5
            inv_binary_conf_mask = ~binary_conf_mask
        else:
            conf_mask = torch.nn.functional.interpolate(
                conf_mask.float().unsqueeze(1), size=(H_hi, W_hi), mode="nearest",
            ).squeeze(1).bool()
            inv_conf_mask = ~conf_mask
            binary_conf_mask = conf_mask
            inv_binary_conf_mask = inv_conf_mask
        new_conf_quad = (conf_mask, inv_conf_mask, binary_conf_mask, inv_binary_conf_mask)

    if img_cull_masks is not None:
        img_cull_masks = [
            torch.nn.functional.interpolate(
                m.unsqueeze(0).float(), size=(H_hi, W_hi), mode="nearest",
            ).squeeze(0).bool()
            for m in img_cull_masks
        ]

    return gt_images, vggt_depths, img_cull_masks, new_conf_quad


# ---------------------------------------------------------------------------
# View sampling stack helper (single-view sampling without replacement)
# ---------------------------------------------------------------------------
class ViewSampler:
    """Draws view indices without replacement, refilling on empty.

    """
    def __init__(self):
        self._stack: List[int] = []

    def next(self, n_views: int) -> int:
        from random import randint
        if not self._stack:
            self._stack.extend(range(n_views))
        return self._stack.pop(randint(0, len(self._stack) - 1))


# ---------------------------------------------------------------------------
# GT-image background filler (channel-first, no double permute)
# ---------------------------------------------------------------------------
def fill_gt_with_random_bg(
    gt_img: torch.Tensor, mask: Optional[torch.Tensor],
    bg_color: torch.Tensor,
) -> torch.Tensor:
    """Return a fresh ``(3, H, W)`` tensor with the cull-masked region
    replaced by ``bg_color``."""
    out = gt_img.clone()
    if mask is not None:
        # mask: (1, H, W) bool. Index out at all 3 channels with mask[0].
        out[:, mask[0]] = bg_color[:, None]
    return out


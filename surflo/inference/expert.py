"""Monodepth expert utilities; normals are derived from its depth.

The workflow:

  1. Run a DepthAnything3-style model on each input image at a high
     ``pred_res`` and resize the predictions back to the input image
     resolution in disparity space (more accurate than raw bilinear in
     depth space).
  2. Optionally derive per-view normals from the *high-res* monodepth
     prediction (using high-res cameras), with optional VGGT-depth
     median/scale alignment when reference depths and confidence masks
     are available, and finally bilinearly resize those normals back to
     the input image resolution.

The two outputs are exactly what
:func:`surflo.inference.engine.guided_inference`
expects in its ``monodepths`` / ``normal_guidances`` arguments
(``(N, 1, H, W)`` and ``(N, 3, H, W)`` respectively, both at the
*input* image resolution and ordered to match
``batch["rgb_images"][scene_idx]``).
"""

from __future__ import annotations
import logging

from typing import List, Optional, Tuple

import math
import numpy as np
import torch
from einops import rearrange

# Make ``depth_anything_3`` importable from the in-repo checkout. This runs as
# an import side effect so that merely importing this module is enough; see
# surflo/utils/da3_path.py for where DA3 is looked for.
from surflo.utils.da3_path import ensure_da3_importable

ensure_da3_importable()

from surflo.inference.inference_utils import build_high_res_cameras
from surflo.structures.cameras import (  # noqa: F401  (re-exported for callers)
    Camera,
    get_cameras_from_intrinsics_and_extrinsics,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading + per-image inference
# ---------------------------------------------------------------------------


def _load_da3_model(
    model_id: str,
    device: torch.device,
    *,
    eval_mode: bool = True,
) -> torch.nn.Module:
    """Load a DepthAnything3 model from HuggingFace Hub onto ``device``."""
    # Importing at call time so the script can be parsed even when the DA3
    # package isn't installed (e.g. a CPU-only setup that only runs FFM).
    from depth_anything_3.api import DepthAnything3  # type: ignore

    da3 = DepthAnything3.from_pretrained(model_id).to(device)
    if eval_mode:
        da3.eval()
    return da3


class MonodepthExpert:
    """Lightweight wrapper that loads a DA3 model once and reuses it.

    Use this when running monodepth on multiple scenes back to back:
    reloading ``DepthAnything3.from_pretrained`` per scene costs seconds
    per call and can blow up VRAM.

    :meth:`predict` behaves like the free function
    :func:`call_monodepth_expert` (same ``return_unresized`` semantics).
    """

    def __init__(
        self,
        model_id: str = "depth-anything/da3mono-large",
        device: torch.device | str = "cuda",
        *,
        eval_mode: bool = True,
    ) -> None:
        self.model_id = str(model_id)
        self.device = torch.device(device)
        self.model = _load_da3_model(self.model_id, self.device, eval_mode=eval_mode)

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,  # (N, 3, H, W) in [0, 1]
        *,
        pred_res: int = 1596,
        process_res_method: str = "upper_bound_resize",
        return_unresized: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Run monodepth on a stack of images.

        Returns ``mono_depths_resized`` (back to input resolution via
        disparity-space bilinear), or ``(mono_depths_resized, mono_depths)``
        when ``return_unresized=True`` -- the *raw* (high-res) prediction
        is the one we want for normal extraction.
        """
        device = self.device
        # Preprocess to uint8 NHWC (DA3 inference API).
        images_uint8 = (
            images.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0
        ).astype(np.uint8)
        images_uint8 = rearrange(images_uint8, "n c h w -> n h w c")
        images_list = list(images_uint8)

        mono_depths: List[torch.Tensor] = []
        for i in range(len(images_list)):
            mono_depth = self.model.inference(
                [images_list[i]],
                process_res=pred_res,
                process_res_method=process_res_method,
            )
            # (1, 1, H_pred, W_pred)
            mono_depths.append(
                torch.from_numpy(mono_depth.depth).to(device).unsqueeze(0)
            )

        # Resize to input image res via disparity-space interpolation.
        mono_depths_resized = [
            1.0
            / torch.nn.functional.interpolate(
                1.0 / depth.clamp(min=1e-6),
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for depth in mono_depths
        ]

        if return_unresized:
            return torch.cat(mono_depths_resized, dim=0), torch.cat(mono_depths, dim=0)
        return torch.cat(mono_depths_resized, dim=0)


@torch.no_grad()
def call_monodepth_expert(
    images: torch.Tensor,  # (N, 3, H, W)
    monodepth_id: str = "depth-anything/da3mono-large",
    pred_res: int = 1596,
    process_res_method: str = "upper_bound_resize",
    device: torch.device = "cuda",
    return_unresized: bool = False,
    expert: Optional[MonodepthExpert] = None,
):
    """One-shot wrapper around :class:`MonodepthExpert`.

    When ``expert`` is provided the model is reused; otherwise a fresh
    :class:`MonodepthExpert` is instantiated and discarded after the call.
    """
    own = expert is None
    if own:
        expert = MonodepthExpert(monodepth_id, device)

    out = expert.predict(
        images=images,
        pred_res=pred_res,
        process_res_method=process_res_method,
        return_unresized=return_unresized,
    )

    if own:
        # Free the local copy so we don't hold the DA3 weights longer
        # than the caller needs them.
        del expert
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Depth -> normal helpers
# ---------------------------------------------------------------------------


def depth_to_normal_with_mask(view, depth: torch.Tensor):
    """Per-pixel normal estimator using the camera intrinsics & finite differences."""
    Fx = view.image_width / (2 * math.tan(view.FoVx / 2.0))
    Fy = view.image_height / (2 * math.tan(view.FoVy / 2.0))
    Cx = float(view.image_width - 1) / 2
    Cy = float(view.image_height - 1) / 2

    W, H = view.image_width, view.image_height
    x = (torch.arange(W, device="cuda", dtype=torch.float32) - Cx) / Fx
    y = (torch.arange(H, device="cuda", dtype=torch.float32) - Cy) / Fy
    points = torch.cat(
        [depth * x[None, None], depth * y[None, :, None], depth], dim=0
    )
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dy, dx, dim=0), dim=0)
    output = torch.nn.functional.pad(normal_map, (1, 1, 1, 1))

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


def get_normals_guidance_from_monodepth(
    monodepths: torch.Tensor,
    cameras: List[Camera],
    reference_depths: Optional[torch.Tensor] = None,
    conf_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-view normals from monodepth, optionally aligned to VGGT.

    Args:
        monodepths: ``(N, 1, H, W)`` (any resolution; ``cameras`` will be
            resized to match if needed).
        cameras: length ``N`` list of :class:`~surflo.structures.cameras.Camera`.
        reference_depths: optional ``(N, 1, H, W)`` reference depth at
            ``cameras[i].image_height/_width``; when supplied, the
            monodepth is rescaled per-view to match the median + median-
            absolute-deviation of the reference inside ``conf_masks``.
        conf_masks: optional ``(N, 1, H, W)`` boolean mask at the same
            resolution as ``reference_depths``; used to gate the median
            statistics. When ``None`` an all-true mask is used.

    Returns:
        ``(N, 3, H, W)`` normals at the *monodepth* resolution.
    """
    depth_h, depth_w = monodepths.shape[-2:]
    camera_h, camera_w = cameras[0].image_height, cameras[0].image_width
    if camera_h != depth_h or camera_w != depth_w:
        cameras_to_use = build_high_res_cameras(cameras, H=depth_h, W=depth_w)
    else:
        cameras_to_use = cameras

    if conf_masks is not None:
        conf_h, conf_w = conf_masks.shape[-2:]
        if conf_h != depth_h or conf_w != depth_w:
            conf_masks_to_use = torch.nn.functional.interpolate(
                conf_masks.float(),
                size=(depth_h, depth_w),
                mode="nearest",
            ).bool()
        else:
            conf_masks_to_use = conf_masks
    else:
        conf_masks_to_use = torch.ones_like(monodepths).bool()
        conf_masks = torch.ones_like(reference_depths).bool()

    normals: List[torch.Tensor] = []
    for ci in range(len(cameras_to_use)):
        depth = monodepths[ci].clone()  # (1, H, W)
        camera = cameras_to_use[ci]

        if reference_depths is not None:
            ref_depth = reference_depths[ci]  # (1, H, W)

            conf_mask = conf_masks[ci]  # (1, H, W)
            conf_mask_to_use = conf_masks_to_use[ci]  # (1, H, W)

            # Empty per-view confidence mask -> medians over a 0-element
            # tensor return NaN, which poisons depth, normals, and finally
            # the normal_guidance loss in `guided_inference`. Fall
            # back to the full image so this view at least produces a valid
            # (if globally rescaled) normal map.
            if not bool(conf_mask.any()):
                _log.warning(
                    f"[normal_guidance] view {ci}: VGGT-conf mask is empty "
                    f"(conf_threshold_for_normal_guidance may be too strict); "
                    f"falling back to full image for depth alignment."
                )
                conf_mask = torch.ones_like(conf_mask)
                conf_mask_to_use = torch.ones_like(conf_mask_to_use)

            ref_med = ref_depth[conf_mask].median()
            ref_scale = (ref_depth[conf_mask] - ref_med).abs().median()

            mono_med = depth[conf_mask_to_use].median()
            mono_scale = (depth[conf_mask_to_use] - mono_med).abs().median()

            depth = (depth - mono_med) / mono_scale
            depth = depth * ref_scale + ref_med

            depth = depth.clamp(min=1e-6)

        normal, _ = depth_to_normal_with_mask(camera, depth)  # (3, H, W)
        normals.append(normal.unsqueeze(0))  # (1, 3, H, W)

    return torch.cat(normals, dim=0)


# ---------------------------------------------------------------------------
# High-level entry point: scene -> (monodepths, normals)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_guidance_tensors(
    batch: dict,
    scene_idx: int,
    *,
    expert: MonodepthExpert,
    pred_res: int = 1596,
    process_res_method: str = "upper_bound_resize",
    conf_threshold_for_normal_guidance: float = 2.0,
    compute_monodepths: bool = True,
    compute_normals: bool = True,
    cameras: Optional[List[Camera]] = None,
    device: Optional[torch.device | str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute per-scene ``(monodepths, normal_guidances)`` for FFM guidance.

    Monodepth runs at ``pred_res``; normals are extracted from the *high-res*
    prediction with VGGT-depth-aligned median+scale (gated by
    ``vggt_depth_conf > conf_threshold_for_normal_guidance``), then bilinearly
    resized back to the input image resolution.

    Args:
        batch: an FFM batch (must contain ``rgb_images``,
            ``vggt_intrinsics``, ``vggt_extrinsics``; for
            ``compute_normals=True`` it must additionally contain
            ``vggt_depth`` and ``vggt_depth_conf``).
        scene_idx: index into the leading batch dimension of ``batch``.
        expert: a (re-usable) :class:`MonodepthExpert`.
        pred_res, process_res_method: forwarded to ``expert.predict``.
        conf_threshold_for_normal_guidance: VGGT confidence threshold
            gating the median-stats alignment of monodepth to VGGT depth.
        compute_monodepths, compute_normals: skip-flags. When
            ``compute_normals=False`` the function returns ``(monodepths, None)``;
            when ``compute_monodepths=False`` it returns ``(None, normals)``.
            (Computing normals always requires a monodepth pass under
            the hood; the returned monodepth tensor is just suppressed.)
        cameras: optional pre-built cameras matching the scene
            (re-uses them if provided; otherwise re-derives from
            ``vggt_intrinsics`` / ``vggt_extrinsics``).
        device: optional override; defaults to ``expert.device``.

    Returns:
        ``(mono_depths_resized, monodepth_normals)`` where each entry is
        either a tensor at the input image resolution or ``None`` when
        the corresponding compute flag is off.
    """
    if not (compute_monodepths or compute_normals):
        return None, None

    device = torch.device(device) if device is not None else expert.device
    images = batch["rgb_images"][scene_idx].to(device)  # (N, 3, H, W)

    # Always need both resolutions when computing normals (so we can
    # extract them on the high-res prediction); when normals are off,
    # we still ask for the unresized one for free since the model
    # already produced it.
    mono_resized, mono_highres = expert.predict(
        images=images,
        pred_res=pred_res,
        process_res_method=process_res_method,
        return_unresized=True,
    )

    monodepth_normals: Optional[torch.Tensor] = None
    if compute_normals:
        if cameras is None:
            cameras = get_cameras_from_intrinsics_and_extrinsics(
                intrinsics=batch["vggt_intrinsics"][scene_idx],
                extrinsics=batch["vggt_extrinsics"][scene_idx],
                images=batch["rgb_images"][scene_idx],
                data_device=str(device),
            )

        if "vggt_depth" not in batch or "vggt_depth_conf" not in batch:
            raise RuntimeError(
                "compute_guidance_tensors(compute_normals=True) requires "
                "`vggt_depth` and `vggt_depth_conf` in the batch."
            )

        ref_depth = batch["vggt_depth"][scene_idx].permute(0, 3, 1, 2).to(device)
        conf_mask = (
            batch["vggt_depth_conf"][scene_idx].unsqueeze(1).to(device)
            > float(conf_threshold_for_normal_guidance)
        )

        # Extract normals from the high-res monodepth, not the resized one.
        monodepth_normals = get_normals_guidance_from_monodepth(
            monodepths=mono_highres,
            cameras=cameras,
            reference_depths=ref_depth,
            conf_masks=conf_mask,
        )
        # Resize the normals back to the input image resolution so they
        # can be indexed by camera index in ``guided_inference``.
        monodepth_normals = torch.nn.functional.interpolate(
            monodepth_normals,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    return (
        mono_resized if compute_monodepths else None,
        monodepth_normals,
    )

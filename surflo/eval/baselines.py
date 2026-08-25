"""Feed-forward depth baselines for quantitative evaluation.

Two baselines run per-scene forward passes that return the same contract
(``points`` / ``depth`` / ``confidence`` / ``extrinsics`` / ``intrinsics`` +
optional ``processed_images``), so ``scripts/evaluate.py`` can score them (raw
point map or TSDF mesh) through the exact same alignment + Chamfer/F1 core as
Surflo:

* **VGGT** — the VGGT-1B backbone already vendored inside the Surflo model.
  ``run_vggt_predictor`` is just ``model.preprocess_images(...)`` unpacked.
* **DepthAnything3 (DA3)** — a standalone HuggingFace model. It returns depth +
  cameras but no world point map, so it is unprojected here via
  :func:`surflo.utils.geometry.depths_to_points_parallel`.

None of the core Surflo model / inference code depends on this module.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange

from surflo.utils.da3_path import ensure_da3_importable
from surflo.utils.geometry import depths_to_points_parallel

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DepthAnything3 loading
# ---------------------------------------------------------------------------
def load_da3_model(
    model_id: str,
    device: torch.device,
    *,
    eval_mode: bool = True,
) -> torch.nn.Module:
    """Load a DepthAnything3 model from HuggingFace Hub onto ``device``."""
    ensure_da3_importable()
    # Imported lazily so this module stays importable without the DA3 package.
    from depth_anything_3.api import DepthAnything3  # type: ignore

    _log.info(f"[da3] Loading DepthAnything3 from {model_id!r}...")
    da3 = DepthAnything3.from_pretrained(model_id).to(device)
    if eval_mode:
        da3.eval()
    return da3


# ---------------------------------------------------------------------------
# Per-scene forward passes
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_vggt_predictor(model, rgb_images: torch.Tensor) -> Dict[str, Any]:
    """Run the (Surflo-vendored) VGGT backbone on one scene.

    Args:
        rgb_images: ``(N, 3, H, W)`` float in ``[0, 1]``.

    Returns the same dict shape as :func:`run_da3_predictor`.
    """
    extra = model.preprocess_images(images=rgb_images.unsqueeze(0))
    points = extra["vggt_world_points"].squeeze(0).float()       # (N, H, W, 3)
    depth = extra["vggt_depth"].squeeze(0).squeeze(-1).float()   # (N, H, W)
    extrinsics = extra["vggt_extrinsics"].squeeze(0).float()     # (N, 3, 4)
    intrinsics = extra["vggt_intrinsics"].squeeze(0).float()     # (N, 3, 3)
    conf = extra.get("vggt_depth_conf")
    if conf is not None:
        conf = conf.squeeze(0).float()                           # (N, H, W)
    return {
        "points": points,
        "depth": depth,
        "confidence": conf,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "processed_images": None,
    }


@torch.no_grad()
def run_da3_predictor(
    da3_model: torch.nn.Module,
    rgb_images: torch.Tensor,        # (N, 3, H, W), float in [0, 1]
    *,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> Dict[str, Any]:
    """Run DepthAnything3 on one scene and unproject its depth to a point map.

    DA3 rescales internally (default ``process_res=504``); the returned depth /
    intrinsics / ``processed_images`` all live at that rescaled resolution.
    """
    device = rgb_images.device

    # DA3 expects a list of HWC uint8 numpy arrays.
    images_uint8 = (
        rgb_images.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0
    ).astype(np.uint8)
    images_uint8 = rearrange(images_uint8, "n c h w -> n h w c")
    images_list = list(images_uint8)

    pred = da3_model.inference(
        images_list,
        process_res=process_res,
        process_res_method=process_res_method,
    )

    depth = torch.from_numpy(np.ascontiguousarray(pred.depth)).float().to(device)            # (N, H', W')
    intrinsics = torch.from_numpy(np.ascontiguousarray(pred.intrinsics)).float().to(device)  # (N, 3, 3)

    extrinsics = torch.from_numpy(np.ascontiguousarray(pred.extrinsics)).float().to(device)
    if extrinsics.ndim == 3 and extrinsics.shape[-2:] == (4, 4):
        extrinsics = extrinsics[:, :3, :]
    if extrinsics.shape[-2:] != (3, 4):
        raise RuntimeError(
            f"[da3] Unexpected extrinsics shape {tuple(extrinsics.shape)}; "
            f"expected (N, 3, 4) or (N, 4, 4)."
        )

    conf: Optional[torch.Tensor] = None
    if pred.conf is not None:
        conf = torch.from_numpy(np.ascontiguousarray(pred.conf)).float().to(device)          # (N, H', W')

    # DA3 returns no world point map -> unproject depth with its own cameras.
    points = depths_to_points_parallel(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth=depth,
        to_world=True,
    )                                                                                        # (N, H', W', 3)

    processed_images: Optional[torch.Tensor] = None
    if pred.processed_images is not None:
        proc = torch.from_numpy(np.ascontiguousarray(pred.processed_images))
        proc = proc.to(dtype=torch.float32, device=device) / 255.0
        proc = rearrange(proc, "n h w c -> n c h w")                                         # (N, 3, H', W')
        processed_images = proc

    return {
        "points": points,
        "depth": depth,
        "confidence": conf,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "processed_images": processed_images,
    }


@torch.no_grad()
def run_baseline_predictor(
    predictor: str,
    *,
    model,
    batch: dict,
    device: torch.device,
    da3_model: Optional[torch.nn.Module] = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> Dict[str, Any]:
    """Dispatch to the VGGT / DA3 forward pass for one preprocessed scene.

    The scene RGB is read from ``batch['rgb_images']`` (needs
    ``scripts/preprocess.py --save_rgb_images``). The returned dict adds a
    ``tsdf_images`` key: the per-view RGB at the depth resolution used for TSDF
    color sampling (DA3's ``processed_images`` when it rescales, else the input
    RGB).
    """
    rgb = batch.get("rgb_images")
    if rgb is None:
        raise RuntimeError(
            f"predictor={predictor!r} needs 'rgb_images' in the batch; "
            f"re-run scripts/preprocess.py with --save_rgb_images."
        )
    scene_rgb = rgb[0].to(device)   # (N, 3, H, W)

    if predictor == "vggt":
        out = run_vggt_predictor(model, scene_rgb)
    elif predictor == "da3":
        if da3_model is None:
            raise RuntimeError("predictor='da3' requires a loaded da3_model.")
        out = run_da3_predictor(
            da3_model, scene_rgb,
            process_res=process_res, process_res_method=process_res_method,
        )
    else:
        raise ValueError(
            f"Unknown baseline predictor={predictor!r}; expected 'vggt' or 'da3'."
        )

    out["tsdf_images"] = (
        out["processed_images"] if out["processed_images"] is not None else scene_rgb
    )
    return out

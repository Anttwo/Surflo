import logging
from typing import List, Optional
from copy import deepcopy

import torch

from surflo.structures.cameras import Camera

_log = logging.getLogger(__name__)


def build_high_res_cameras(
    cameras: List[Camera],
    guidance_images: Optional[torch.Tensor] = None,  # (N, 3, H, W)
    H: Optional[int] = None,
    W: Optional[int] = None,
) -> List[Camera]:
    """Copy ``cameras`` with their image size overridden to a higher resolution.

    Two callers, two modes:

    * ``guidance_images`` -- high-res RGB supervision for the rendering guidance.
      The new size is taken from the tensor and the images are stored on the
      cameras (see :func:`surflo.inference.camera_refine.upsample_supervision`).
    * ``H`` / ``W`` -- used by the DepthAnything-3 expert, which predicts at the
      monodepth model's native resolution rather than the VGGT input resolution.
      No RGB is attached in that case.
    """
    assert (guidance_images is not None) or (H is not None and W is not None)
    if guidance_images is not None:
        H_hi, W_hi = int(guidance_images.shape[2]), int(guidance_images.shape[3])
    else:
        H_hi, W_hi = H, W

    new_cameras = deepcopy(cameras)

    # Sanity-check the aspect ratio against the VGGT inputs so the caller gets a
    # clear warning when the high-res images were not cropped to match the
    # encoder inputs.
    H_vggt = int(new_cameras[0].image_height)
    W_vggt = int(new_cameras[0].image_width)
    ratio_hi = W_hi / H_hi
    ratio_vggt = W_vggt / H_vggt
    if abs(ratio_hi - ratio_vggt) / max(ratio_vggt, 1e-6) > 0.02:
        _log.warning(
            f"high-res aspect ratio {ratio_hi:.4f} differs from the VGGT "
            f"input aspect ratio {ratio_vggt:.4f} by more than 2%. The rendered "
            f"Gaussians will not align pixel-for-pixel with the supervision."
        )

    _log.info(f"rendering at {H_hi}x{W_hi} (VGGT inputs at {H_vggt}x{W_vggt}).")

    # Override per-camera image dimensions and stored RGB tensor.
    for ci in range(len(cameras)):
        new_cameras[ci].original_image = (
            guidance_images[ci] if guidance_images is not None else None
        )
        new_cameras[ci].image_height = H_hi
        new_cameras[ci].image_width = W_hi

    return new_cameras

"""Load a folder of JPG/PNG images into a Surflo-ready batch.

Mirrors the sampling logic the demo uses: sample ``n_images`` frames from a
directory (deterministic ``uniform`` linspace or seeded ``random``), resize
with VGGT's ``no_stretch`` preprocessing, then run
:meth:`surflo.model.ffm.FFM.preprocess_images` (VGGT-1B encode) to produce the
cached-token batch that both plain and guided inference consume.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from surflo.data.utils import load_and_preprocess_images

_log = logging.getLogger(__name__)

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def list_images_in_folder(folder: str) -> List[str]:
    fnames = sorted(fn for fn in os.listdir(folder) if fn.endswith(IMAGE_EXTS))
    return [os.path.join(folder, fn) for fn in fnames]


def sample_image_indices(
    n_available: int,
    n_requested: int,
    sampling: str,
    seed: int,
) -> np.ndarray:
    """Pick which frames to keep.

    ``sampling`` is ``"uniform"`` (deterministic linspace) or ``"random"``
    (deterministic per ``seed``).
    """
    n = min(int(n_requested), int(n_available))
    if n <= 0:
        raise ValueError(
            f"n_requested={n_requested} but n_available={n_available}; "
            f"need at least one image to sample."
        )
    if sampling == "random":
        rng = np.random.default_rng(int(seed))
        idxs = np.sort(rng.choice(n_available, size=n, replace=False))
    elif sampling in ("uniform", "uniform-deterministic", "linspace"):
        if n == 1:
            idxs = np.array([n_available // 2])
        else:
            idxs = np.linspace(0, n_available - 1, num=n).round().astype(int)
            idxs = np.unique(idxs)
    else:
        raise ValueError(
            f"Unknown sampling={sampling!r}; expected 'uniform' or 'random'."
        )
    return idxs


@torch.no_grad()
def build_image_folder_batch(
    model,
    *,
    folder: str,
    n_images: int,
    sampling: str = "uniform",
    seed: int = 42,
    target_size: int = 518,
    cull_radius: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Folder of JPG/PNG -> single pre-processed batch (``B = 1``).

    Returns ``(batch, selected_paths)`` where ``batch`` is the output of
    ``model.preprocess_images`` (cached VGGT tokens + geometry) ready for
    :func:`surflo.inference.plain.run_plain_inference` or the guided path.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"image_folder {folder!r} does not exist or is not a directory."
        )
    paths = list_images_in_folder(folder)
    if not paths:
        raise RuntimeError(f"No JPG/PNG images found in {folder!r}.")

    device = model.device if device is None else device

    idxs = sample_image_indices(
        n_available=len(paths),
        n_requested=n_images,
        sampling=sampling,
        seed=seed,
    )
    selected = [paths[i] for i in idxs]
    _log.info(
        f"[image_folder] {folder}: loading {len(selected)}/{len(paths)} "
        f"images (sampling={sampling}, target_size={target_size})."
    )
    for p in selected:
        _log.info(f"  - {os.path.basename(p)}")

    images = load_and_preprocess_images(
        selected, mode="no_stretch", target_size=int(target_size),
        rotate_portrait=True,  # inference: correct the landscape bias
    ).to(device)

    cr = (
        float(cull_radius)
        if cull_radius is not None and float(cull_radius) > 0.0
        else None
    )
    batch = model.preprocess_images(images=images, cull_radius=cr)
    batch["seq_name"] = [os.path.basename(folder.rstrip("/"))]
    return batch, selected

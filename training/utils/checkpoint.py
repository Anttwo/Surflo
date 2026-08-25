"""Preemption-safe checkpoint saving/loading."""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr


class DDPCheckpointSaver:
    """Saves a checkpoint (only on rank 0) under one or more names."""

    def __init__(self, checkpoint_folder: str, checkpoint_names: List[str], rank: int, epoch: int):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.worker_id = rank
        self.epoch = epoch

    def save_checkpoint(self, model: nn.Module, **kwargs: Any) -> None:
        checkpoint = dict(**kwargs)
        checkpoint["model"] = model.state_dict()

        if self.worker_id == 0:
            for ckpt_name in self.checkpoint_names:
                checkpoint_path = os.path.join(self.checkpoint_folder, f"{ckpt_name}.pt")
                logging.info(f"Saving checkpoint at epoch {self.epoch} to {checkpoint_path}")
                robust_torch_save(checkpoint, checkpoint_path)


def robust_torch_save(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """Save a checkpoint atomically with a rolling single-generation backup.

    Preemption-safe save strategy:

    1. Remove any stale ``<path>.tmp`` left by a previously crashed save.
    2. Write the new checkpoint to ``<path>.tmp``, ``flush()`` + ``fsync()``
       so the bytes are on disk before we touch the committed file.
    3. If ``<path>`` already exists, ``os.replace`` it onto ``<path>.bak``
       (atomically overwriting any previous backup).
    4. ``os.replace`` ``<path>.tmp`` onto ``<path>``.

    After a *successful* save, the valid files on disk are ``<path>`` (new)
    and ``<path>.bak`` (previous, absent on the first save). If interrupted,
    the load path below falls back to ``<path>.bak``.
    """
    tmp_path = checkpoint_path + ".tmp"
    bak_path = checkpoint_path + ".bak"

    # 1. Clean up a stale tmp from a previous crashed save, if any.
    if g_pathmgr.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError as e:
            logging.warning(f"robust_torch_save: could not remove stale tmp {tmp_path}: {e!r}")

    # 2. Write to the tmp file and fsync so a crash after the rename in (4)
    #    cannot leave us with a truncated payload at the committed path.
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(tmp_path, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError as e:
            logging.warning(f"robust_torch_save: fsync failed on {tmp_path}: {e!r}")

    # 3. Rotate the current committed file into the backup slot.
    if g_pathmgr.exists(checkpoint_path):
        try:
            os.replace(checkpoint_path, bak_path)
        except OSError as e:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise RuntimeError(
                f"robust_torch_save: could not rotate {checkpoint_path} -> {bak_path}: {e!r}"
            ) from e

    # 4. Atomically commit the tmp file as the new checkpoint.
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint_with_fallback(ckpt_path: str, map_location: Any = "cpu") -> Tuple[Dict[str, Any], str]:
    """Load ``ckpt_path`` with automatic fallback to ``ckpt_path + ".bak"``.

    Returns ``(checkpoint_dict, actual_path_loaded)``. Raises if neither the
    primary file nor the backup is loadable.
    """
    bak_path = ckpt_path + ".bak"

    candidates: List[str] = []
    if g_pathmgr.exists(ckpt_path):
        candidates.append(ckpt_path)
    if g_pathmgr.exists(bak_path):
        candidates.append(bak_path)

    if not candidates:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path} or {bak_path}")

    last_error: Optional[BaseException] = None
    for path in candidates:
        try:
            with g_pathmgr.open(path, "rb") as f:
                checkpoint = torch.load(f, map_location=map_location)
        except Exception as e:
            logging.warning(
                f"load_checkpoint_with_fallback: failed to load {path}: {e!r}. "
                f"{'Trying backup...' if path != candidates[-1] else ''}"
            )
            last_error = e
            continue
        if path != ckpt_path:
            logging.warning(
                f"load_checkpoint_with_fallback: primary checkpoint at {ckpt_path} was "
                f"unusable; loaded backup {path} instead."
            )
        return checkpoint, path

    raise RuntimeError(
        f"Could not load checkpoint from {ckpt_path} or its backup {bak_path}. "
        f"Last error: {last_error!r}"
    ) from last_error

"""Weights & Biases logger for scalars and point clouds, on rank 0 only.

Two payload kinds are supported: scalar metrics (:meth:`log_dict`) and 3D point
clouds as ``wandb.Object3D`` (:meth:`log_point_cloud`, driven by the ``viz``
block of ``configs/logging/default.yaml``). Images, videos, histograms and
``wandb.watch`` are not supported.

/!\ Point clouds dominate the on-disk size of a run: ``viz`` defaults to 100k
points per cloud, prediction *and* ground truth, for several scenes, and an
offline run directory grows into the hundreds of GB over a long training chain.
Set ``logging.viz.enabled=false`` when only the curves matter.

Pending values are buffered and flushed with :meth:`commit` so a whole step is
logged in a single ``wandb.log`` call (keeping train/val step alignment).
"""
import atexit
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

import wandb

from .distributed import get_machine_local_and_dist_rank


class WandBLogger:
    """Minimal wandb wrapper. Only rank 0 logs."""

    def __init__(self, entity: str, project: str, exp_name: str, summary_writer_method: Any = None) -> None:
        self._run: Optional[Any] = None
        _, self._rank = get_machine_local_and_dist_rank()
        if self._rank == 0:
            try:
                self._run = wandb.init(project=project, entity=entity, name=exp_name)
            except Exception:
                logging.exception("Failed to initialize wandb; continuing without logging.")
                self._run = None
        else:
            logging.debug(f"Not logging on this process because rank {self._rank} != 0")

        self.last_train_step = -1
        self._pending: Dict[str, Any] = {}

        atexit.register(self.close)

    @property
    def writer(self) -> Optional[Any]:
        return self._run

    def flush(self) -> None:
        """wandb logs are sent on commit; no-op, kept for API compatibility."""
        return

    def _resolve_step(self, step: int, phase=None) -> int:
        """Keep train/val step alignment (val is logged just after last train)."""
        if phase == "train":
            self.last_train_step = step
        elif phase is not None and phase != "train":
            step = self.last_train_step + 1 if self.last_train_step >= 0 else step
        return step

    def commit(self, step: int, phase=None) -> None:
        """Flush all pending scalars to wandb in a single log call."""
        if not self._run or not self._pending:
            return
        resolved = self._resolve_step(step, phase)
        try:
            wandb.log(self._pending, step=resolved)
        except Exception:
            logging.exception("Failed to commit wandb log")
        self._pending = {}

    def close(self) -> None:
        """Finish the wandb run, flushing any buffered data first."""
        if self._run is not None:
            try:
                if self._pending:
                    wandb.log(self._pending)
                    self._pending = {}
                wandb.finish()
            except Exception:
                logging.exception("Exception while finishing wandb run")
            finally:
                self._run = None

    def log_dict(self, payload: Dict[str, Any], step: int, phase=None) -> None:
        """Buffer multiple scalar values for the current step."""
        if not self._run:
            return
        for k, v in payload.items():
            self.log(k, v, step, phase=phase)

    def log(self, name: str, data: Any, step: int, phase=None) -> None:
        """Buffer a single scalar (or 1-element tensor) for the current step."""
        if not self._run:
            return
        value: Any = data
        if isinstance(data, torch.Tensor):
            if data.numel() == 1:
                value = data.item()
            else:
                value = data.detach().cpu().numpy()
        self._pending[name] = value

    def log_point_cloud(
        self, name: str, points: torch.Tensor,
        colors: Optional[torch.Tensor] = None, step: int = 0, phase=None,
    ) -> None:
        """Buffer a 3D point cloud (``wandb.Object3D``) for the current step.

        Flushed alongside scalars by the next :meth:`commit`.

        Args:
            points: ``(N, 3)`` XYZ positions.
            colors: optional ``(N, 3)`` RGB in ``[0, 1]``; appended as ``(N, 6)``.
        """
        if not self._run:
            return
        try:
            pts = points.detach().cpu().float().numpy()
            if colors is not None:
                cols = (colors.detach().cpu().float().clamp(0.0, 1.0) * 255.0).numpy()
                cloud = np.concatenate([pts, cols], axis=-1)  # (N, 6) XYZ + RGB
            else:
                cloud = pts
            self._pending[name] = wandb.Object3D(cloud)
        except Exception:
            logging.exception("Failed to log point cloud to wandb")

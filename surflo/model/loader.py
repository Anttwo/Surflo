"""Checkpoint loading for the Surflo model.

The released checkpoints store both the raw training weights (``"model"``) and
an exponential-moving-average copy (``"ema_state"``, produced by
``ema_pytorch.EMA`` with ``include_online_model=False``). Evaluation uses the
EMA weights. The EMA state dict simply prefixes every model key with
``"ema_model."`` plus two bookkeeping buffers (``initted`` / ``step``), so we
load it by stripping that prefix — no dependency on ``ema_pytorch`` at
inference time.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Union

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

_log = logging.getLogger(__name__)

_EMA_PREFIX = "ema_model."


def _ema_weights(ema_state: dict) -> dict:
    """Extract model weights from an ``ema_pytorch`` state dict."""
    return {
        k[len(_EMA_PREFIX):]: v
        for k, v in ema_state.items()
        if k.startswith(_EMA_PREFIX)
    }


def load_checkpoint_weights(model: torch.nn.Module, ckpt: dict, use_ema: bool = True) -> None:
    """Load Surflo weights from a loaded checkpoint dict into ``model`` in place.

    Two checkpoint layouts are accepted:
      * ``"ema_state"`` present -> EMA weights when ``use_ema``, otherwise the
        raw ``"model"`` weights;
      * older checkpoints -> EMA weights inlined into ``"model"`` under an
        ``ema_model.*`` prefix.
    """
    if "ema_state" in ckpt:
        if use_ema:
            model.load_state_dict(_ema_weights(ckpt["ema_state"]), strict=True)
            _log.info("Loaded EMA weights from checkpoint.")
        else:
            model.load_state_dict(ckpt["model"], strict=True)
            _log.info("Loaded training weights (EMA disabled).")
        return

    model_sd = ckpt["model"]
    has_ema_keys = any(_EMA_PREFIX in k for k in model_sd)
    if use_ema and has_ema_keys:
        model.load_state_dict(_ema_weights(model_sd), strict=True)
        _log.info("Loaded EMA weights from checkpoint (legacy format).")
    else:
        model.load_state_dict(model_sd, strict=True)
        _log.info("Loaded training weights.")


def load_model(
    model_cfg: DictConfig,
    ckpt_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    use_ema: bool = True,
) -> torch.nn.Module:
    """Instantiate the Surflo model from a Hydra config and load a checkpoint.

    Args:
        model_cfg: the ``model`` config node (see ``configs/model/surflo.yaml``).
            Instantiated with ``_recursive_=False`` so that ``FFM.__init__``
            recursively builds its own submodules (matching training).
        ckpt_path: path to a Surflo checkpoint (the released one is
            ``surflo_v0.pt``; training writes ``checkpoint.pt`` /
            ``checkpoint_<epoch>.pt``). If ``None``, an untrained model is
            returned (the VGGT backbone is still loaded from the hub).
        device: device to place the model on.
        use_ema: load the EMA weights (default, matches evaluation).

    Returns:
        The model in ``eval()`` mode on ``device``.
    """
    device = torch.device(device)
    model = instantiate(model_cfg, _recursive_=False)
    model.to(device)
    model.eval()

    if ckpt_path is None:
        _log.warning("No checkpoint specified — returning an untrained model.")
        return model
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    _log.info(f"Loading checkpoint: {ckpt_path}")
    # mmap keeps the (large) optimizer state off-RAM; only referenced tensors
    # are materialised by load_state_dict.
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    load_checkpoint_weights(model, ckpt, use_ema=use_ema)
    del ckpt
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model

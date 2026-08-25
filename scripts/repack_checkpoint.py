"""Repack a training checkpoint into a smaller, inference-only one.

A training checkpoint stores two full copies of the weights (``model``, the
online weights, and ``ema_state``, the EMA shadow) plus the optimizer state.
Inference reads the EMA weights only -- ``surflo.model.loader`` checks
``ema_state`` first and prefers it whenever ``use_ema`` is set, which is the
default and what every reported number uses. Dropping ``model``, ``optimizer``
and ``scaler`` therefore leaves inference bit-identical while removing roughly
half the file.

    python scripts/repack_checkpoint.py \\
        logs/<run>/ckpts/surflo_v0.pt \\
        checkpoint_200_inference.pt

The result **cannot be used to resume training** (no optimizer state, no online
weights). Keep the original if you may want to continue the run.

Pass ``--keep-online`` to retain ``model`` as well -- useful if you want a
checkpoint that can still be evaluated with ``use_ema=false``.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# Keys that carry training state and are never read during inference.
_TRAINING_ONLY = ("optimizer", "scaler")
# The online weights: a second full copy of the model, unused when EMA is used.
_ONLINE_WEIGHTS = "model"
# Small bookkeeping worth keeping so the file stays self-describing.
_METADATA = ("epoch", "steps", "time_elapsed")


def _nbytes(obj: Any) -> int:
    """Recursive size in bytes of the tensors reachable from ``obj``."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    return 0


def _fingerprint(state: Dict[str, Any]) -> Dict[str, str]:
    """Per-tensor content hash, for verifying the repack changed no weights."""
    out: Dict[str, str] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            out[k] = hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]
    return out


def repack(src: Path, dst: Path, *, keep_online: bool, force: bool) -> int:
    if not src.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {src}")
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} exists; pass --force to overwrite.")
    if dst.resolve() == src.resolve():
        raise ValueError("Refusing to overwrite the source checkpoint in place.")

    # mmap keeps the (large) discarded tensors off RAM: only what we re-save is
    # actually read from disk.
    ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)

    if "ema_state" not in ckpt:
        raise RuntimeError(
            f"{src} has no 'ema_state'. This checkpoint was written with EMA "
            f"disabled, so there are no EMA weights to keep and stripping "
            f"'model' would leave nothing to load. Keys present: {sorted(ckpt)}."
        )

    drop = list(_TRAINING_ONLY) + ([] if keep_online else [_ONLINE_WEIGHTS])

    print(f"  source: {src}  ({src.stat().st_size / 1e9:.2f} GB on disk)")
    print("  key sizes:")
    for k in ckpt:
        mark = "DROP" if k in drop else "keep"
        print(f"    [{mark}] {k:<14} {_nbytes(ckpt[k]) / 1e9:7.2f} GB")

    out: Dict[str, Any] = {"ema_state": ckpt["ema_state"]}
    if keep_online and _ONLINE_WEIGHTS in ckpt:
        out[_ONLINE_WEIGHTS] = ckpt[_ONLINE_WEIGHTS]
    for k in _METADATA:
        if k in ckpt:
            out[k] = ckpt[k]

    before = _fingerprint(ckpt["ema_state"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)

    # ---- verify: reload and compare the EMA tensors byte-for-byte ----
    reloaded = torch.load(dst, map_location="cpu", mmap=True, weights_only=False)
    after = _fingerprint(reloaded["ema_state"])
    if before != after:
        changed = sorted(k for k in before if before.get(k) != after.get(k))
        raise RuntimeError(
            f"Verification FAILED: {len(changed)} EMA tensor(s) differ after "
            f"repack, e.g. {changed[:5]}. The output was written but must not "
            f"be used."
        )

    src_gb, dst_gb = src.stat().st_size / 1e9, dst.stat().st_size / 1e9
    print(f"\n  wrote: {dst}  ({dst_gb:.2f} GB)")
    print(f"  kept keys: {sorted(out)}")
    print(f"  verified : {len(after)} EMA tensors bit-identical to the source")
    print(f"  size     : {src_gb:.2f} GB -> {dst_gb:.2f} GB "
          f"({(1 - dst_gb / src_gb) * 100:.0f}% smaller)")
    if not keep_online:
        print("\n  NOTE: this file is inference-only -- it cannot resume training.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Strip training-only state from a Surflo checkpoint.",
    )
    p.add_argument("src", type=Path, help="Input checkpoint (.pt).")
    p.add_argument("dst", type=Path, help="Output checkpoint (.pt).")
    p.add_argument("--keep-online", action="store_true",
                   help="Also keep the online 'model' weights (allows use_ema=false).")
    p.add_argument("--force", action="store_true", help="Overwrite the output if it exists.")
    args = p.parse_args()
    return repack(args.src, args.dst, keep_online=args.keep_online, force=args.force)


if __name__ == "__main__":
    sys.exit(main())

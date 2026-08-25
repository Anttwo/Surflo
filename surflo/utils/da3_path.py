"""Locate the Depth-Anything-3 source tree and put it on ``sys.path``.

DA3 is not an installed package: its source lives in the repository (as a
submodule) and its importable package sits one level down, under
``Depth-Anything-3/src/depth_anything_3``. Anything that wants to import
``depth_anything_3`` must add that ``src`` directory to ``sys.path`` first —
which is what :func:`ensure_da3_importable` does.

A consequence worth remembering when debugging: a bare
``importlib.util.find_spec("depth_anything_3")`` reports "missing" on every
machine until this has run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# surflo/utils/da3_path.py -> parents[2] == the repository root
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Checked in order, so a checkout that keeps Depth-Anything-3 at the repo root
# (rather than under submodules/) also works.
_CANDIDATES = (
    _REPO_ROOT / "submodules" / "Depth-Anything-3" / "src",
    _REPO_ROOT / "Depth-Anything-3" / "src",
)


def da3_source_dir() -> Optional[Path]:
    """Return the DA3 ``src`` directory, or ``None`` if it cannot be found.

    ``$DA3_SRC`` wins when set, for installations outside the repository; it is
    returned as given, without an existence check, so a wrong value surfaces as
    a normal ImportError rather than being silently ignored.
    """
    env = os.environ.get("DA3_SRC")
    if env:
        return Path(env)
    for candidate in _CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def ensure_da3_importable() -> Optional[Path]:
    """Prepend the DA3 source directory to ``sys.path`` (idempotent).

    Returns the directory that was made importable, or ``None`` if no DA3
    checkout was found.
    """
    src = da3_source_dir()
    if src is None:
        return None
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src

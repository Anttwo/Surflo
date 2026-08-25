"""Surflo: Consistent 3D Surface Flow Model with Global State.

A clean, inference-focused re-packaging of the Surflo method: a variable
number of unposed RGB images is encoded by a frozen VGGT-1B backbone and a
Perceiver compressor into a single fixed-size global latent (``K`` tokens),
from which a per-point flow-matching ODE decodes an oriented surface point
cloud at arbitrary resolution. An optional rendering-guidance mechanism
couples the points via Gaussian-splatting gradients, and a wrapping-based
extractor turns the Gaussians into a mesh.

Project page: https://anttwo.github.io/surflo/

The :mod:`surflo.api` facade (``Surflo`` / ``SceneState`` / :func:`save_ply`) is
re-exported here for the "load it and play with it" workflow; see
``examples/quickstart.py``.
"""

__version__ = "0.1.0"

from .api import (
    SceneState,
    Surflo,
    save_ply,
    save_mesh,
    set_global_seeds,
    load_preset,
    load_and_preprocess_images,
)

__all__ = [
    "Surflo",
    "SceneState",
    "save_ply",
    "save_mesh",
    "set_global_seeds",
    "load_preset",
    "load_and_preprocess_images",
    "__version__",
]

"""Evaluation-only helpers for the non-Surflo baselines.

This subpackage is a self-contained, eval-only addition: it lets
``scripts/evaluate.py`` score the **VGGT** and **DepthAnything3 (DA3)**
feed-forward baselines (raw point maps or TSDF-fused meshes) using the exact
same alignment + Chamfer/F1 metric core as the Surflo evaluation. None of the
core Surflo model / inference code depends on anything here.

Two modules:

* :mod:`surflo.eval.baselines`  — per-scene VGGT / DA3 forward passes.
* :mod:`surflo.eval.tsdf_mesh`  — multi-resolution TSDF -> mesh -> surface
  sampling.
"""

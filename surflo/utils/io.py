"""PLY writers and color / normal helpers for Surflo outputs.

Blender-friendly PLY writers (binary little-endian by default) plus the
small color-mapping helpers used across the inference entry points:

  * :func:`write_points_ply` / :func:`write_mesh_ply` -- point-cloud and
    triangle-mesh PLY writers built on :mod:`trimesh`, with a stdlib
    binary-LE fallback that can embed both real normals (``nx/ny/nz``)
    and per-vertex RGB.
  * :func:`normals_to_rgb_uint8` -- the ``(1 - n) / 2`` normal->RGB mapping
    used everywhere in the demos.
  * :func:`uint8_colors_from_floats` / :func:`rgb_from_image_tensor` /
    :func:`tensor_to_numpy` -- numpy/color conversion helpers.

Nothing here mutates global state on import.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Color / numpy helpers.
# ---------------------------------------------------------------------------

def tensor_to_numpy(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        return np.asarray(t)
    return t.detach().cpu().float().numpy()


def normals_to_rgb_uint8(normals: torch.Tensor) -> np.ndarray:
    """``(N, 3)`` unit normals -> ``(N, 3)`` uint8 RGB.

    Implements ``(1 - n) / 2`` clamped to ``[0, 1]`` (matches the demo's
    ``_color_by_normals``) and scales to ``[0, 255]`` for PLY storage.
    """
    arr = tensor_to_numpy(normals)
    if arr is None or arr.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    arr = np.clip((1.0 - arr) / 2.0, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def uint8_colors_from_floats(colors: torch.Tensor) -> np.ndarray:
    """``(N, 3)`` float colors in ``[0, 1]`` -> ``(N, 3)`` uint8."""
    arr = tensor_to_numpy(colors)
    if arr is None or arr.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def rgb_from_image_tensor(images: torch.Tensor) -> np.ndarray:
    """``(N, 3, H, W)`` float in ``[0, 1]`` -> ``(N*H*W, 3)`` uint8.

    The per-pixel color buffer is laid out in the same 1-to-1 order as a
    flattened point map.
    """
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4:
        raise ValueError(
            f"rgb_from_image_tensor: expected (N, 3, H, W); got shape "
            f"{tuple(images.shape)}."
        )
    arr = images.detach().cpu().float().clamp(0.0, 1.0).numpy()
    n, c, h, w = arr.shape
    if c != 3:
        raise ValueError(f"rgb_from_image_tensor: expected 3 channels; got {c}.")
    arr = np.transpose(arr, (0, 2, 3, 1)).reshape(n * h * w, 3)
    return (arr * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# PLY writers (trimesh under the hood).
# ---------------------------------------------------------------------------

def _import_trimesh():
    try:
        import trimesh  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "trimesh is required to write PLY files but is not installed. "
            "Run `pip install trimesh` and retry."
        ) from e
    return trimesh


def _write_pointcloud_ply_binary_le(
    path: Path,
    *,
    points_f32: np.ndarray,             # (P, 3) float32
    normals_f32: Optional[np.ndarray],  # (P, 3) float32 or None
    colors_u8: Optional[np.ndarray],    # (P, 3) uint8 or None
) -> None:
    """Write a binary little-endian PLY with x/y/z (+ optional nx/ny/nz +
    optional red/green/blue) per vertex.

    Used when we want to embed both real normals and per-point RGB in the
    same PLY (Blender can then use either). The header property order and
    lowercased names match what Blender's PLY importer expects.
    """
    n = int(points_f32.shape[0])
    has_nrm = normals_f32 is not None
    has_col = colors_u8 is not None

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += ["property float x", "property float y", "property float z"]
    if has_nrm:
        header += ["property float nx", "property float ny", "property float nz"]
    if has_col:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header", ""]
    header_bytes = "\n".join(header).encode("ascii")

    dtype: List[Tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_nrm:
        dtype += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if has_col:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.zeros(n, dtype=dtype)
    arr["x"] = points_f32[:, 0]
    arr["y"] = points_f32[:, 1]
    arr["z"] = points_f32[:, 2]
    if has_nrm:
        arr["nx"] = normals_f32[:, 0]
        arr["ny"] = normals_f32[:, 1]
        arr["nz"] = normals_f32[:, 2]
    if has_col:
        arr["red"] = colors_u8[:, 0]
        arr["green"] = colors_u8[:, 1]
        arr["blue"] = colors_u8[:, 2]

    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(arr.tobytes(order="C"))


def write_points_ply(
    path: str,
    points,
    *,
    colors_uint8: Optional[np.ndarray] = None,
    normals=None,
    binary: bool = True,
) -> int:
    """Write a (optionally colored / oriented) point cloud to ``path`` (PLY).

    ``normals`` are stored as ``nx/ny/nz`` properties (orthogonal to the RGB
    channels, so callers can keep both). When normals are passed we always go
    through the stdlib binary writer; otherwise ``binary`` selects between
    trimesh binary and ASCII. Returns the number of points written.
    """
    pts = tensor_to_numpy(points)
    if pts is None or pts.size == 0:
        pts = np.zeros((0, 3), dtype=np.float32)
    pts = np.ascontiguousarray(pts.reshape(-1, 3).astype(np.float32))
    n = int(pts.shape[0])

    cols = colors_uint8
    if cols is not None:
        cols = np.ascontiguousarray(np.asarray(cols).reshape(-1, 3).astype(np.uint8))
        if cols.shape[0] != n:
            raise ValueError(f"colors_uint8 has {cols.shape[0]} rows but points has {n}.")

    nrm = tensor_to_numpy(normals) if normals is not None else None
    if nrm is not None:
        nrm = np.ascontiguousarray(nrm.reshape(-1, 3).astype(np.float32))
        if nrm.shape[0] != n:
            raise ValueError(f"normals has {nrm.shape[0]} rows but points has {n}.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if nrm is not None:
        _write_pointcloud_ply_binary_le(
            out_path, points_f32=pts, normals_f32=nrm, colors_u8=cols,
        )
        return n

    trimesh = _import_trimesh()
    pc = trimesh.PointCloud(vertices=pts.astype(np.float64), colors=cols)
    pc.export(str(out_path), encoding="binary" if binary else "ascii")
    return n


def write_mesh_ply(
    path: str,
    verts,
    faces,
    *,
    vertex_colors_uint8: Optional[np.ndarray] = None,
    binary: bool = True,
) -> Tuple[int, int]:
    """Write a triangle mesh to ``path`` (PLY). Returns ``(n_verts, n_faces)``."""
    trimesh = _import_trimesh()

    v = tensor_to_numpy(verts)
    f = tensor_to_numpy(faces)
    if v is None or f is None or v.size == 0 or f.size == 0:
        v = np.zeros((0, 3), dtype=np.float64)
        f = np.zeros((0, 3), dtype=np.int32)
    v = np.ascontiguousarray(v.reshape(-1, 3).astype(np.float64))
    f = np.ascontiguousarray(f.reshape(-1, 3).astype(np.int32))

    cols = vertex_colors_uint8
    if cols is not None:
        cols = np.ascontiguousarray(cols.reshape(-1, 3).astype(np.uint8))
        if cols.shape[0] != v.shape[0]:
            raise ValueError(
                f"vertex_colors_uint8 has {cols.shape[0]} rows but verts has {v.shape[0]}."
            )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=cols, process=False)
    mesh.export(str(out_path), encoding="binary" if binary else "ascii")
    return int(v.shape[0]), int(f.shape[0])

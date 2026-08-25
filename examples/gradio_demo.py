"""Gradio demo for Surflo -- a thin UI over the :mod:`surflo` Python API.

The model is loaded once at startup from ``--ckpt`` (before the server is
available), then the UI walks the same steps the API exposes, in order:

  1. Select + encode images                 -> :meth:`Surflo.encode`
  2. Inspect the encoded global state
  3. Decode (plain /                         -> :meth:`SceneState.reconstruct`
     guided {minimal,short,long,...})
  4. Visualize / save the colored cloud     -> :meth:`SceneState.color_points` + save_ply
  5. Extract + color + save the mesh        -> :meth:`SceneState.extract_mesh`
                                                :meth:`SceneState.color_mesh` + save_mesh

Everything heavy (the guided rendering guidance, mesh extraction, TSDF coloring)
needs the CUDA rasterizers + a GPU, exactly like ``scripts/infer.py``.

Run it from the repository root::

    pip install -e ".[demo]"          # installs gradio
    python examples/gradio_demo.py --ckpt /path/to/surflo_v0.pt

then open the printed URL. Pass ``--device`` to change the torch device,
``--share`` for a public link, ``--host`` / ``--port`` to change the bind address.

Pass ``--dev`` to swap the browser file-upload in step 1 for three plain fields
-- image directory, number of images (0 = all), sampling (uniform / random) --
that load images from the *server's* filesystem (the machine running this
script), mirroring the inference scripts. This is convenient when accessing the
demo over SSH, where the upload widget can only reach the client (laptop) disk.
"""
from __future__ import annotations

import argparse
import atexit
import logging
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow ``python examples/gradio_demo.py`` from a source checkout by putting the
# package root (the parent of examples/) on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gradio as gr  # noqa: E402

from surflo import (  # noqa: E402
    SceneState,
    Surflo,
    save_mesh,
    save_ply,
    set_global_seeds,
)

CONFIGS_DIR = _REPO_ROOT / "configs"


def _discover_guided_presets() -> List[str]:
    """Guided presets = the stems of ``configs/guided/*.yaml``.

    Discovered from disk (not hardcoded) so dropping a new YAML into
    ``configs/guided/`` makes it selectable in the demo with no code change.
    """
    return sorted(p.stem for p in (CONFIGS_DIR / "guided").glob("*.yaml"))


GUIDED_PRESETS = _discover_guided_presets()
# Preferred default if present, otherwise the first discovered preset.
DEFAULT_GUIDED_PRESET = (
    "minimal" if "minimal" in GUIDED_PRESETS
    else (GUIDED_PRESETS[0] if GUIDED_PRESETS else "minimal")
)
MODES = ["plain", "guided"]

# Fixed defaults for the simplified UI.
DEFAULT_MAX_VIEWS = 0             # 0 = use all selected images
DEFAULT_CULL_RADIUS = 10.0       # matches the CLI
DEFAULT_NUM_STEPS = 100          # plain-mode ODE steps
# Opacity cull for the displayed / exported point cloud ONLY. Meshing keeps every
# Gaussian (the reconstruction result is left unculled), so the wrapping mesh is
# built from all Gaussians -- matching scripts/infer.py (mesh always uses all Gaussians).
DEFAULT_OPACITY_THRESHOLD = 0.1

# Global-state portrait (latent-state heatmap). Off by default; flip to True to
# re-enable it. Controls both the UI component and the encode->viz wiring, so
# the feature can be brought back later without re-adding any code.
SHOW_GLOBAL_STATE = False

# Dev mode (``--dev``): replace the browser file-upload with three fields (image
# directory, num images, sampling) that read images from the *server's* disk --
# handy over SSH, where the upload widget can only see the client machine. Set
# from ``--dev`` in ``main`` before ``build_demo`` runs.
DEV_MODE = False

try:
    _GR_MAJOR = int(str(gr.__version__).split(".")[0])
except Exception:  # noqa: BLE001
    _GR_MAJOR = 0


def _bypass_localhost_proxy() -> None:
    """Make sure loopback addresses bypass any configured HTTP/SOCKS proxy.

    Gradio >= 6 does a startup self-check against ``http://127.0.0.1:<port>``.
    On machines that export ``HTTP(S)_PROXY`` / ``ALL_PROXY`` (very common in
    managed / sandboxed setups) that request is routed through the proxy and
    comes back ``403``, aborting ``launch()``. Adding the loopback hosts to
    ``no_proxy`` exempts them; outward traffic (e.g. Hugging-Face weight
    downloads during "Load model") still goes through the proxy.
    """
    loopback = ["localhost", "127.0.0.1", "::1", "0.0.0.0"]
    for var in ("no_proxy", "NO_PROXY"):
        current = [h.strip() for h in os.environ.get(var, "").split(",") if h.strip()]
        for h in loopback:
            if h not in current:
                current.append(h)
        os.environ[var] = ",".join(current)

# The model is loaded once at startup (see ``main``) and shared by every session.
MODEL: Optional[Surflo] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_exc(e: Exception) -> str:
    return f"ERROR: {type(e).__name__}: {e}\n\n{traceback.format_exc()}"


# Inline CSS spinner (class defined in SURFLO_CSS). Prepended to "processing"
# status lines so the browser shows a live animation while the (blocking) decode
# runs -- the animation is client-side, so it keeps spinning without server pings.
_SPINNER = '<span class="surflo-spinner"></span>'


def _compose_infer_cfg(mode: str, preset: str):
    """Hydra-compose the ``infer`` config for a mode/preset (like the CLI)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    overrides = [f"mode={mode}"]
    if mode == "guided":
        overrides.append(f"guided={preset}")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name="infer", overrides=overrides)


def _is_guided_result(result: Optional[Dict[str, Any]]) -> bool:
    """True when ``result`` carries the per-point Gaussians guided runs produce."""
    if not isinstance(result, dict):
        return False
    pts = result.get("points")
    sc = result.get("aux_scales")
    return (
        isinstance(pts, torch.Tensor)
        and isinstance(sc, torch.Tensor)
        and sc.shape[0] == pts.shape[0]
        and pts.shape[0] > 0
    )


def _cull_points_by_opacity(
    result: Dict[str, Any], threshold: Optional[float],
) -> Dict[str, Any]:
    """Return a copy of a guided result keeping only opacity >= threshold Gaussians.

    Used ONLY for the displayed / exported point cloud (its ``points`` /
    ``normals``); meshing and RGB TSDF coloring keep the full, unculled result so
    both see every Gaussian. No-op for a plain result (no per-point opacities) or
    when ``threshold`` is None / <= 0.

    Culls every per-Gaussian tensor (incl. ``aux_means`` / ``aux_colors_sh``) so
    the returned copy stays internally consistent as a standalone guided result.
    """
    if threshold is None or float(threshold) <= 0.0:
        return result
    opac = result.get("aux_opacities")
    if not isinstance(opac, torch.Tensor):
        return result
    # Flatten to a 1-D per-Gaussian mask so it indexes (N, C) tensors whether
    # aux_opacities is stored as (N,) or (N, 1).
    opac = opac.detach().float().reshape(-1)
    total = int(opac.numel())
    if total == 0:
        return result
    keep = opac >= float(threshold)
    out = dict(result)
    for k in ("points", "aux_means", "normals", "colors", "aux_opacities",
              "aux_scales", "aux_quats", "aux_colors", "aux_normals",
              "aux_colors_sh"):
        v = result.get(k)
        if isinstance(v, torch.Tensor) and v.shape[0] == total:
            out[k] = v[keep]
    return out


# --- Export files & their lifetime -------------------------------------------
# Every artifact the UI hands to a Model3D / DownloadButton has to exist on disk
# (those components take a filepath, not bytes), and Gradio then *copies* it into
# its own cache so the browser can fetch it over HTTP. So each export costs two
# files, and a colored mesh GLB runs to hundreds of MB. Three things keep that
# bounded:
#
#   1. `_out_path` memoizes on the caller's per-reconstruction cache dict, so
#      flipping color modes back and forth re-serves one file per mode instead
#      of writing a new one per click.
#   2. Files live under one process-wide root, in a per-session subdirectory that
#      `_cleanup_session` removes when the browser tab closes (see `build_demo`).
#   3. `atexit` drops the whole root, covering sessions whose unload never fired.
#
# Gradio's own copies are expired by the `delete_cache` setting on `gr.Blocks`.
_TMP_ROOT: Optional[Path] = None


def _tmp_root() -> Path:
    """The process-wide directory holding every session's exports."""
    global _TMP_ROOT
    if _TMP_ROOT is None:
        _TMP_ROOT = Path(tempfile.mkdtemp(prefix="surflo_demo_"))
        atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    return _TMP_ROOT


def _session_dir(request: Optional[gr.Request]) -> Path:
    """Per-session export directory, created on first use.

    Sessions without a hash (a direct API call rather than a browser tab) share
    a ``_nosession`` bucket; it is not tied to any tab, so it is only reclaimed
    by the ``atexit`` sweep.
    """
    session = getattr(request, "session_hash", None) if request is not None else None
    d = _tmp_root() / (str(session) if session else "_nosession")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _out_path(cache: Dict[str, Any], key: str, suffix: str,
              request: Optional[gr.Request]) -> Tuple[str, bool]:
    """Path for export ``key``, reused across calls.

    Returns ``(path, is_fresh)``. ``is_fresh`` is True when the caller still has
    to write the file; False means a previous call already wrote it and it is
    still on disk, so re-serving the same path is enough.

    The path is memoized on ``cache`` -- the same per-reconstruction dict that
    already memoizes the colors -- so it is dropped exactly when the colors
    are, i.e. when a new reconstruction resets the state.
    """
    cached = cache.get(key)
    if cached is not None and Path(cached).is_file():
        return cached, False
    path = str(_session_dir(request) / f"surflo{suffix}")
    cache[key] = path
    return path, True


def _cleanup_session(request: Optional[gr.Request] = None) -> None:
    """Remove a session's exports when its browser tab goes away.

    Registered with ``demo.unload``. Gradio dispatches unload events with an
    empty input list and a populated ``gr.Request``, so the session hash is
    available here; the default keeps this callable if that ever changes, in
    which case cleanup falls back to the ``atexit`` sweep.
    """
    session = getattr(request, "session_hash", None) if request is not None else None
    if not session or _TMP_ROOT is None:
        return
    shutil.rmtree(_TMP_ROOT / str(session), ignore_errors=True)


# --- Viewer-only orientation -------------------------------------------------
# The model outputs points in an OpenCV-style world frame (Y down, Z forward),
# so the raw scene shows up upside-down and facing away in both viewers. This
# rotation reorients it to a Y-up, front-facing view for BOTH the Plotly point
# cloud and the Model3D GLB. It is *visualization only* -- saved .ply downloads
# keep the original model frame.
#
# Default: 180 deg about X (flip Y and Z). To tweak the orientation, swap this
# for another proper rotation, e.g.
#   flip Y only:  np.diag([1, -1,  1])
#   flip Z only:  np.diag([1,  1, -1])
#   180 about Y:  np.diag([-1, 1, -1])
_VIEW_ROT = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
# Plotly initial camera, matched to `_VIEW_ROT` (Y up, looking at the front).
_PLOTLY_CAMERA = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.4, y=1.0, z=1.4),
)


def _apply_view_rot(xyz: np.ndarray) -> np.ndarray:
    """Apply the viewer-only reorientation to an ``(N, 3)`` array of coords."""
    return np.ascontiguousarray(np.asarray(xyz, dtype=np.float32) @ _VIEW_ROT.T)


# --- First-camera display frame ----------------------------------------------
# Babylon's viewer (gr.Model3D) frames every loaded model the same way: an
# orbit camera at alpha=90 deg / beta=75 deg, radius = 1.1x the bbox diagonal,
# looking at the bbox center. Target and distance derive from the bounding
# box, and the VIEW DIRECTION is a fixed world-space vector. Instead of
# fighting the camera API (the framing is animated, and overrides race
# against it), the *displayed geometry* is transformed so that this default
# framing IS (approximately) the first reconstruction camera:
#
#   * rotate the scene so camera 0's optical axis / screen-up map onto the
#     default view direction / on-screen up -> exact gaze direction and roll;
#   * translate + scale so the orbit pivot (the bbox center, pinned by
#     symmetric padding vertices in `_mesh_to_glb`) sits on camera 0's axis
#     near the scene center (camera-rig extent, `_scene_extent_for_view`),
#     with unit scene radius;
#   * the default camera then sits ON camera 0's axis. Its distance matches
#     camera 0 exactly when the mesh is compact enough (`_mesh_to_glb`
#     inflates the padding to land it); otherwise the residual error is a
#     pure pull-back along the shared axis (camera 0's view, zoomed out).
#
# The Plotly point cloud reuses the same transform and sets its
# (deterministic) camera to the same pose, so both viewers open alike. All of
# this is display-only: the downloadable .ply files keep original coordinates.
_BAB_BETA = float(np.deg2rad(75.0))
_BAB_RADIUS_FACTOR = 1.1
# Default view direction u (unit, camera -> target) and its on-screen up
# (+Y projected perpendicular to u, normalized: (0, sin(beta), -cos(beta))).
_BAB_VIEW_DIR = np.array([0.0, -np.cos(_BAB_BETA), -np.sin(_BAB_BETA)])
_BAB_SCREEN_UP = np.array([0.0, np.sin(_BAB_BETA), -np.cos(_BAB_BETA)])
# Viewer-frame basis as columns (right, up, view-dir): a proper rotation.
_BAB_FRAME = np.stack(
    [np.cross(_BAB_SCREEN_UP, _BAB_VIEW_DIR), _BAB_SCREEN_UP, _BAB_VIEW_DIR],
    axis=1,
)


def _apply_view_frame(xyz: np.ndarray, frame: Dict[str, Any]) -> np.ndarray:
    """World coords ``(N, 3)`` -> the first-camera display frame (float64)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    out = (xyz - frame["pivot"]) @ np.asarray(frame["rot"]).T / frame["scale"]
    return np.ascontiguousarray(out)


# --- Normal-color softening (mesh) -------------------------------------------
# The mesh's "Normals" mode uses the same world-frame mapping as the point cloud
# (`normals_to_rgb_uint8`), then softens it the way the project page does
# (anttwo.github.io/surflo, `surflo.js`, cache key 'surflo-soft-normals-v1'):
# a small desaturation, a pull toward midgrey, and a slight gamma. Together these
# turn a raw normal map's harsh primaries into pastels.
_NORMAL_DESAT = 0.28         # blend toward luminance
_NORMAL_MIDGREY = 0.55       # the grey every channel is pulled toward
_NORMAL_MIDGREY_MIX = 0.82   # 1.0 = no pull, 0.0 = flat grey
_NORMAL_GAMMA = 0.92         # rounds off the extremes


def _scatter3d_figure(
    points: np.ndarray,
    colors: np.ndarray,
    marker_size: float = 1.5,
    view_frame: Optional[Dict[str, Any]] = None,
):
    """Build an interactive Plotly ``Scatter3d`` figure (pan / zoom / rotate).

    With a ``view_frame`` (see `_first_camera_view_frame`) the displayed points
    are transformed into the first-camera display frame and the camera is set
    to camera 0's pose exactly (Plotly's camera is a deterministic part of the
    figure, so no bbox tricks are needed): symmetric axis ranges ``[-k, k]`` +
    a cubic scene box make the orbit center (``camera.center`` = the middle of
    the ranges) the pivot on camera 0's axis, and the eye sits at camera 0 --
    Plotly eye units span the range box as a unit cube, so a display distance
    ``d`` is ``d / (2 k)``. Purely a display transform; the saved ``.ply``
    keeps the original coordinates.
    """
    import plotly.graph_objects as go

    cols = np.clip(np.asarray(colors, dtype=np.float32).reshape(-1, 3), 0.0, 1.0)
    cols_u8 = (cols * 255.0 + 0.5).astype(np.uint8)
    rgb = [f"rgb({r},{g},{b})" for r, g, b in cols_u8]

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    axis = dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False, title="")
    if view_frame is not None and pts.shape[0] > 0:
        pts = _apply_view_frame(pts, view_frame).astype(np.float32)
        k = max(float(np.abs(pts).max()), 1e-6)
        axis = dict(axis, range=[-k, k])
        # Camera 0 sits at -cam_dist along the view direction from the pivot
        # (the origin); same pose the mesh viewer's default framing aims for.
        eye = (-float(view_frame["cam_dist"]) * _BAB_VIEW_DIR) / (2.0 * k)
        scene_kwargs = dict(
            aspectmode="cube",
            bgcolor="rgb(17,24,39)",
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
            ),
        )
    else:
        pts = _apply_view_rot(pts)
        scene_kwargs = dict(aspectmode="data", bgcolor="rgb(17,24,39)", camera=_PLOTLY_CAMERA)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color=rgb, opacity=1.0),
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        scene=dict(scene_kwargs, xaxis=axis, yaxis=axis, zaxis=axis),
        paper_bgcolor="rgb(17,24,39)",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        # Explicit height: an autosize Plotly figure collapses to zero height in
        # gr.Plot on Gradio >= 6.18 (gradio#13539), which is why the plot only
        # appeared on the second click. A fixed height renders on the first try.
        autosize=False,
        height=520,
    )
    return fig


_GLB_MAGIC = 0x46546C67
_GLB_CHUNK_JSON = 0x4E4F534A

# glTF declares COLOR_0 to be **linear**, and Babylon (which backs `gr.Model3D`)
# gamma-encodes the frame on the way out -- `applyImageProcessing` ends with
# `toGammaSpace`, which without the optional `USE_EXACT_SRGB_CONVERSIONS` engine
# flag (Gradio does not set it) is `pow(x, 1 / 2.2)`. Vertex colors written raw
# therefore come out visibly brightened -- 0.57 would display as 0.78, nearly
# white -- so the GLB writer pre-encodes them to cancel it.
#
# This is a *viewer* correction: it is applied only when building the GLB, so the
# downloadable `.ply` keeps the true, uncompensated colors (verified in Blender).
#
# Both exponents were settled **by eye against the rendered demo**, not derived.
# Babylon's shader source implies 2.2 should cancel its transform exactly (the PBR
# pixel shader always defines FROMLINEARSPACE, and `surfaceAlbedo *= vColor.rgb`
# applies no conversion to COLOR_0), yet in practice RGB looked oversaturated at
# 2.2 and matches the point cloud at 1.8. That gap is unexplained -- it is not the
# surround, since both viewers render on near-identical dark backgrounds (this
# block on `--surflo-panel` #141417, the Plotly cloud on rgb(17, 24, 39)).
# Treat these as calibration constants: re-tune by eye, don't re-derive them.
_VIEWER_GAMMA = 2.2      # softened normal colors
_VIEWER_GAMMA_RGB = 1.8  # TSDF RGB colors


_GLB_CHUNK_BIN = 0x004E4942


def _glb_add_unlit_colors(path: str, colors: np.ndarray) -> None:
    """Rewrite a GLB in place, giving it float vertex colors and an unlit material.

    Two things are wrong with a plain trimesh export for our purposes:

    * trimesh emits **no material** for a vertex-colored mesh, and glTF says a
      primitive without one gets the *default* material -- metallic-roughness with
      ``metallicFactor = 1``, which is fully lit. That is the shading we want gone.
      A material flagged ``KHR_materials_unlit`` makes ``COLOR_0`` pass straight
      through with no lighting term (Blender's Emission shader). The extension is
      understood by Babylon -- which backs ``gr.Model3D`` -- as well as three.js
      and model-viewer, so the GLB looks the same wherever it is opened.
    * trimesh stores ``COLOR_0`` as ``UNSIGNED_BYTE``. After the `_VIEWER_GAMMA`
      pre-encoding that is not enough precision in the shadows: a target of 0.05
      encodes to 0.0015, which rounds to byte 0 -- pure black. We therefore write
      ``COLOR_0`` ourselves as ``FLOAT``.

    The color data is **appended** to the BIN chunk, so every existing bufferView
    offset stays valid and only the JSON chunk needs rebuilding. ``colors`` is
    ``(V, 3)`` float in [0, 1], already gamma pre-encoded by the caller.
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < 12 or struct.unpack_from("<I", blob, 0)[0] != _GLB_MAGIC:
        return  # not a GLB container; leave it alone

    # Split the container into its (type, payload) chunks.
    chunks, off = [], 12
    while off + 8 <= len(blob):
        c_len, c_type = struct.unpack_from("<II", blob, off)
        chunks.append([c_type, blob[off + 8: off + 8 + c_len]])
        off += 8 + c_len

    json_i = next((i for i, c in enumerate(chunks) if c[0] == _GLB_CHUNK_JSON), None)
    bin_i = next((i for i, c in enumerate(chunks) if c[0] == _GLB_CHUNK_BIN), None)
    if json_i is None or bin_i is None:
        return

    doc = json.loads(chunks[json_i][1].decode("utf-8"))

    # --- append COLOR_0 (float RGBA) to the BIN chunk -------------------------
    rgba = np.ones((colors.shape[0], 4), dtype=np.float32)
    rgba[:, :3] = np.clip(colors, 0.0, 1.0)
    bin_payload = chunks[bin_i][1]
    bin_payload += b"\x00" * (-len(bin_payload) % 4)  # keep 4-byte alignment
    color_offset = len(bin_payload)
    bin_payload += rgba.tobytes()
    chunks[bin_i][1] = bin_payload

    doc.setdefault("bufferViews", []).append({
        "buffer": 0,
        "byteOffset": color_offset,
        "byteLength": int(rgba.nbytes),
        "target": 34962,  # ARRAY_BUFFER
    })
    doc.setdefault("accessors", []).append({
        "bufferView": len(doc["bufferViews"]) - 1,
        "componentType": 5126,  # FLOAT
        "count": int(rgba.shape[0]),
        "type": "VEC4",
    })
    color_accessor = len(doc["accessors"]) - 1
    doc.setdefault("buffers", [{}])[0]["byteLength"] = len(bin_payload)

    # --- unlit material ------------------------------------------------------
    used = doc.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in used:
        used.append("KHR_materials_unlit")
    doc.setdefault("materials", []).append({
        "name": "surflo_unlit",
        # Wrapping meshes are not reliably outward-facing, and an unlit material
        # offers no shading cue to hide backfaces behind.
        "doubleSided": True,
        "pbrMetallicRoughness": {
            # White base color: COLOR_0 multiplies it, so the vertex colors are
            # reproduced exactly.
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "extensions": {"KHR_materials_unlit": {}},
    })
    unlit = len(doc["materials"]) - 1

    for m in doc.get("meshes", []):
        for prim in m.get("primitives", []):
            prim["material"] = unlit
            prim.setdefault("attributes", {})["COLOR_0"] = color_accessor

    payload = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    payload += b" " * (-len(payload) % 4)  # chunks are 4-byte aligned
    chunks[json_i][1] = payload

    body = b"".join(struct.pack("<II", len(p), t) + p for t, p in chunks)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", _GLB_MAGIC, 2, 12 + len(body)) + body)


def _mesh_to_glb(
    mesh: Any,
    path: str,
    gamma: float = _VIEWER_GAMMA,
    view_frame: Optional[Dict[str, Any]] = None,
) -> None:
    """Export a Surflo mesh (verts / faces / verts_colors) to GLB via trimesh.

    This GLB is only used for display, not the downloadable ``.ply``. Without
    a ``view_frame`` the vertices are just reoriented with ``_VIEW_ROT``.

    With a ``view_frame`` (see `_first_camera_view_frame`) the displayed copy
    is transformed into the first-camera display frame, and eight
    *unreferenced* (hence invisible) padding vertices are appended at
    ``(+-hx, +-hy, +-hz)``: they make the bounding box symmetric around the
    origin, so its center -- the only orbit target the Babylon viewer behind
    ``gr.Model3D`` supports -- is pinned to the orbit pivot on camera 0's
    axis. Babylon's default camera then sits on that axis at 1.1x the bbox
    diagonal; when camera 0 is farther out than the minimal box allows, the
    padding is inflated so the default camera lands *exactly* at camera 0.
    Purely a display transform; the downloadable ``.ply`` is untouched.

    When the mesh carries vertex colors the GLB is patched to be unlit, so RGB and
    normal colors are shown exactly as computed (see `_glb_add_unlit_colors`). An
    uncolored mesh keeps the default lit material -- unlit with no colors would
    render as a featureless flat silhouette.
    """
    import trimesh

    faces = mesh.faces.detach().cpu().numpy().astype(np.int64)

    n_pad = 0
    if view_frame is not None:
        verts = _apply_view_frame(mesh.verts.detach().cpu().numpy(), view_frame)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        h = np.maximum(np.abs(lo), np.abs(hi))  # [-h, h] box contains the mesh
        h_norm = float(np.linalg.norm(h))
        if h_norm > 0:
            # Default camera distance = 1.1 x the (padded) bbox diagonal; if
            # camera 0 sits farther out, inflate the box to land it exactly.
            min_dist = _BAB_RADIUS_FACTOR * 2.0 * h_norm
            want = float(view_frame["cam_dist"])
            if want > min_dist:
                h = h * (want / min_dist)
            signs = np.array(
                [
                    [sx, sy, sz]
                    for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
                ],
                dtype=np.float64,
            )
            corners = signs * h
            verts = np.concatenate([verts, corners], axis=0)
            n_pad = corners.shape[0]
    else:
        verts = _apply_view_rot(mesh.verts.detach().cpu().numpy()).astype(np.float64)

    # Geometry only: the vertex colors are appended afterwards as FLOAT, which
    # trimesh cannot emit (it quantizes COLOR_0 to bytes).
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tm.export(path)

    vc = getattr(mesh, "verts_colors", None)
    if vc is not None:
        col = vc.detach().cpu().float().clamp(0, 1).numpy()
        if gamma != 1.0:  # display-only pre-encode, see `_VIEWER_GAMMA`
            col = np.power(col, gamma)
        if n_pad:  # COLOR_0 must cover the padding vertices too
            col = np.concatenate([col, np.zeros((n_pad, 3), dtype=col.dtype)], axis=0)
        _glb_add_unlit_colors(path, col)


def _scene_extent_for_view(
    scene: Optional[SceneState], result: Optional[Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Scene center (world, ``(3,)``) + radius from the CAMERA RIG.

    Uses `get_cameras_spatial_extent` on the scene's cameras: the rig is
    compact around the captured subject, unlike the mesh bounding box, which
    distant background geometry stretches arbitrarily. Returns ``(None, None)``
    when the cameras are unavailable; the mesh is then shown un-normalized
    with Babylon's default framing.
    """
    try:
        from surflo.structures.cameras import get_cameras_spatial_extent
        from surflo.utils.guided_result import (
            build_refined_cameras_from_guided_result,
        )

        cameras = build_refined_cameras_from_guided_result(
            scene.batch,
            scene.scene_idx,
            result if isinstance(result, dict) else {},
            apply_camera_correction=False,
            data_device="cpu",
        )
        ext = get_cameras_spatial_extent(cameras)
        center = ext["avg_cam_center"].detach().cpu().float().numpy().reshape(3)
        radius = float(ext["radius"])
    except Exception:  # noqa: BLE001
        return None, None
    if not np.isfinite(radius) or radius <= 0 or not np.all(np.isfinite(center)):
        return None, None
    return center, radius


# The demo tags each encoded scene with whether camera 0's SOURCE image was
# portrait -- i.e. whether ``load_and_preprocess_images(rotate_portrait=True)``
# rotated it 90 deg to landscape before VGGT. That fact is not recoverable from
# the (already-rotated) batch, so it is stashed on the SceneState at encode
# time and read back here. Viewer-only; it never changes the saved geometry.
_FIRST_PORTRAIT_ATTR = "_surflo_demo_first_portrait"


def _first_image_is_portrait(path: Optional[str]) -> bool:
    """True if the image file is portrait (height > width) -- the same test
    ``load_and_preprocess_images`` uses to decide the inference rotation."""
    if not path:
        return False
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.height) > int(img.width)
    except Exception:  # noqa: BLE001
        return False


def _first_camera_view_frame(
    scene: Optional[SceneState], result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Display transform aligning the viewers' default framing with camera 0.

    See the `_BAB_*` block above for the strategy. Returns::

        {"rot":      (3, 3) world -> display rotation (subsumes _VIEW_ROT),
         "pivot":    (3,) world-space orbit pivot, on camera 0's optical axis,
         "scale":    scene radius (camera-rig extent),
         "cam_dist": camera 0 -> pivot distance, in display (scene-radius) units}

    or None when the pose / extent is unavailable (callers then fall back to
    the plain `_VIEW_ROT` display path).
    """
    center, radius = _scene_extent_for_view(scene, result)
    if center is None:
        return None
    try:
        ext = scene.extrinsics[0].detach().cpu().float().numpy()  # (3, 4) w2c
    except Exception:  # noqa: BLE001
        return None
    R = ext[:, :3].astype(np.float64)
    c0 = -R.T @ ext[:, 3].astype(np.float64)
    fwd = R[2, :].copy()          # camera 0 optical axis (+Z row) in world
    n = float(np.linalg.norm(fwd))
    if not np.isfinite(n) or n < 1e-9:
        return None
    fwd /= n
    # "Up" reference in world used to roll the view. Normally camera 0's up axis
    # (-y; OpenCV y points down). But if camera 0's source image was portrait, it
    # was rotated 90 deg CCW to landscape before VGGT, so the scene's true up now
    # points along camera 0's LEFT axis (-x): use that to roll the displayed
    # scene back upright. (Empirically, ROTATE_90 sends the image top to the left
    # edge; if a portrait scene ever shows up upside-down, flip -R[0] to R[0].)
    if getattr(scene, _FIRST_PORTRAIT_ATTR, False):
        up0 = -R[0, :]            # camera 0's left axis == the original image's up
    else:
        up0 = -R[1, :]            # camera 0's up axis
    up0 = up0 - np.dot(up0, fwd) * fwd  # re-orthogonalize (R is a prediction)
    n = float(np.linalg.norm(up0))
    if n < 1e-9:
        return None
    up0 /= n
    # Proper rotation mapping camera 0's (right, up, forward) onto _BAB_FRAME.
    cam_frame = np.stack([np.cross(up0, fwd), up0, fwd], axis=1)
    rot = _BAB_FRAME @ cam_frame.T

    # Orbit pivot: the point of camera 0's axis closest to the scene center
    # (best orbiting feel while keeping the default camera on the axis).
    depth = float(np.dot(np.asarray(center, np.float64) - c0, fwd))
    depth = max(depth, 0.05 * float(radius))  # scene center ~behind the camera
    if not np.isfinite(depth):
        return None
    return {
        "rot": rot,
        "pivot": c0 + depth * fwd,
        "scale": float(radius),
        "cam_dist": depth / float(radius),
    }


# ---------------------------------------------------------------------------
# Global-state portrait
# ---------------------------------------------------------------------------
def _finalize_state_img(rgb: torch.Tensor, upscale: int = 1) -> np.ndarray:
    """``(H, W, 3)`` float ``[0, 1]`` tensor -> crisp uint8 numpy for gr.Image."""
    arr = (rgb.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
    if upscale > 1:  # nearest-neighbour keeps the token grid sharp when upsized
        arr = np.kron(arr, np.ones((upscale, upscale, 1), dtype=np.uint8))
    return arr


def get_state_image(scene) -> np.ndarray:
    """Bicolored heatmap of the global state ``(K, D)`` (teal = +, lavender = -)."""
    global_state = scene.global_state.clone().float()  # (1, K, D) bf16 -> f32
    if global_state.ndim == 3:
        global_state = global_state[0]
    mean_val, std_val = global_state.mean(), global_state.std()
    global_state = mean_val + (global_state - mean_val).clamp(-3 * std_val, 3 * std_val)
    global_state = global_state.sign() * torch.log1p(global_state.abs())
    rng = (global_state.max() - global_state.min()).clamp_min(1e-8)
    global_state = (global_state - global_state.min()) / rng
    color1 = torch.tensor([94 / 255, 224 / 255, 214 / 255], device=global_state.device).view(1, 1, 3)
    color2 = torch.tensor([192 / 255, 144 / 255, 240 / 255], device=global_state.device).view(1, 1, 3)
    interp = 2 * global_state.unsqueeze(-1) - 1
    rgb = color1 * interp.clamp_min(0) - color2 * interp.clamp_max(0)
    return _finalize_state_img(rgb, upscale=1)  # already 512 px wide


def state_viz_cb(scene: Optional[SceneState]):
    """Render the global-state portrait after encoding."""
    if scene is None:
        return None
    try:
        return get_state_image(scene)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def preview_images_cb(paths: Optional[List[str]]):
    return paths or None


def _sample_dir_images(
    image_dir: str, n_images: Optional[int], sampling: str, seed: int = 42,
) -> List[str]:
    """List + subsample a server-side folder exactly like the folder loader.

    Reuses the same ``list_images_in_folder`` + ``sample_image_indices`` helpers
    that ``Surflo.encode(folder, ...)`` calls, so the dev-mode gallery preview
    shows precisely the frames that will be encoded. ``n_images`` <= 0 means all.
    """
    from surflo.data.image_folder import list_images_in_folder, sample_image_indices

    paths = list_images_in_folder(image_dir)
    if not paths:
        return []
    n = int(n_images) if n_images and int(n_images) > 0 else len(paths)
    idxs = sample_image_indices(len(paths), n, sampling, int(seed))
    return [paths[i] for i in idxs]


def list_dir_cb(image_dir: Optional[str], n_images, sampling: str) -> str:
    """Dev mode: *list* (as text) the frames that will be encoded.

    Deliberately returns only a text summary -- never image data -- so nothing
    is copied into Gradio's cache (server-side paths outside the app dir aren't
    even allowed there). Encoding reads these files straight from disk as tensors.
    """
    image_dir = (image_dir or "").strip()
    if not image_dir:
        return "Enter an image directory (a path on the server)."
    if not os.path.isdir(image_dir):
        return f"Not a directory on the server: `{image_dir}`"
    try:
        sampled = _sample_dir_images(image_dir, n_images, sampling)
    except Exception as e:  # noqa: BLE001
        return _fmt_exc(e)
    if not sampled:
        return f"No JPG/PNG images found in `{image_dir}`."
    names = [os.path.basename(p) for p in sampled]
    shown = ", ".join(names[:12]) + (" …" if len(names) > 12 else "")
    return (
        f"**{len(sampled)}** image(s) selected from `{image_dir}` "
        f"(sampling: {sampling}) — {shown}. Click **Encode** to continue."
    )


def encode_dir_cb(image_dir: Optional[str], n_images, sampling: str):
    """Dev-mode encode: read images from a server-side folder, then encode.

    Same output signature (and streamed spinner) as :func:`encode_cb`; only the
    inputs differ (a directory + sampling instead of uploaded files).
    """
    if MODEL is None:
        yield None, "Model is not loaded (check the server logs)."
        return
    image_dir = (image_dir or "").strip()
    if not image_dir:
        yield None, "Enter an image directory (a path on the server)."
        return
    if not os.path.isdir(image_dir):
        yield None, f"Not a directory on the server: {image_dir!r}"
        return
    try:
        sampled = _sample_dir_images(image_dir, n_images, sampling)
    except Exception as e:  # noqa: BLE001
        yield None, _fmt_exc(e)
        return
    if not sampled:
        yield None, f"No JPG/PNG images found in {image_dir!r}."
        return
    yield None, f"{_SPINNER}Loading and encoding {len(sampled)} image(s) from {image_dir}…"
    try:
        cr = float(DEFAULT_CULL_RADIUS) if DEFAULT_CULL_RADIUS and DEFAULT_CULL_RADIUS > 0 else None
        n = int(n_images) if n_images and int(n_images) > 0 else None
        # Pass the folder (not the sampled list) so encoding goes through the
        # exact CLI code path (build_image_folder_batch); it re-selects the same
        # frames because _sample_dir_images shares its sampler + seed.
        scene = MODEL.encode(
            image_dir, n_images=n, image_sampling=sampling, cull_radius=cr,
        )
        # Tag the scene for the viewer-only portrait-rotation compensation
        # (camera 0 = the first sampled frame; see `_first_camera_view_frame`).
        setattr(scene, _FIRST_PORTRAIT_ATTR, _first_image_is_portrait(sampled[0]))
        nviews = int(scene.images.shape[0])
        gs = tuple(scene.global_state.shape[1:])
        yield scene, (
            f"Encoded {nviews} view(s) into a global state with size {gs}. "
            f"Ready to reconstruct."
        )
    except Exception as e:  # noqa: BLE001
        yield None, _fmt_exc(e)


def encode_cb(paths):
    if MODEL is None:
        yield None, "Model is not loaded (check the server logs)."
        return
    if not paths:
        yield None, "Select some images first (step 1)."
        return
    paths = [p for p in paths]
    if DEFAULT_MAX_VIEWS and DEFAULT_MAX_VIEWS > 0:
        paths = paths[: int(DEFAULT_MAX_VIEWS)]
    # Streamed immediately so the spinner animates client-side while the (blocking)
    # VGGT encode + global-state compression runs.
    yield None, f"{_SPINNER}Encoding {len(paths)} image(s) into the global state…"
    try:
        cr = float(DEFAULT_CULL_RADIUS) if DEFAULT_CULL_RADIUS and DEFAULT_CULL_RADIUS > 0 else None
        scene = MODEL.encode(paths, cull_radius=cr)
        # Tag the scene for the viewer-only portrait-rotation compensation
        # (camera 0 = the first selected image; see `_first_camera_view_frame`).
        setattr(scene, _FIRST_PORTRAIT_ATTR, _first_image_is_portrait(paths[0]))
        n = int(scene.images.shape[0])
        gs = tuple(scene.global_state.shape[1:])
        yield scene, (
            f"Encoded {n} view(s) into a global state with size {gs}. "
            f"Ready to reconstruct."
        )
    except Exception as e:  # noqa: BLE001
        yield None, _fmt_exc(e)


def reconstruct_cb(
    scene: Optional[SceneState],
    mode: str,
    preset: str,
    num_query_points: int,
    seed: int,
):
    if scene is None:
        yield None, None, "Encode some images first (step 1)."
        return
    # Streamed as soon as the button is clicked so the user gets immediate
    # feedback while the (potentially long) decode runs as a single blocking call.
    tag = mode if mode != "guided" else f"guided={preset}"
    try:
        set_global_seeds(int(seed))
        meta = {"mode": mode, "preset": preset}
        if mode == "plain":
            yield None, None, f"{_SPINNER}Decoding [{tag}]…"
            # Timer starts at the first ODE step: plain has no expert / heavy
            # preprocessing, so wrapping the solve is already exactly that.
            t0 = time.perf_counter()
            result = scene.reconstruct(
                mode="plain",
                num_query_points=int(num_query_points),
                num_steps=int(DEFAULT_NUM_STEPS),
                seed=int(seed),
            )
        else:
            cfg = _compose_infer_cfg(mode, preset)
            block = cfg.guided
            # Compute the DepthAnything-3 monodepth / normal priors *before* the
            # timer starts and cache them on the scene, so the reported duration
            # covers only the ODE solve -- not the (heavy) expert depth pass.
            want_mono = bool(block.get("use_monodepth_guidance", False))
            want_norm = bool(block.get("use_normal_guidance", False))
            if want_mono or want_norm:
                yield None, None, (
                    f"{_SPINNER}Computing expert priors for guidance…"
                )
                scene.compute_guidance_experts(
                    expert_cfg=cfg.expert,
                    compute_monodepths=want_mono,
                    compute_normals=want_norm,
                )
            yield None, None, (
                f"{_SPINNER}Decoding [{tag}]… This can take a while for guided modes."
            )
            t0 = time.perf_counter()
            result = scene.reconstruct(
                mode=mode,
                config_block=block,
                expert_cfg=cfg.expert,
                # Keep ALL Gaussians (no opacity cull) so mesh extraction sees
                # every Gaussian, matching scripts/infer.py (mesh always uses all
                # Gaussians). The point cloud is culled to opacity >=
                # DEFAULT_OPACITY_THRESHOLD later, at display/export time.
                opacity_threshold=None,
                num_query_points=int(num_query_points),
            )
        elapsed = time.perf_counter() - t0
        n_pts = int(result["points"].shape[0])
        guided = _is_guided_result(result)
        note = "" if guided else " (plain -> no Gaussians; RGB color / mesh disabled)"
        yield result, meta, f"Reconstructed {n_pts} points [{tag}] in {elapsed:.1f}s.{note}"
    except Exception as e:  # noqa: BLE001
        yield None, None, _fmt_exc(e)


def _color_export_points(
    scene: SceneState,
    cache: Dict[str, Any],
    result: Dict[str, Any],
    color_mode: str,
    request: Optional[gr.Request] = None,
):
    """(Re)color the cloud, save the full-res PLY, and build the interactive plot.

    Per-mode colors (RGB TSDF / normals) are cached on ``cache`` so flipping
    between color modes only recomputes the first time each mode is requested.
    The exported PLY is memoized alongside them, so flipping back to a mode
    already exported re-serves that file instead of writing a new one.
    """
    from surflo.utils.io import normals_to_rgb_uint8

    # The reconstruction result is kept unculled (all Gaussians) so meshing sees
    # every Gaussian; the point cloud, however, only shows / exports the
    # opacity >= DEFAULT_OPACITY_THRESHOLD subset (matches the CLI's
    # point_cloud_{normals,rgb}.ply). Culling a plain result is a no-op.
    pc_result = _cull_points_by_opacity(result, DEFAULT_OPACITY_THRESHOLD)
    points = pc_result["points"].detach()

    if color_mode == "RGB":
        if not _is_guided_result(result):
            return cache, gr.update(), gr.update(), (
                "RGB coloring needs a guided result (Gaussians). "
                "Use a guided run, or pick 'Normals'."
            )
        # Color only the filtered points, but render the TSDF depth from the full
        # (unculled) result so occlusion reflects every Gaussian, matching the
        # CLI's point_cloud_rgb.ply.
        if cache.get("rgb") is None:
            cache["rgb"] = scene.color_points(points, result).detach().cpu().float().numpy()
        col_np = cache["rgb"]
        full_path, fresh = _out_path(
            cache, "path_rgb", f"_points_{color_mode.lower()}.ply", request,
        )
        if fresh:
            save_ply(pc_result, full_path, colors=torch.from_numpy(col_np))
    else:  # Normals
        if cache.get("normals") is None:
            normals = pc_result.get("normals")
            if isinstance(normals, torch.Tensor):
                cache["normals"] = normals_to_rgb_uint8(normals.detach()).astype(np.float32) / 255.0
            else:
                cache["normals"] = np.full((points.shape[0], 3), 0.7, dtype=np.float32)
        col_np = cache["normals"]
        full_path, fresh = _out_path(
            cache, "path_normals", f"_points_{color_mode.lower()}.ply", request,
        )
        if fresh:
            save_ply(pc_result, full_path, color="normals")

    pts_np = points.cpu().float().numpy()
    # Plot every point (no subsampling); the saved PLY is also full-res.
    # Viewer-only framing, matching the mesh viewer: open with (approximately)
    # camera 0's view (the PLY saved above keeps the original coordinates).
    frame = _first_camera_view_frame(scene, result)
    fig = _scatter3d_figure(pts_np, col_np, view_frame=frame)
    msg = (
        f"Point cloud: {int(pts_np.shape[0])} points "
        f"(opacity ≥ {DEFAULT_OPACITY_THRESHOLD:g}), {color_mode} colors "
        f"(drag to rotate, scroll to zoom)."
    )
    return (
        cache,
        gr.update(value=fig),
        gr.update(value=full_path, visible=True),
        msg,
    )


def visualize_pointcloud_cb(
    scene: Optional[SceneState],
    cache: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    color_mode: str,
    request: Optional[gr.Request] = None,
):
    """Show / save the cloud (button *and* the RGB/Normals radio share this)."""
    if scene is None or not isinstance(result, dict):
        return None, gr.update(), gr.update(), "Run a reconstruction first (step 2)."
    if not isinstance(cache, dict):
        cache = {"rgb": None, "normals": None}
    try:
        return _color_export_points(scene, cache, result, color_mode, request)
    except Exception as e:  # noqa: BLE001
        return cache, None, None, _fmt_exc(e)


def _mesh_normal_colors(mesh: Any) -> torch.Tensor:
    """Per-vertex normal-map RGB (float ``[0, 1]``), softened.

    Uses `normals_to_rgb_uint8` -- the exact mapping the point cloud uses, on
    world-frame normals -- so both views share one palette, then applies the
    project page's softening (see the ``_NORMAL_*`` block above) so the colors
    read as pastels rather than harsh primaries.
    """
    from surflo.utils.io import normals_to_rgb_uint8

    u8 = normals_to_rgb_uint8(mesh.vertex_normals.detach().cpu())  # (V, 3) uint8
    rgb = u8.astype(np.float32) / 255.0

    gray = rgb.mean(axis=1, keepdims=True)
    rgb = rgb + (gray - rgb) * _NORMAL_DESAT                   # desaturate
    rgb = _NORMAL_MIDGREY + (rgb - _NORMAL_MIDGREY) * _NORMAL_MIDGREY_MIX
    rgb = np.power(np.clip(rgb, 0.0, 1.0), _NORMAL_GAMMA)      # soften extremes
    return torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))


def _color_export_mesh(
    scene: SceneState,
    cache: Dict[str, Any],
    result: Optional[Dict[str, Any]],
    color_mode: str,
    request: Optional[gr.Request] = None,
):
    """(Re)color the *cached* mesh and re-export -- never re-extracts geometry.

    RGB TSDF colors and normal colors are both cached on ``cache`` so flipping
    between color modes only recomputes the first time each is requested. The
    exported .ply and .glb are memoized the same way: flipping back to a mode
    already exported re-serves those files rather than rewriting them, which
    matters here because a colored GLB can run to hundreds of MB.
    """
    mesh = cache["mesh"]
    n_v, n_f = int(mesh.verts.shape[0]), int(mesh.faces.shape[0])

    if color_mode == "RGB (TSDF)":
        if not isinstance(result, dict) or not _is_guided_result(result):
            return cache, None, None, "RGB (TSDF) coloring needs a guided result."
        if cache.get("rgb") is None:
            cache["rgb"] = scene.color_mesh(mesh, result, assign_to_mesh=False).detach()
        mesh.verts_colors = cache["rgb"]
    elif color_mode == "Normals":
        if cache.get("normals") is None:
            cache["normals"] = _mesh_normal_colors(mesh)
        mesh.verts_colors = cache["normals"]
    else:  # "None"
        mesh.verts_colors = None

    tag = color_mode.split()[0].lower()
    ply_path, ply_fresh = _out_path(cache, f"path_ply_{tag}", f"_mesh_{tag}.ply", request)
    if ply_fresh:
        save_mesh(mesh, ply_path)

    # Viewer-only framing: transform the displayed GLB so Babylon's default
    # framing reproduces camera 0's view (see `_first_camera_view_frame` /
    # `_mesh_to_glb`); the downloadable .ply above is untouched.
    # Recomputed even when the GLB is reused: `mesh_update` below reads
    # `cam_dist` from it on every call.
    frame = _first_camera_view_frame(scene, result)
    view_path, glb_fresh = _out_path(cache, f"path_glb_{tag}", f"_mesh_{tag}.glb", request)
    if glb_fresh:
        # RGB is on raw colors while we pin down the viewer's real transform; the
        # softened normal colors keep the pre-encode (see `_VIEWER_GAMMA`).
        _mesh_to_glb(
            mesh, view_path,
            gamma=_VIEWER_GAMMA_RGB if color_mode == "RGB (TSDF)" else _VIEWER_GAMMA,
            view_frame=frame,
        )
    if frame is not None:
        # Best-effort radius override: after the rotation the default orbit
        # angles already look along camera 0's axis, so -- if the viewer
        # honors it -- this slides the camera to camera 0 exactly, cancelling
        # any pull-back left by the geometry (see `_mesh_to_glb`). Harmless
        # when the geometry already landed it (same value), or if ignored.
        mesh_update = gr.update(
            value=view_path,
            camera_position=(None, None, float(frame["cam_dist"])),
        )
    else:
        mesh_update = gr.update(value=view_path)

    return (
        cache,
        mesh_update,
        gr.update(value=ply_path, visible=True),
        f"Mesh: V={n_v}, F={n_f}, color={color_mode}.",
    )


def extract_mesh_cb(
    scene: Optional[SceneState],
    result: Optional[Dict[str, Any]],
    meta: Optional[Dict[str, Any]],
    mesh_color_mode: str,
    request: Optional[gr.Request] = None,
):
    """Extract the mesh once, cache its geometry, then color + export."""
    if scene is None or not isinstance(result, dict):
        return None, None, None, "Run a reconstruction first (step 2)."
    if not _is_guided_result(result):
        return None, None, None, (
            "Meshing needs a guided result (Gaussians). Run a "
            "guided reconstruction first."
        )
    try:
        preset = (meta or {}).get("preset", DEFAULT_GUIDED_PRESET)
        mode = (meta or {}).get("mode", "guided")
        cfg = _compose_infer_cfg(mode, preset)
        mesh = scene.extract_mesh(result, mesh_cfg=cfg.mesh)
        if int(mesh.verts.shape[0]) == 0 or int(mesh.faces.shape[0]) == 0:
            return None, None, None, "Mesh extraction produced an empty mesh."
        cache = {"mesh": mesh, "rgb": None, "normals": None}
        return _color_export_mesh(scene, cache, result, mesh_color_mode, request)
    except Exception as e:  # noqa: BLE001
        return None, None, None, _fmt_exc(e)


def recolor_mesh_cb(
    scene: Optional[SceneState],
    cache: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    color_mode: str,
    request: Optional[gr.Request] = None,
):
    """Switch mesh color mode without re-extracting the geometry."""
    if not isinstance(cache, dict) or cache.get("mesh") is None:
        return None, None, None, "Extract a mesh first (step 4)."
    try:
        return _color_export_mesh(scene, cache, result, color_mode, request)
    except Exception as e:  # noqa: BLE001
        return cache, None, None, _fmt_exc(e)


def _reset_downstream():
    """After a new reconstruction: drop cached colorings, clear the (always-on)
    viewers, and hide the download buttons until content is regenerated."""
    clear_view = gr.update(value=None)
    hide_dl = gr.update(value=None, visible=False)
    return None, None, clear_view, hide_dl, clear_view, hide_dl


def on_mode_change(mode: str):
    """Show the guided preset picker only when it applies."""
    return gr.update(visible=(mode == "guided"))


# ---------------------------------------------------------------------------
# Look & feel -- matches the project page https://anttwo.github.io/surflo/
# (Roboto, near-black bg #0a0a0a, teal accent #5ee0d6, pill buttons, thin
# headings, uppercase wide-tracked section labels).
# ---------------------------------------------------------------------------
def _surflo_theme():
    """Minimal, version-tolerant Gradio theme: Roboto + teal on dark neutrals.

    Exact colors are enforced in ``SURFLO_CSS`` (robust across Gradio versions);
    the theme only needs to supply the font + a coherent dark base.
    """
    try:
        fonts = [gr.themes.GoogleFont("Roboto"), "system-ui", "sans-serif"]
    except Exception:  # noqa: BLE001
        fonts = ["Roboto", "system-ui", "sans-serif"]
    for kwargs in (
        dict(primary_hue="teal", secondary_hue="purple", neutral_hue="slate", font=fonts),
        dict(primary_hue="teal", neutral_hue="slate"),
        {},
    ):
        try:
            return gr.themes.Base(**kwargs)
        except Exception:  # noqa: BLE001
            continue
    return None


SURFLO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;1,300&display=swap');

.gradio-container, .gradio-container.dark {
  --surflo-bg: #0a0a0a;
  --surflo-panel: #141417;
  --surflo-panel-2: #101012;
  --surflo-text: #e6e6e6;
  --surflo-subdued: #8a8a8a;
  --surflo-accent: #5ee0d6;
  --surflo-accent-hover: #7ff4ea;
  --surflo-hairline: rgba(94, 224, 214, 0.16);
  --surflo-hairline-strong: rgba(94, 224, 214, 0.38);
  --surflo-lav: #b89cff;
  /* re-point Gradio's own theme variables at the Surflo palette */
  --body-background-fill: var(--surflo-bg);
  --background-fill-primary: var(--surflo-panel);
  --background-fill-secondary: var(--surflo-panel-2);
  --block-background-fill: var(--surflo-panel);
  --body-text-color: var(--surflo-text);
  --body-text-color-subdued: var(--surflo-subdued);
  --border-color-primary: var(--surflo-hairline);
  --color-accent: var(--surflo-accent);
  --color-accent-soft: rgba(94, 224, 214, 0.12);
  --link-text-color: var(--surflo-accent);
  --link-text-color-hover: var(--surflo-accent-hover);
  --slider-color: var(--surflo-accent);
  /* Radio / checkbox choices: transparent when unselected, teal when selected
     (Gradio's defaults paint an opaque grey pill on every option). */
  --checkbox-label-background-fill: transparent;
  --checkbox-label-background-fill-hover: rgba(94, 224, 214, 0.10);
  --checkbox-label-background-fill-focus: transparent;
  --checkbox-label-background-fill-selected: rgba(94, 224, 214, 0.14);
  /* Dropdowns read `--input-background-fill`; point it at the dark input color
     so they match the text fields instead of showing a grey box. */
  --input-background-fill: #0e0e10;
  --input-background-fill-focus: #0e0e10;
}

.gradio-container {
  font-family: 'Roboto', system-ui, sans-serif !important;
  color: var(--surflo-text) !important;
  background:
    radial-gradient(1100px 620px at 84% -10%, rgba(94, 224, 214, 0.10), transparent 60%),
    radial-gradient(900px 520px at 6% -4%, rgba(184, 156, 255, 0.08), transparent 55%),
    var(--surflo-bg) !important;
}

.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  font-family: 'Roboto', sans-serif !important;
  font-weight: 100 !important;
  letter-spacing: -0.02em !important;
  color: var(--surflo-text) !important;
}

/* ---- hero header ---- */
#surflo-hero, #surflo-hero .block, #surflo-hero.block {
  border: none !important; background: transparent !important; box-shadow: none !important;
}
#surflo-hero .surflo-eyebrow {
  text-transform: uppercase; letter-spacing: 0.34em; font-size: 12px; font-weight: 400;
  color: var(--surflo-accent);
}
#surflo-hero .surflo-title {
  font-weight: 100; letter-spacing: -0.03em; line-height: 1.0;
  font-size: clamp(40px, 7vw, 72px); margin: 6px 0 6px;
}
#surflo-hero .surflo-title .accent { color: var(--surflo-accent); }
#surflo-hero .surflo-sub {
  color: var(--surflo-subdued); font-weight: 300; font-size: 15px; max-width: 70ch; line-height: 1.5;
}
#surflo-hero .surflo-sub code { color: var(--surflo-accent) !important; background: transparent !important; }
#surflo-hero .surflo-pipeline { margin-top: 14px; }
#surflo-hero .surflo-pill {
  display: inline-block; margin: 5px 8px 0 0; padding: 5px 14px; border-radius: 999px;
  border: 1px solid var(--surflo-hairline); color: var(--surflo-text);
  font-size: 12px; letter-spacing: 0.08em; background: rgba(94, 224, 214, 0.05);
}

/* ---- section cards (accordions) ---- */
.gradio-container .surflo-step {
  background: var(--surflo-panel) !important;
  border: 1px solid var(--surflo-hairline) !important;
  border-radius: 18px !important;
  overflow: hidden;
}
.gradio-container .label-wrap, .gradio-container .label-wrap > span {
  text-transform: uppercase; letter-spacing: 0.2em; font-weight: 500 !important;
  color: var(--surflo-accent) !important;
}

/* ---- inner blocks ---- */
.gradio-container .block {
  background: var(--surflo-panel-2) !important;
  border: 1px solid var(--surflo-hairline) !important;
  border-radius: 12px !important;
}
/* Markdown captions & status lines: plain text within the section, without the
   panel background / hairline capsule that the generic .block rule adds. */
.gradio-container .surflo-note {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.gradio-container .surflo-note .prose { background: transparent !important; }
/* Per-preset cost bullets in the step-2 caption. Gradio's default list spacing
   is loose enough that the summary would dominate the section, so tighten the
   margins and pull the bullets in under the paragraph above them. */
.gradio-container .surflo-perf ul {
  margin: 6px 0 6px 0 !important;
  padding-left: 1.2em !important;
}
.gradio-container .surflo-perf li {
  margin: 0 !important;
  line-height: 1.5 !important;
}
.gradio-container .surflo-perf code { padding: 0 3px !important; }
/* Inline "processing" spinner. The keyframes live in this global stylesheet, so
   the ring keeps rotating client-side even while the server is blocked inside a
   decode call (the status Markdown just emits a <span class="surflo-spinner">). */
.gradio-container .surflo-spinner {
  display: inline-block;
  width: 0.9em; height: 0.9em;
  margin-right: 0.5em;
  vertical-align: -0.12em;
  border: 2px solid rgba(94, 224, 214, 0.25);
  border-top-color: var(--surflo-accent);
  border-radius: 50%;
  animation: surflo-spin 0.8s linear infinite;
}
@keyframes surflo-spin { to { transform: rotate(360deg); } }
.gradio-container label, .gradio-container .block-label,
.gradio-container span[data-testid="block-info"] {
  color: var(--surflo-subdued) !important; letter-spacing: 0.03em;
}

/* ---- buttons: pill + uppercase ---- */
.gradio-container button {
  border-radius: 999px !important;
  font-family: 'Roboto', sans-serif !important;
  text-transform: uppercase; letter-spacing: 0.12em; font-weight: 500 !important;
  transition: all 220ms cubic-bezier(.2, .8, .2, 1);
}
.gradio-container button.primary, .gradio-container .primary {
  background: var(--surflo-accent) !important; color: #06231f !important;
  border: 1px solid var(--surflo-accent) !important;
  box-shadow: 0 10px 30px -12px rgba(94, 224, 214, 0.55);
}
.gradio-container button.primary:hover, .gradio-container .primary:hover {
  background: var(--surflo-accent-hover) !important;
  box-shadow: 0 12px 38px -10px rgba(94, 224, 214, 0.7); transform: translateY(-1px);
}
.gradio-container button.secondary, .gradio-container .secondary {
  background: #1b1b1f !important; color: var(--surflo-text) !important;
  border: 1px solid var(--surflo-hairline) !important;
}
.gradio-container button.secondary:hover {
  border-color: var(--surflo-hairline-strong) !important; color: var(--surflo-accent) !important;
}
/* Primary section action: a full-width pill anchored at the bottom of a card,
   instead of a small button floating on the right of an input row. */
.gradio-container .surflo-cta { margin-top: 10px !important; }
.gradio-container .surflo-cta button {
  width: 100% !important;
  padding: 13px 28px !important;
  font-size: 0.98rem !important;
  letter-spacing: 0.16em !important;
}

/* ---- inputs / sliders / choices ---- */
/* Text-like inputs only: never restyle radio/checkbox/range here, or the native
   checked fill (the inner dot / tick) gets painted over and only the ring shows. */
.gradio-container input:not([type="radio"]):not([type="checkbox"]):not([type="range"]),
.gradio-container textarea, .gradio-container select {
  background: #0e0e10 !important; color: var(--surflo-text) !important; border-radius: 10px !important;
}
.gradio-container input:not([type="radio"]):not([type="checkbox"]):not([type="range"]):focus,
.gradio-container textarea:focus {
  border-color: var(--surflo-accent) !important;
  box-shadow: 0 0 0 2px rgba(94, 224, 214, 0.25) !important;
}
/* Slider track uses the accent color (native rendering is fine here). */
.gradio-container input[type="range"] { accent-color: var(--surflo-accent) !important; }

/* Custom radio / checkbox: Gradio renders these with appearance:none, so the
   browser draws no dot/tick and `accent-color` is a no-op. Draw them ourselves
   (radial-gradient disk for the radio, SVG tick for the checkbox) so the
   checked state always shows a light-cyan fill. */
.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"] {
  -webkit-appearance: none !important;
  appearance: none !important;
  width: 18px !important; height: 18px !important;
  min-width: 18px !important; flex: 0 0 auto !important;
  border: 2px solid var(--surflo-accent) !important;
  background: transparent !important;
  box-shadow: none !important;
  cursor: pointer;
}
.gradio-container input[type="radio"] { border-radius: 999px !important; }
.gradio-container input[type="checkbox"] { border-radius: 5px !important; }
.gradio-container input[type="radio"]:checked {
  background: radial-gradient(circle at 50% 50%,
    var(--surflo-accent) 0 42%, transparent 48%) !important;
  border-color: var(--surflo-accent) !important;
}
.gradio-container input[type="checkbox"]:checked {
  background-color: var(--surflo-accent) !important;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath d='M3.5 8.5l3 3 6-7' fill='none' stroke='%2306231f' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
  background-size: 12px 12px !important;
  background-position: center !important;
  background-repeat: no-repeat !important;
  border-color: var(--surflo-accent) !important;
}
.gradio-container .wrap label.selected, .gradio-container label.selected {
  background: rgba(94, 224, 214, 0.14) !important;
  border-color: var(--surflo-accent) !important; color: var(--surflo-accent) !important;
}

/* ---- links / code / scrollbar ---- */
.gradio-container a { color: var(--surflo-accent) !important; }
.gradio-container a:hover { color: var(--surflo-accent-hover) !important; }
.gradio-container code, .gradio-container pre {
  background: #0e0e10 !important; color: var(--surflo-accent) !important; border-radius: 8px;
}
.gradio-container ::-webkit-scrollbar { width: 10px; height: 10px; }
.gradio-container ::-webkit-scrollbar-thumb { background: rgba(94, 224, 214, 0.25); border-radius: 999px; }
.gradio-container footer { display: none !important; }

/* Latent-state portrait: stretch to the full box width (pixelated on upscale is
   fine). Gradio's ImagePreview centers the image at native size via
   `.image-frame{width:auto}` + `object-fit:scale-down` (which never upscales),
   so we force the frame full-width and let the <img> scale up to fill it. */
.gradio-container .surflo-crisp .image-container,
.gradio-container .surflo-crisp .image-frame {
  width: 100% !important;
  height: auto !important;
}
.gradio-container .surflo-crisp .image-frame img,
.gradio-container .surflo-crisp img {
  image-rendering: pixelated;
  width: 100% !important;
  height: auto !important;
  max-width: 100% !important;
  object-fit: fill !important;
}
"""

_HERO_HTML = """
<div class="surflo-eyebrow">arXiv preprint · 2026</div>
<div class="surflo-title"><span class="accent">Surflo</span></div>
<div class="surflo-sub" style="color: #fff;">Consistent 3D Surface Flow Model with Global State</div>
<div class="surflo-sub">An interactive demo driving the Surflo Python API end to end.</div>
<div class="surflo-pipeline">
  <span class="surflo-pill">1 · Load & encode</span>
  <span class="surflo-pill">2 · Decode with FM</span>
  <span class="surflo-pill">3 · Points</span>
  <span class="surflo-pill">4 · Mesh</span>
</div>
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    # Gradio >= 6 moved `theme` and `css` from the Blocks constructor to launch();
    # older versions want them on Blocks. Set them wherever the version accepts them.
    blocks_kwargs: Dict[str, Any] = {"title": "Surflo demo"}
    if _GR_MAJOR and _GR_MAJOR < 6:
        blocks_kwargs["css"] = SURFLO_CSS
        blocks_kwargs["theme"] = _surflo_theme()
    # Expire Gradio's *own* copies of everything it serves (GRADIO_TEMP_DIR,
    # default <tmp>/gradio) -- the demo cannot reach those by managing its own
    # paths. Sweep hourly, dropping files older than a day.
    #
    # This is age-based, not reference-counted: it will delete a file a
    # still-open tab points at, and browser-uploaded input images live in the
    # same cache. Keep the age comfortably above any realistic session length.
    if _GR_MAJOR and _GR_MAJOR >= 5:
        blocks_kwargs["delete_cache"] = (3600, 86400)
    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(_HERO_HTML, elem_id="surflo-hero")

        scene_state = gr.State(None)
        result_state = gr.State(None)
        meta_state = gr.State(None)

        # -- Step 1: load images & encode -----------------------------------
        # Two input variants share the status / Encode button below:
        #   * default    -> browser file-upload + gallery preview
        #   * dev (--dev) -> server-side directory + sampling fields + text list
        # (Dev mode shows a *text* list, not a gallery: a gallery would try to
        # copy the server-side images into Gradio's cache, which is both wasteful
        # and disallowed for paths outside the app dir.)
        images_in = dev_dir = dev_n = dev_sampling = dev_load_btn = None
        gallery = dev_preview = None
        with gr.Accordion("1 · Load images & encode", open=True, elem_classes=["surflo-step"]):
            if DEV_MODE:
                gr.Markdown(
                    "_**Dev mode.** Point to a folder of views **on the server** "
                    "(the machine running this script), choose how many to sample "
                    "and how, then click **List images** to check the selection and "
                    "**Encode** to compress them into the 128 × 512 global state. "
                    "Images are read straight from disk — nothing is uploaded or cached._",
                    elem_classes=["surflo-note"],
                )
                dev_dir = gr.Textbox(
                    label="Image directory (on the server)",
                    placeholder="/path/to/scene/images",
                )
                with gr.Row():
                    dev_n = gr.Number(
                        value=0, precision=0, label="Num images (0 = all)",
                    )
                    dev_sampling = gr.Radio(
                        ["uniform", "random"], value="uniform", label="Sampling",
                    )
                # Explicit list (no live .change, no gallery): just a text summary
                # of the selected frames, computed on click after the fields are set.
                dev_load_btn = gr.Button(
                    "List images", variant="secondary", elem_classes=["surflo-cta"],
                )
                dev_preview = gr.Markdown(elem_classes=["surflo-note"])
            else:
                gr.Markdown(
                    "_Select one or more views of a scene (the only input Surflo needs), "
                    "then click **Encode** to compress them into the 128 × 512 global state._",
                    elem_classes=["surflo-note"],
                )
                images_in = gr.File(
                    label="Select images (browse your computer)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                gallery = gr.Gallery(label="Selected images", columns=6, height=180)
            encode_status = gr.Markdown(elem_classes=["surflo-note"])
            encode_btn = gr.Button(
                "Encode", variant="primary", elem_classes=["surflo-cta"],
            )

        # -- Optional: global-state portrait (dev toggle SHOW_GLOBAL_STATE) --
        # Kept behind a flag so it can be brought back without re-adding code.
        state_img = None
        if SHOW_GLOBAL_STATE:
            with gr.Accordion("Global state", open=True, elem_classes=["surflo-step"]):
                gr.Markdown(
                    "_Visualization of the encoded latent state (128 × 512), with "
                    "positive values in green and negative values in purple._",
                    elem_classes=["surflo-note"],
                )
                state_img = gr.Image(
                    label="Global state · 128 × 512", interactive=False,
                    elem_classes=["surflo-crisp"],
                )

        # -- Step 2: decode with FM ------------------------------------------
        with gr.Accordion("2 · Decode with Flow Matching", open=True, elem_classes=["surflo-step"]):
            gr.Markdown(
                "_Decode the global state into a 3D surface. Pick an inference mode "
                "(with or without guidance), and a guidance preset. The guidance produces "
                "Gaussians that can be used for meshing._\n"
                "\n"
                # Preset names stay upright (inline code / plain text); the cost
                # and description are italic, matching the caption around them.
                "- *plain (no guidance)* *— ~8 s · coarse*\n"
                "- `minimal`             *— ~15 s · good*\n"
                "- `short`               *— ~25 s · sharp*\n"
                "- `default`             *— ~45 s · sharper, with better detail*\n"
                "- `long`                *— ~95 s · even sharper with many images*\n"
                "\n"
                "_Measured on an NVIDIA H100 at 100 k query points._",
                elem_classes=["surflo-note", "surflo-perf"],
            )
            with gr.Row():
                mode_dd = gr.Dropdown(MODES, value="guided", label="Mode")
                preset_dd = gr.Dropdown(
                    GUIDED_PRESETS, value=DEFAULT_GUIDED_PRESET,
                    label="guided preset", visible=True,
                )
            with gr.Row():
                nqp = gr.Slider(
                    10_000, 300_000, value=100_000, step=10_000,
                    label="Num query points",
                )
                seed_in = gr.Number(value=42, label="Seed", precision=0)
            run_status = gr.Markdown(elem_classes=["surflo-note"])
            run_btn = gr.Button(
                "Decode", variant="primary", elem_classes=["surflo-cta"],
            )

        # -- Step 3: point cloud --------------------------------------------
        cloud_state = gr.State(None)
        with gr.Accordion("3 · Point cloud", open=True, elem_classes=["surflo-step"]):
            gr.Markdown(
                "_Click the button to display the point cloud. Switch **Colors** (RGB / Normals) to update the plot._",
                elem_classes=["surflo-note"],
            )
            pc_color = gr.Radio(
                ["RGB", "Normals"], value="Normals", label="Colors",
            )
            pc_view = gr.Plot(label="Point cloud (drag to rotate, scroll to zoom)")
            pc_status = gr.Markdown(elem_classes=["surflo-note"])
            pc_file = gr.DownloadButton(
                "Download point cloud (.ply)", variant="secondary", visible=False,
            )
            pc_btn = gr.Button(
                "Show point cloud", variant="primary",
                elem_classes=["surflo-cta"],
            )

        # -- Step 4: mesh ---------------------------------------------------
        mesh_state = gr.State(None)
        with gr.Accordion("4 · Mesh (guided decoding only)", open=False, elem_classes=["surflo-step"]):
            gr.Markdown(
                "_Click the button to extract the mesh once, then switch **Vertex colors** (RGB / Normals / None) "
                "freely to update the plot._",
                elem_classes=["surflo-note"],
            )
            mesh_color = gr.Radio(
                ["RGB (TSDF)", "Normals", "None"], value="RGB (TSDF)",
                label="Vertex colors",
            )
            mesh_view = gr.Model3D(label="Mesh", height=420)
            mesh_status = gr.Markdown(elem_classes=["surflo-note"])
            mesh_file = gr.DownloadButton(
                "Download mesh (.ply)", variant="secondary", visible=False,
            )
            mesh_btn = gr.Button(
                "Extract mesh", variant="primary",
                elem_classes=["surflo-cta"],
            )

        # -- wiring ---------------------------------------------------------
        if DEV_MODE:
            dev_inputs = [dev_dir, dev_n, dev_sampling]
            # List the selected frames as text only, on explicit click (no live
            # .change, no gallery -> nothing gets copied into Gradio's cache).
            dev_load_btn.click(list_dir_cb, dev_inputs, [dev_preview])
            enc_evt = encode_btn.click(
                encode_dir_cb,
                dev_inputs,
                [scene_state, encode_status],
                show_progress="hidden",
            )
        else:
            images_in.change(preview_images_cb, [images_in], [gallery])
            enc_evt = encode_btn.click(
                encode_cb,
                [images_in],
                [scene_state, encode_status],
                # Our spinner is the indicator; hide Gradio's built-in progress track.
                show_progress="hidden",
            )
        if SHOW_GLOBAL_STATE:
            enc_evt.then(state_viz_cb, [scene_state], [state_img])
        mode_dd.change(on_mode_change, [mode_dd], [preset_dd])
        run_btn.click(
            reconstruct_cb,
            [scene_state, mode_dd, preset_dd, nqp, seed_in],
            [result_state, meta_state, run_status],
            # Suppress Gradio's built-in progress track (the teal bar that renders
            # above the status text); our streamed "Decoding…" message is the indicator.
            show_progress="hidden",
        ).then(
            # A new reconstruction invalidates the cached point / mesh colorings and
            # hides the (now stale) point-cloud / mesh viewers until re-generated.
            _reset_downstream, None,
            [cloud_state, mesh_state, pc_view, pc_file, mesh_view, mesh_file],
        )
        pc_click_inputs = [scene_state, cloud_state, result_state, pc_color]
        pc_click_outputs = [cloud_state, pc_view, pc_file, pc_status]
        pc_btn.click(visualize_pointcloud_cb, pc_click_inputs, pc_click_outputs)
        # Auto-update the plot when the color mode changes (no re-color if cached).
        pc_color.change(visualize_pointcloud_cb, pc_click_inputs, pc_click_outputs)
        mesh_btn.click(
            extract_mesh_cb,
            [scene_state, result_state, meta_state, mesh_color],
            [mesh_state, mesh_view, mesh_file, mesh_status],
        )
        # Switch colors without re-extracting the geometry.
        mesh_recolor_inputs = [scene_state, mesh_state, result_state, mesh_color]
        mesh_recolor_outputs = [mesh_state, mesh_view, mesh_file, mesh_status]
        mesh_color.change(recolor_mesh_cb, mesh_recolor_inputs, mesh_recolor_outputs)

        # Drop this session's exports when its browser tab closes. Sessions that
        # end without firing this (a killed server) are caught by the atexit
        # sweep over the whole root instead.
        demo.unload(_cleanup_session)

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surflo Gradio demo.")
    p.add_argument(
        "--ckpt", type=str, required=True,
        help="Checkpoint path (e.g. surflo_v0.pt); loaded once before the server starts.",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device (cuda / cpu).")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    p.add_argument("--port", type=int, default=7860, help="Port.")
    p.add_argument("--share", action="store_true", help="Create a public link.")
    p.add_argument(
        "--dev", action="store_true",
        help=(
            "Dev mode: load images from a server-side directory (fields for "
            "path / num images / sampling) instead of the browser upload widget."
        ),
    )
    return p.parse_args()


def main() -> None:
    # The library logs through the stdlib `logging` module. These entry points
    # are not Hydra apps (which would configure it for us), so set it up here or
    # nothing below WARNING would be shown.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    global MODEL, DEV_MODE
    args = parse_args()
    DEV_MODE = bool(args.dev)
    _bypass_localhost_proxy()

    ckpt = args.ckpt.strip()
    if not Path(ckpt).is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    print(f"[surflo-demo] Loading model on {args.device} from {ckpt} ...", flush=True)
    MODEL = Surflo.from_checkpoint(ckpt, device=args.device)
    if DEV_MODE:
        print(
            "[surflo-demo] Dev mode ON: images are loaded from a server-side "
            "directory (path / num / sampling fields).",
            flush=True,
        )
    print("[surflo-demo] Model loaded. Starting the demo server ...", flush=True)

    demo = build_demo()
    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.host, server_port=args.port, share=args.share,
    )
    if _GR_MAJOR >= 6:
        launch_kwargs["theme"] = _surflo_theme()
        launch_kwargs["css"] = SURFLO_CSS
    try:
        demo.queue().launch(**launch_kwargs)
    except TypeError:
        # Some versions don't accept `theme` / `css` on launch(); retry without them.
        launch_kwargs.pop("theme", None)
        launch_kwargs.pop("css", None)
        demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()

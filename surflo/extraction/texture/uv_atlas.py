"""xatlas-based UV unwrap for arbitrary triangle meshes.

Producing a textured mesh (per-vertex colors -> single 2D texture + UV map
-> GLB) instead of a per-vertex-color PLY requires a 2D atlas: every face
mapped to a (u, v) region of [0, 1]^2. That is done with
`xatlas <https://github.com/jpcy/xatlas>`_, which handles arbitrary geometry,
packs charts automatically, and ships a small Python binding.

The unwrap duplicates vertices along chart seams -- xatlas needs one UV per
chart corner, so a vertex on the boundary between two charts must carry two
UVs. The returned ``vertex_remap`` lets the caller scatter any per-vertex
attribute (e.g. TSDF-seeded colors) onto the new vertex array instead of
recomputing it.

xatlas is an OPTIONAL dependency, needed only for ``texture.mode=uv_texture``.
The import is lazy and raises an actionable error if that path is selected
without it installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from surflo.structures.mesh import Meshes


__all__ = ["UVAtlasResult", "unwrap_mesh"]


_log = logging.getLogger(__name__)


@dataclass
class UVAtlasResult:
    """Output of :func:`unwrap_mesh`.

    Attributes:
        mesh: a fresh :class:`Meshes` with the UV-unwrapped topology
            (``verts`` and ``verts_colors`` of length ``V'`` >= ``V``;
            ``faces`` of shape ``(F, 3)`` indexing the new verts).
            ``verts_colors`` is populated when the input mesh had any
            (scattered through :attr:`vertex_remap`).
        uvs: ``(V', 2)`` float tensor of per-vertex UVs in ``[0, 1]``,
            (0, 0) bottom-left (xatlas / nvdiffrast convention).
        vertex_remap: ``(V',)`` long tensor mapping each new vertex
            back to its original index in ``mesh.verts``. Use this to
            scatter any per-vertex attribute (e.g. ``my_attr[remap]``).
        atlas_width / atlas_height: texel dimensions xatlas packed
            into. The caller is free to override the actual texture
            resolution it allocates, but using these values matches
            the chart sizes xatlas computed from
            :attr:`ChartOptions.texels_per_unit` (we let xatlas pick a
            target resolution and just record what it chose).
    """
    mesh: Meshes
    uvs: torch.Tensor
    vertex_remap: torch.Tensor
    atlas_width: int
    atlas_height: int


def _import_xatlas():
    """Lazy import with a friendly error message.

    Keeps xatlas off the startup path for users who only run the
    vertex-color path (the default).
    """
    try:
        import xatlas  # type: ignore[import-not-found]
        return xatlas
    except ImportError as e:
        raise ImportError(
            "xatlas is required for the UV-texture path "
            "(`eval.texture.mode == 'uv_texture'`). "
            "Install it with `pip install xatlas` and re-run."
        ) from e


def unwrap_mesh(
    mesh: Meshes,
    *,
    resolution: int = 2048,
    padding: int = 2,
    max_chart_iterations: int = 2,
    brute_force: bool = False,
) -> UVAtlasResult:
    """Compute a UV atlas for ``mesh`` and return a re-indexed mesh.

    Args:
        mesh: input triangle mesh on CUDA or CPU; we move geometry
            tensors to CPU + numpy for xatlas (it's a CPU library) and
            push results back to the input mesh's device.
        resolution: target packed atlas resolution (texels). Passed to
            :class:`xatlas.PackOptions.resolution`. Smaller charts get
            more padding relative to chart size at high resolutions,
            so picking the FINAL texture resolution here keeps gutter
            sizing consistent.
        padding: per-chart gutter in texels. ``2`` is xatlas's default
            and is just enough for bilinear sampling to stay inside
            each chart. Bump to 4-8 if you plan to use trilinear /
            mipmaps without a dedicated dilation pass.
        max_chart_iterations: xatlas chart-construction iterations
            (:class:`xatlas.ChartOptions.max_iterations`). ``2`` is the
            xatlas default and typical quality/speed tradeoff.
        brute_force: forwarded to :class:`xatlas.PackOptions`; ``False``
            uses the default packer (fast, sometimes leaves wasted
            atlas space). Use ``True`` only when you need the tightest
            possible packing.

    Returns:
        :class:`UVAtlasResult` (see its docstring).

    Raises:
        ImportError: xatlas is not installed.
        RuntimeError: the input mesh is empty or xatlas rejected it
            (degenerate triangles, non-manifold geometry that breaks
            chart construction, etc.).
    """
    xatlas = _import_xatlas()

    device = mesh.verts.device
    n_v_in = int(mesh.verts.shape[0])
    n_f_in = int(mesh.faces.shape[0])
    if n_v_in == 0 or n_f_in == 0:
        raise RuntimeError(
            f"unwrap_mesh: empty mesh (V={n_v_in}, F={n_f_in})."
        )

    # xatlas is a CPU library. We don't fight it: move geometry to numpy.
    # The atlas itself is small (a handful of MB at 2048^2 even with seams),
    # so the device round-trip cost is dominated by the GPU geometry
    # transfer, not the algorithm.
    verts_np = mesh.verts.detach().cpu().numpy().astype(np.float32)
    faces_np = mesh.faces.detach().cpu().numpy().astype(np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts_np, faces_np)

    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = int(max_chart_iterations)

    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(resolution)
    pack_options.padding = int(padding)
    # The xatlas python binding exposes the brute-force flag as
    # ``bruteForce`` (camelCase, matching the C++ field name) in recent
    # releases (0.0.11+). Older releases used ``brute_force``; fall
    # back to that if the camelCase attribute isn't there so the
    # wrapper keeps working with whichever build the user has pinned.
    if hasattr(pack_options, "bruteForce"):
        pack_options.bruteForce = bool(brute_force)
    elif hasattr(pack_options, "brute_force"):
        pack_options.brute_force = bool(brute_force)
    elif bool(brute_force):
        _log.warning(
            "[uv-atlas] xatlas.PackOptions exposes neither 'bruteForce' nor "
            "'brute_force'; ignoring brute_force=True."
        )

    _log.info(
        f"[uv-atlas] xatlas.generate(V={n_v_in}, F={n_f_in}, "
        f"resolution={resolution}, padding={padding}, "
        f"max_chart_iters={max_chart_iterations})..."
    )
    atlas.generate(chart_options=chart_options, pack_options=pack_options)

    vmapping, indices, uvs = atlas[0]
    # xatlas returns:
    #   vmapping : (V',)  uint32 - new vertex i -> original vertex id
    #   indices  : (F, 3) uint32 - face indices in the NEW vertex array
    #   uvs      : (V', 2) float32 - per-vertex UV in [0, 1] (origin BL)
    # All on CPU.
    vmapping_t = torch.from_numpy(vmapping.astype(np.int64)).to(device=device)
    indices_t = torch.from_numpy(indices.astype(np.int64)).to(device=device)
    uvs_t = torch.from_numpy(uvs.astype(np.float32)).to(device=device)

    # Scatter per-vertex attributes through the remap. Geometry is the
    # only one we always have; vertex colors come along if the input
    # mesh carried any (e.g. TSDF-seeded colors from
    # :func:`_evaluate_mesh_colors_all_vertices`).
    new_verts = mesh.verts.detach().to(device=device, dtype=torch.float32)[vmapping_t]
    new_verts_colors: Optional[torch.Tensor] = None
    if mesh.verts_colors is not None and mesh.verts_colors.shape[0] == n_v_in:
        new_verts_colors = (
            mesh.verts_colors.detach().to(device=device, dtype=torch.float32)[vmapping_t]
        )

    new_mesh = Meshes(
        verts=new_verts,
        faces=indices_t.to(torch.int32),
        verts_colors=new_verts_colors,
    )

    n_v_out = int(new_verts.shape[0])
    n_f_out = int(indices_t.shape[0])
    n_dupes = n_v_out - n_v_in
    _log.info(
        f"[uv-atlas] done: V'={n_v_out} (+{n_dupes} seam duplicates from V={n_v_in}), "
        f"F={n_f_out}, atlas={atlas.width}x{atlas.height}."
    )

    return UVAtlasResult(
        mesh=new_mesh,
        uvs=uvs_t,
        vertex_remap=vmapping_t,
        atlas_width=int(atlas.width),
        atlas_height=int(atlas.height),
    )

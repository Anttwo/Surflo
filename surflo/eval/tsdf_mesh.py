"""TSDF -> multi-resolution mesh -> camera-aware surface sampling.

The eval-only mesh path used by the ``vggt`` / ``da3`` baselines when
``use_tsdf=true`` (``scripts/evaluate.py``).

Every heavy primitive it calls already exists in ``surflo``:

* :func:`surflo.extraction.depth_fusion.tsdf.fuse_depth_maps`
* :func:`surflo.extraction.extractors.masked.extract_mesh` (marching tetrahedra)
* :class:`surflo.structures.mesh.Meshes` / :func:`combine_meshes`
* :func:`surflo.structures.cameras.get_cameras_from_intrinsics_and_extrinsics`
  / :func:`is_in_view_frustum`
* :func:`surflo.data.surface_sampling.sample_points_on_mesh`

The pipeline mirrors ``FFM.meshify_by_tsdf`` but sizes the grids from the
per-scene normalization stats (the same ``(mean, scale)`` that define the
dataset cull ball) rather than from the camera spatial extent.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch

from surflo.data.surface_sampling import sample_points_on_mesh
from surflo.extraction.depth_fusion.tsdf import fuse_depth_maps
from surflo.extraction.extractors.masked import extract_mesh
from surflo.structures.cameras import (
    Camera,
    get_cameras_from_intrinsics_and_extrinsics,
    is_in_view_frustum,
)
from surflo.structures.mesh import Meshes, combine_meshes

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regular tetrahedral grid (torch-only, no other dependencies).
# ---------------------------------------------------------------------------
def _get_cube_tets(Nx: int = 1, Ny: int = 1, Nz: int = 1) -> torch.Tensor:
    """Tet indices of the first cube of a regular grid (6 tets per cube)."""
    _Nx, _Ny, _Nz = Nx + 1, Ny + 1, Nz + 1
    return torch.tensor(
        [
            [0, _Ny * _Nz + _Nz, _Nz, 1],
            [0, _Ny * _Nz, _Ny * _Nz + _Nz, 1],
            [_Nz + 1, _Nz, _Ny * _Nz + _Nz, 1],
            [_Nz + 1, 1, _Ny * _Nz + _Nz, _Ny * _Nz + _Nz + 1],
            [_Ny * _Nz + 1, _Ny * _Nz + _Nz, _Ny * _Nz, 1],
            [_Ny * _Nz + 1, _Ny * _Nz + _Nz, 1, _Ny * _Nz + _Nz + 1],
        ],
        dtype=torch.int32,
    )


def get_tet_grid(
    n_cube_per_axis: Union[int, Tuple[int, int, int]],
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verts (in ``[0, 1]^3``) and tet indices of a regular tetrahedral grid."""
    if isinstance(n_cube_per_axis, int):
        n_cube_per_axis = (n_cube_per_axis, n_cube_per_axis, n_cube_per_axis)

    grid_verts = torch.meshgrid(
        torch.arange(n_cube_per_axis[0] + 1, device=device),
        torch.arange(n_cube_per_axis[1] + 1, device=device),
        torch.arange(n_cube_per_axis[2] + 1, device=device),
    )
    grid_verts = torch.stack(grid_verts, dim=-1).reshape(-1, 3)
    norm_factor = 1.0 / max(n_cube_per_axis)
    grid_verts = grid_verts * norm_factor

    cube_tets = _get_cube_tets(
        Nx=n_cube_per_axis[0], Ny=n_cube_per_axis[1], Nz=n_cube_per_axis[2],
    ).to(device)  # (6, 4)

    grid_vert_indices = torch.meshgrid(
        torch.arange(n_cube_per_axis[0], device=device),
        torch.arange(n_cube_per_axis[1], device=device),
        torch.arange(n_cube_per_axis[2], device=device),
    )
    Ny, Nz = n_cube_per_axis[1] + 1, n_cube_per_axis[2] + 1
    grid_vert_indices = (
        grid_vert_indices[0] * (Ny * Nz)
        + grid_vert_indices[1] * Nz
        + grid_vert_indices[2]
    )
    grid_vert_indices = grid_vert_indices.reshape(-1, 1, 1)

    grid_tets = grid_vert_indices + cube_tets[None, :, :]
    grid_tets = grid_tets.reshape(-1, 4)
    return grid_verts, grid_tets


# ---------------------------------------------------------------------------
# Per-scene multi-resolution TSDF mesh
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_aabb_tet_grid(
    scene_mean: torch.Tensor,    # (3,)
    scene_scale: torch.Tensor,   # (3,)
    radius_scale: float,
    n_cube_per_axis: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tet grid covering ``mean ± radius_scale * scale`` per axis."""
    pivots, tets = get_tet_grid(n_cube_per_axis=n_cube_per_axis, device=device)
    pivots = 2.0 * pivots - 1.0                                        # [-1, 1]^3
    pivots = scene_mean[None, :] + pivots * (radius_scale * scene_scale)[None, :]
    return pivots, tets


@torch.no_grad()
def extract_multires_tsdf_mesh(
    *,
    extrinsics: torch.Tensor,    # (N, 3, 4) world-to-cam
    intrinsics: torch.Tensor,    # (N, 3, 3)
    depths: torch.Tensor,        # (N, H, W)
    images: torch.Tensor,        # (N, 3, H, W)
    scene_mean: torch.Tensor,    # (3,)
    scene_scale: torch.Tensor,   # (3,)
    radius_scales: Tuple[float, ...],
    n_cube_per_axis: int,
    trunc_margin_factor: float,
    initial_sdf_value: float,
) -> Meshes:
    """Fuse depth maps into multi-res AABB grids and combine into one mesh.

    For each grid in ``radius_scales``: build pivots ``mean ± k * scale`` per
    axis, TSDF-fuse the depth maps, and marching-tet extract a submesh. Shell
    ``i >= 1`` keeps only verts OUTSIDE the previous shell (anisotropic
    per-scene metric ``‖(v - mean) / scale‖_2 >= prev_scale``) so shells don't
    double-cover.
    """
    cameras = get_cameras_from_intrinsics_and_extrinsics(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        images=images,
    )
    device = scene_mean.device

    # Scalar "scene radius" that sizes the TSDF truncation margin
    # (trunc_margin = trunc_margin_factor * scene_radius). Passing it
    # explicitly avoids falling back to the camera spatial extent inside
    # fuse_depth_maps -- we want the band tied to the same per-scene scale
    # that defined cull_radius.
    scene_radius_scalar = float(scene_scale.mean().item())

    submeshes: List[Meshes] = []
    for k_scale in radius_scales:
        pivots, tets = build_aabb_tet_grid(
            scene_mean=scene_mean,
            scene_scale=scene_scale,
            radius_scale=float(k_scale),
            n_cube_per_axis=n_cube_per_axis,
            device=device,
        )
        tsdf, _colors = fuse_depth_maps(
            pivots=pivots,
            cameras=cameras,
            images=images,
            depths=depths,
            trunc_margin_factor=trunc_margin_factor,
            scene_radius=scene_radius_scalar,
            initial_sdf_value=initial_sdf_value,
        )
        m = extract_mesh(
            delaunay_tets=tets,
            pivots=pivots,
            pivots_sdf=tsdf,
            pivots_colors=None,
            filter_large_edges=False,
            collapse_large_edges=False,
        )
        submeshes.append(m)
        _log.info(
            f"[tsdf]   shell @ {k_scale:.2f}*scale: "
            f"{m.verts.shape[0]} verts, {m.faces.shape[0]} faces"
        )

    # Clean shells: shell i (i>=1) keeps only verts OUTSIDE the previous one.
    cleaned: List[Meshes] = [submeshes[0]]
    for i in range(1, len(submeshes)):
        m = submeshes[i]
        if m.verts.shape[0] == 0 or m.faces.shape[0] == 0:
            cleaned.append(m)
            continue
        std_dist = ((m.verts - scene_mean) / scene_scale).norm(dim=-1)
        prev_scale = float(radius_scales[i - 1])
        keep_mask = std_dist >= prev_scale
        if int(keep_mask.sum().item()) == 0:
            cleaned.append(Meshes(
                verts=torch.zeros(0, 3, device=device, dtype=m.verts.dtype),
                faces=torch.zeros(0, 3, device=device, dtype=m.faces.dtype),
            ))
            continue
        cleaned.append(m.submesh(vert_mask=keep_mask))
        _log.info(
            f"[tsdf]   shell @ {radius_scales[i]:.2f}*scale after shell-clean: "
            f"{cleaned[-1].verts.shape[0]} verts, {cleaned[-1].faces.shape[0]} faces"
        )

    return combine_meshes(cleaned)


@torch.no_grad()
def cull_mesh_by_radius(
    mesh: Meshes,
    scene_mean: torch.Tensor,    # (3,)
    scene_scale: torch.Tensor,   # (3,)
    cull_radius: float,
) -> Meshes:
    """Drop triangles whose centroid lies outside ``‖(c-μ)/scale‖₂ < r``.

    Triangle-level (rather than vertex-level) culling avoids spurious sliver
    triangles right at the cut boundary.
    """
    if mesh.faces.shape[0] == 0:
        return mesh
    face_verts = mesh.verts[mesh.faces.long()]              # (F, 3, 3)
    centroids = face_verts.mean(dim=1)                       # (F, 3)
    std_dist = ((centroids - scene_mean) / scene_scale).norm(dim=-1)
    keep = std_dist < float(cull_radius)
    if int(keep.sum().item()) == int(keep.numel()):
        return mesh
    return mesh.submesh(face_mask=keep)


@torch.no_grad()
def sample_points_on_mesh_camera_aware(
    mesh: Meshes,
    n_points: int,
    *,
    cameras: List[Camera],
    frustum_cull_before_sampling: bool = True,
    frustum_cull_znear: Optional[float] = None,
    n_face_max: int = 2 ** 24,
    distance_batch_size: int = 100_000,
    generator: Optional[torch.Generator] = None,
    return_normals: bool = False,
):
    """Camera-distance-biased surface sampling for the post-TSDF stage.

    Returns ``points`` (``(n_points, 3)``), or ``(points, normals)`` when
    ``return_normals`` is set -- the face normal of the triangle each sample
    was drawn from, which is what the baselines are scored on for normal
    accuracy. Marching tetrahedra emits outward-facing triangles, so those
    normals share the GT's orientation convention.

    Per-triangle probabilities are ``area / min_dist_to_visible_camera ** 2``
    (via ``sample_points_on_mesh(..., adjust_with_distance_to_cameras=True)``)
    so density is approximately uniform once projected back into the input
    cameras. When ``frustum_cull_before_sampling`` is true, a triangle is kept
    iff ALL THREE of its vertices fall inside AT LEAST ONE camera frustum
    (strict, conservative "fully observed" semantics).

    ``sample_points_on_mesh`` takes no ``torch.Generator``; determinism is
    pinned by forking the global RNG with the generator's seed for the
    duration of the call only.
    """
    device = mesh.verts.device

    def _empty():
        pts = torch.zeros((0, 3), device=device, dtype=mesh.verts.dtype)
        return (pts, pts.clone()) if return_normals else pts

    if mesh.faces.shape[0] == 0 or n_points <= 0:
        return _empty()

    if frustum_cull_before_sampling:
        face_keep = torch.zeros(
            mesh.faces.shape[0], device=device, dtype=torch.bool,
        )
        faces_long = mesh.faces.long()
        for cam in cameras:
            verts_in_view = is_in_view_frustum(
                mesh.verts, cam, znear=frustum_cull_znear,
            )                                                    # (V,)
            face_keep = face_keep | verts_in_view[faces_long].all(dim=1)
        if int(face_keep.sum().item()) < mesh.faces.shape[0]:
            mesh = Meshes(
                verts=mesh.verts,
                faces=mesh.faces[face_keep],
                verts_colors=mesh.verts_colors,
            )
        if mesh.faces.shape[0] == 0:
            return _empty()

    sample_kwargs = dict(
        mesh=mesh,
        n_points=n_points,
        adjust_with_distance_to_cameras=True,
        cameras=cameras,
        batch_size_for_face_to_camera_distance=distance_batch_size,
        znear=frustum_cull_znear,
        n_face_max=n_face_max,
        return_normals=return_normals,
    )
    if generator is None:
        return sample_points_on_mesh(**sample_kwargs)

    seed = int(generator.initial_seed())
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        return sample_points_on_mesh(**sample_kwargs)

import logging
import numpy as np
import torch
from scipy.spatial import Delaunay
from typing import Optional, Tuple, Callable
from surflo.rendering.gaussians import Gaussians
from surflo.utils.geometry import build_rotation
import trimesh

_log = logging.getLogger(__name__)

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def _orient_tets_positive(points: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    # tets: (N, 4) long, points: (N_p, 3)
    v = points[tets]                                  # (N, 4, 3)
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    e3 = v[:, 3] - v[:, 0]
    signed_vol = (torch.cross(e1, e2, dim=-1) * e3).sum(-1)
    negative = signed_vol < 0
    if negative.any():
        # Swap V2 and V3 in negatively-oriented tets to flip their handedness.
        flipped = tets.clone()
        flipped[negative] = flipped[negative][:, [0, 1, 3, 2]]
        return flipped
    return tets


def get_gaussian_std_in_direction(
    directions: torch.Tensor,
    gaussians: Optional[Gaussians]=None,
    gaussian_scaling: Optional[torch.Tensor]=None,
    gaussian_rotation: Optional[torch.Tensor]=None,
    normalize_directions: bool = True,
) -> torch.Tensor:
    """
    Get the standard deviation of the Gaussian in the given directions.
    
    If gaussians is provided, the scaling and rotation are extracted from the Gaussians.
    If gaussian_scaling and gaussian_rotation are provided, they are used instead of the Gaussians.
    
    Args:
        directions (torch.Tensor): A vector of shape (N_gaussians, n_directions, 3).
        gaussians (GaussianModel): The Gaussian model, with N_gaussians Gaussians.
        gaussian_scaling (torch.Tensor): The scaling of the Gaussians. Has shape (N_gaussians, 3).
        gaussian_rotation (torch.Tensor): The rotation of the Gaussians. Has shape (N_gaussians, 3, 3).
        normalize_directions (bool): Whether to normalize the directions.

    Returns:
        torch.Tensor: The standard deviation of the Gaussian in the direction of the given vector, of shape (N_gaussians, n_directions).
    """
    assert (
        gaussians is not None
        or (gaussian_scaling is not None and gaussian_rotation is not None)
    )
    
    # Get transposed scaled rotation
    if gaussian_scaling is None:
        gaussian_scaling = gaussians.scales.detach()
    if gaussian_rotation is None:
        gaussian_rotation = gaussians.rotations.detach()
    transposed_scaled_rotation = build_scaling_rotation(
        s=gaussian_scaling,  # (N_gaussians, 3)
        r=gaussian_rotation,  # (N_gaussians, 3, 3)
    ).transpose(-1, -2)  # (N_gaussians, 3, 3)
    
    if normalize_directions:
        directions_to_use = torch.nn.functional.normalize(directions, dim=-1)
    else:
        directions_to_use = directions
    
    scaled_directions = torch.bmm(
        transposed_scaled_rotation,  # (N_gaussians, 3, 3)
        directions_to_use.permute(0, 2, 1),  # (N_gaussians, 3, n_directions)
    ).permute(0, 2, 1)  # (N_gaussians, n_directions, 3)
    
    direction_stds = scaled_directions.norm(dim=-1)  # (N_gaussians, n_directions)
    return direction_stds  # (N_gaussians, n_directions)


def get_regular_gaussian_pivots(
    gaussians:Gaussians, 
    std_factor:float=3.33,
    opacity_threshold:Optional[float]=None,
    override_opacity:Optional[torch.Tensor]=None,
    verbose:bool=False,
):
    """
    Get the tetra points of the Gaussian model.

    Args:
        downsample_ratio (float, optional): The ratio to downsample the tetra points. Defaults to None.
        return_sdf_values (bool, optional): Whether to return the SDF values. Defaults to False.
        xyz_idx (torch.Tensor, optional): The indices of the tetra points to return. 
            If opacity_threshold is provided, xyz_idx should index points that are not filtered out by the opacity threshold,
            such that xyz_idx.max() < (self.get_opacity_with_3D_filter > opacity_threshold).sum().
            Defaults to None. 
            Overrides downsample_ratio if both are provided.
        verbose (bool, optional): Whether to print verbose information. Defaults to False.
        scale_points_with_downsample_ratio (bool, optional): Whether to scale the points with the downsample ratio. 
            Defaults to True. Overrides scale_points_factor if both are provided.
        scale_points_factor (float, optional): The factor to scale the points. Defaults to None.
        opacity_threshold (float, optional): The opacity threshold to filter the points. Defaults to None.
        override_opacity (torch.Tensor, optional): The opacities to use for the tetra points. 
        return_min_scales (bool, optional): Whether to return the minimum scales of the vertices. Defaults to False.
        point_shifts (torch.Tensor, optional): The shifts to apply to the points. Defaults to None. Has shape (N_points, 9, 3).
    Raises:
        ValueError: If SDF values are not used but return_sdf_values is True.

    Returns:
        vertices (torch.Tensor): The vertices of the tetra points.
        vertices_scale (torch.Tensor): The scale of the vertices.
        sdf_values (torch.Tensor, optional): The SDF values of the tetra points.
    """
    M = trimesh.creation.box()
    M.vertices *= 2
        
    xyz = gaussians.means
    scale = gaussians.scales * std_factor
    rots = build_rotation(gaussians.rotations)
    
    # Filter points with small opacity
    if (opacity_threshold is not None) and (opacity_threshold > 0.0):
        if override_opacity is not None:
            opacity = override_opacity
        else:
            opacity = gaussians.opacities
        mask = (opacity > opacity_threshold).squeeze()
        xyz = xyz[mask]
        scale = scale[mask]
        rots = rots[mask]
        if verbose:
            _log.info(f"Number of tetra points after opacity threshold: {xyz.shape[0]}.")
    
    vertices = M.vertices.T    
    vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)  # (N_points, 3, 8)
    
    # scale vertices first
    vertices = vertices * scale.unsqueeze(-1)
    vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
    vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
    # concat center points
    vertices = torch.cat([vertices, xyz], dim=0)

    # scale is not a good solution but use it for now
    scale = scale.max(dim=-1, keepdim=True)[0]
    scale_corner = scale.repeat(1, 8).reshape(-1, 1)
    vertices_scale = torch.cat([scale_corner, scale], dim=0)
    
    return vertices, vertices_scale


def get_gaussian_pivots(
    n_pivots: int = 9,
    gaussians: Optional[Gaussians]=None,
    normals: Optional[torch.Tensor]=None,
    std_factor: float = 3.33,
    xyz: Optional[torch.Tensor]=None,
    scaling: Optional[torch.Tensor]=None,
    rotation: Optional[torch.Tensor]=None,
    use_smallest_axis_as_normal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Get pivots from Gaussians based on their learned normals.
    The learned normal can be replaced by the smallest axis of the Gaussian by setting use_smallest_axis_as_normal to True.
    
    An sdf_function can be provided to only return the pivots for which the surface crosses between the pivots and the center point.
    This drastically reduces the number of pivots to be considered for the mesh extraction, but can hurt the quality of the mesh.

    Args:
        n_pivots (int, optional): _description_. Defaults to 2.
        gaussians (Optional[GaussianModel], optional): _description_. Defaults to None.
        normals (Optional[torch.Tensor], optional): _description_. Defaults to None.
        std_factor (float, optional): _description_. Defaults to 3.0.
        xyz (Optional[torch.Tensor], optional): _description_. Defaults to None.
        scaling (Optional[torch.Tensor], optional): _description_. Defaults to None.
        rotation (Optional[torch.Tensor], optional): _description_. Defaults to None.
        use_smallest_axis_as_normal (bool, optional): _description_. Defaults to False.
        sdf_function (Optional[Callable], optional): _description_. Defaults to None.

    Raises:
        ValueError: If gaussians or xyz, scaling and rotation are not provided.
        ValueError: If n_pivots is not in [2, 3, 7, 9].

    Returns:
        torch.Tensor: A tuple containing the pivots, the pivot scales, and the SDF values for the pivots.
            - pivots (torch.Tensor): The pivots, of shape (N_gaussians, n_pivots, 3).
            - pivot_scales (torch.Tensor): The pivot scales, of shape (N_gaussians, n_pivots, 1).
            - pivot_sdf (Optional[torch.Tensor]): The SDF values for the pivots, of shape (N_gaussians, n_pivots). Only returned if sdf_function is provided.
    """
    assert (
        gaussians is not None
        or (xyz is not None and scaling is not None and rotation is not None and normals is not None)
    ), "Either gaussians or xyz, scaling and rotation must be provided"

    assert n_pivots in [2, 3, 7, 9], f"Invalid number of pivots: {n_pivots}"
    
    if xyz is None:
        xyz = gaussians.means  # (N_gaussians, 3)
    if scaling is None:
        scaling = gaussians.scales  # (N_gaussians, 3)
    if rotation is None:
        rotation = gaussians.rotations  # (N_gaussians, 4)
        
    if use_smallest_axis_as_normal:
        #   > Compute main Gaussian axes
        rots = build_rotation(rotation)  # (N_gaussians, 3, n_axes)
        axes = rots.transpose(-1, -2)  # (N_gaussians, n_axes, 3)
        
        min_scale_idx = torch.argmin(scaling, dim=-1, keepdim=True)  # (N_gaussians, 1)
        
        normals = torch.gather(
            input=axes,  # (N_gaussians, n_axes, 3)
            index=min_scale_idx.unsqueeze(-1).repeat(1, 1, 3),  # (N_gaussians, 1, 3)
            dim=1
        )  # (N_gaussians, 1, 3)
        normals = normals.squeeze(1)  # (N_gaussians, 3)
    else:
        assert normals is not None, "Normals must be provided if use_smallest_axis_as_normal is False"
    
    n_pivots_no_center = n_pivots - 1
    
    # Normalize the normals
    normals = torch.nn.functional.normalize(normals, dim=-1)  # (N_gaussians, 3)
    
    # If using 2 or 3 pivots, we just need the normal
    if n_pivots in [2, 3]:
        # Get the standard deviation of the Gaussian in the direction of the normal
        normal_stds = get_gaussian_std_in_direction(
            directions=normals.unsqueeze(1),  # (N_gaussians, 1, 3)
            gaussians=None,
            gaussian_scaling=scaling,
            gaussian_rotation=rotation, 
            normalize_directions=False,
        )  # (N_gaussians, 1)
    
    # If using 7 or 9 pivots, we need to compute an orthonormal basis relying on the learned normal
    elif n_pivots in [7, 9]:
        # Compute an orthonormal basis relying on the learned normal
        #   > Compute main Gaussian axes
        rots = build_rotation(rotation)  # (N_gaussians, 3, n_axes)
        axes = rots.transpose(-1, -2)  # (N_gaussians, n_axes, 3)
        
        #   > Project axes onto the normal
        axes_projections = (
            normals.unsqueeze(1)  # (N_gaussians, 1, 3)
            * axes  # (N_gaussians, n_axes, 3)
        ).sum(dim=-1).abs()  # (N_gaussians, n_axes)
        
        # Get the two axes with the smallest absolute projection
        two_smallest_axes_idx = torch.argsort(axes_projections, dim=-1)[:, :2]  # (N_gaussians, 2)
        two_smallest_axes = torch.gather(
            input=axes,  # (N_gaussians, n_axes, 3)
            index=two_smallest_axes_idx.unsqueeze(-1).repeat(1, 1, 3),  # (N_gaussians, 2, 3)
            dim=1
        )  # (N_gaussians, 2, 3)
        
        # Use normal as first axis and apply Gram-Schmidt orthonormalization to the two remaining axes
        new_axes = torch.zeros_like(axes)  # (N_gaussians, n_axes, 3)
        new_axes[:, 0, :] = normals  # (N_gaussians, 3)
        new_axes[:, 1, :] = torch.nn.functional.normalize(
            two_smallest_axes[:, 1, :]  # (N_gaussians, 3)
            - (two_smallest_axes[:, 1, :] * new_axes[:, 0, :]).sum(dim=-1, keepdim=True) * new_axes[:, 0, :],  # (N_gaussians, 3)
            dim=-1,
        )  # (N_gaussians, 3)
        new_axes[:, 2, :] = torch.nn.functional.normalize(
            torch.cross(
                new_axes[:, 0, :],  # (N_gaussians, 3)
                new_axes[:, 1, :],  # (N_gaussians, 3)
                dim=-1,
            ),
            dim=-1,
        )  # (N_gaussians, 3)
        
        # Get the standard deviation of the awes
        axes_stds = get_gaussian_std_in_direction(
            directions=new_axes,  # (N_gaussians, n_axes, 3)
            gaussians=None,
            gaussian_scaling=scaling,
            gaussian_rotation=rotation, 
            normalize_directions=False,
        ).unsqueeze(-1)  # (N_gaussians, n_axes, 1)
    
    pivots = torch.zeros(xyz.shape[0], n_pivots_no_center, 3, device=xyz.device)
    
    # Apart from the center point, we use:
    if (n_pivots_no_center == 1):
        # A pivot in front of the Gaussian in the direction of the normal
        pivots[:, 0, :] = xyz + std_factor * normal_stds * normals
        
    elif n_pivots_no_center == 2:
        # Two pivots, one in front and one behind the Gaussian in the direction of the normal
        pivots[:, 0, :] = xyz + std_factor * normal_stds * normals    
        pivots[:, 1, :] = xyz - std_factor * normal_stds * normals
        
    elif n_pivots_no_center == 6:
        # Six pivots, two in each direction of the orthogonal basis
        pivots[:, 0, :] = xyz + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :]
        pivots[:, 1, :] = xyz - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :]
        
        pivots[:, 2, :] = xyz + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :]
        pivots[:, 3, :] = xyz - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :]
        
        pivots[:, 4, :] = xyz + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        pivots[:, 5, :] = xyz - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        
    elif n_pivots_no_center == 8:
        # Eight pivots, one for each corner of the box aligned with the orthogonal basis
        pivots[:, 0, :] = (
            xyz 
            + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 1, :] = (
            xyz 
            + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 2, :] = (
            xyz 
            + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 3, :] = (
            xyz 
            + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 4, :] = (
            xyz 
            - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 5, :] = (
            xyz 
            - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 6, :] = (
            xyz 
            - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 7, :] = (
            xyz 
            - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
    else:
        raise ValueError(f"Invalid number of pivots: {n_pivots}")
    
    # Add the center point of the Gaussian to the pivots
    pivots = torch.cat(
        [
            pivots,  # (N_gaussians, n_pivots_no_center, 3)
            xyz.unsqueeze(1),  # (N_gaussians, 1, 3)
        ], 
        dim=1
    )  # (N_gaussians, n_pivots, 3)
    
    # Get pivot scales
    pivot_scales = 3. * scaling.detach().max(dim=-1, keepdim=True).values.unsqueeze(1).repeat(1, n_pivots, 1)  # (N_gaussians, n_pivots, 1)
    
    return pivots, pivot_scales  # (N_gaussians, n_pivots, 3), (N_gaussians, n_pivots, 1)


_GEODEL_WARNED = False


def _warn_geodel_missing_once() -> None:
    """Warn a single time that the fast Delaunay backend is unavailable.

    Once rather than per call: mesh extraction runs this repeatedly and a
    per-call warning would bury the rest of the log.
    """
    global _GEODEL_WARNED
    if _GEODEL_WARNED:
        return
    _GEODEL_WARNED = True
    _log.warning(
        "geodel is not installed - falling back to scipy.spatial.Delaunay, which is "
        "single-threaded and dominates meshing time. Install it with "
        "`bash install/build_extensions.sh` (or "
        "`pip install git+https://github.com/Anttwo/GeoDel`)."
    )


def _geodel_simplices(points_np: np.ndarray) -> np.ndarray:
    """Tetrahedralize with GeoDel (Geogram ParallelDelaunay3d).

    ``geodel.delaunay3d`` returns a **read-only, zero-copy uint32 view** onto
    Geogram's own buffers. The ``astype(np.int64)`` below therefore does two
    necessary jobs: it copies (so the result is writable and no longer keeps
    Geogram's allocation alive) and it leaves uint32, which torch handles
    poorly and whose sentinel ``NO_INDEX`` is ``0xFFFFFFFF`` rather than -1 —
    index arithmetic on the raw dtype would wrap instead of going negative.
    """
    from geodel import delaunay3d  # imported here so scipy-only installs work

    return delaunay3d(points_np).astype(np.int64)


def return_delaunay_tets(points: torch.Tensor, method: str = "geodel") -> torch.Tensor:
    """Delaunay-tetrahedralize ``points`` (N, 3) -> tet indices (M, 4), int64.

    ``method`` selects the backend:

      * ``"geodel"`` (default) — GeoDel / Geogram ParallelDelaunay3d, multi-threaded.
        Falls back to scipy with a one-time warning when geodel is not installed.
      * ``"scipy"`` — ``scipy.spatial.Delaunay`` (Qhull), single-threaded.

    To add a backend, add a branch returning an (M, 4) index array; the shared
    orientation fix below applies to every backend.

    Note the two are not guaranteed to agree tet-for-tet: cospherical or
    otherwise degenerate point sets admit several valid triangulations, and
    geodel additionally drops duplicate input points (indices still refer to
    the original array, so no reindexing is needed).
    """
    if method == "geodel":
        try:
            simplices = _geodel_simplices(points.detach().cpu().numpy())
        except ImportError:
            _warn_geodel_missing_once()
            method = "scipy"
    if method == "scipy":
        simplices = Delaunay(points.detach().cpu().numpy()).simplices.astype(np.int64)
    elif method != "geodel":
        raise ValueError(f"Invalid method: {method}")

    tets = torch.from_numpy(simplices).to(points.device).long()
    # Applied to every backend. GeoDel already guarantees positive orientation
    # so this is a no-op there, but it costs little next to the triangulation
    # itself and means a change to that guarantee cannot silently produce
    # inverted tets.
    return _orient_tets_positive(points, tets)
import logging
from typing import List, Optional
from tqdm import tqdm
import time
import torch
from surflo.rendering.gaussians import Gaussians
from surflo.structures.cameras import (
    Camera, 
    transform_points_world_to_view, 
    get_cameras_spatial_extent
)
from surflo.extraction.occupancy.pivots import (
    get_gaussian_pivots,
    get_regular_gaussian_pivots, 
    return_delaunay_tets
)
from surflo.extraction.extractors.masked import extract_mesh
from surflo.extraction.marchers.binary_search import refine_intersections_with_binary_search
from surflo.rendering.surflo import integrate_surflo

_log = logging.getLogger(__name__)


@torch.no_grad()
def evaluation_validation(
    view: Camera, points: torch.Tensor, inside: torch.Tensor
) -> torch.Tensor:
    if view.gt_mask is None:
        return inside

    points_cam = points @ view.R + view.T
    pts2d = points_cam[:, :2] / points_cam[:, 2:]
    pts2d = torch.addcmul(
        pts2d.new_tensor(
            [
                (view.Cx * 2.0 + 1.0) / view.image_width - 1.0,
                (view.Cy * 2.0 + 1.0) / view.image_height - 1.0,
            ]
        ),
        pts2d.new_tensor([view.Fx * 2.0 / view.image_width, view.Fy * 2.0 / view.image_height]),
        pts2d,
    )
    sampled_mask = torch.nn.functional.grid_sample(view.gt_mask[None].cuda(), pts2d[None, None], align_corners=True)
    return (sampled_mask.squeeze() > 0.5) & inside


@torch.no_grad()
def compute_valid_mask_single_view(
    fov_camera: Camera, points: torch.Tensor, znear=0.1,
) -> torch.Tensor:
    # Get parameters
    points_shape = points.shape
    Fx = fov_camera.Fx
    Fy = fov_camera.Fy
    Cx = fov_camera.Cx
    Cy = fov_camera.Cy
    H = fov_camera.image_height
    W = fov_camera.image_width
    
    # Transform points to camera space
    points_in_camera_space = transform_points_world_to_view(
        points=points.view(1, -1, 3),
        cameras=[fov_camera],
    ).squeeze(0)  # (N, 3)
    
    # Compute point projections
    pts_projections = torch.stack(
        [
            points_in_camera_space[:,0] * Fx / points_in_camera_space[:,2] + Cx,
            points_in_camera_space[:,1] * Fy / points_in_camera_space[:,2] + Cy
        ],
        -1
    ).float()
    
    # Compute frustum mask
    mask = (
        (pts_projections[:, 0] > 0) 
        & (pts_projections[:, 0] < W) 
        & (pts_projections[:, 1] > 0) 
        & (pts_projections[:, 1] < H) 
        & (points_in_camera_space[:,2] > znear)
    )
    
    return mask.view(points_shape[:-1])


@torch.no_grad()
def compute_valid_mask(points: torch.Tensor, views: List[Camera]) -> torch.Tensor:
    any_valid = []
    chunk_size = 20_000_000
    
    # Get scene radius
    scene_radius = get_cameras_spatial_extent(cameras=views)['radius'].item()
    znear = 0.02 * scene_radius
    
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        # Initialize valid mask as False
        any_valid_chunk = torch.zeros(point_chunk.shape[0], dtype=torch.bool, device="cuda")

        # Iterate over views
        for view in tqdm(views, desc="Rendering progress"):
            # Compute frustum mask for single view
            inside = compute_valid_mask_single_view(fov_camera=view, points=point_chunk, znear=znear).view(-1)
            assert inside.shape == any_valid_chunk.shape
            
            # Combine with GT mask if available
            valid_points = evaluation_validation(view, point_chunk, inside)
            
            # Update valid mask
            any_valid_chunk = torch.logical_or(any_valid_chunk, valid_points)

        any_valid.append(any_valid_chunk)

    return torch.cat(any_valid)


@torch.no_grad()
def pivot_extraction_with_binary_search( 
    views: List[Camera],  
    gaussians: Gaussians, 
    background: torch.Tensor, 
    kernel_size: float, 
    # SDF parameters
    sdf_mode: str = "approximate",
    sdf_isosurface_value: float = 0.0,
    # Regular pivot parameters
    use_regular_pivots: bool = True,
    std_factor: float = 3.33,
    # Normal-based pivot parameters
    n_pivots: int = 9,
    use_smallest_axis_as_normal: bool = True,
    # MTet parameters
    n_points_per_sdf_evaluation: int = 20_000_000,
    index_dtype: torch.dtype=torch.int32,
    use_valid_mask: bool = True,
    filter_large_edges: bool = True,
    collapse_large_edges: bool = False,
    mtet_on_cpu: bool = False,
    # Refinement
    n_binary_steps: int = 10,
    # Delaunay backend: "geodel" (fast, multi-threaded) or "scipy".
    delaunay_method: str = "geodel",
):
    assert index_dtype in [torch.int32, torch.int64], f"Invalid index dtype: {index_dtype}"
    
    # Get scene spatial extent
    scene_radius = get_cameras_spatial_extent(cameras=views)['radius'].item()
    
    # Define frustum parameters
    apply_frustum_culling = True
    standard_scale = 6.
    frustum_near = 0.02 * scene_radius / standard_scale
    frustum_far = 1e6 * scene_radius / standard_scale
        
    assert sdf_mode in ["exact", "approximate"], f"Invalid sdf mode: {sdf_mode}"

    if sdf_mode == "exact":
        raise NotImplementedError("Exact SDF mode is not implemented yet")

    elif sdf_mode == "approximate":
        @torch.no_grad()
        def evaluation_validation(view: Camera, points: torch.Tensor, inside: torch.Tensor) -> torch.Tensor:
            if view.gt_mask is None:
                return inside

            points_cam = points @ view.R + view.T
            pts2d = points_cam[:, :2] / points_cam[:, 2:]
            pts2d = torch.addcmul(
                pts2d.new_tensor(
                    [
                        (view.Cx * 2.0 + 1.0) / view.image_width - 1.0,
                        (view.Cy * 2.0 + 1.0) / view.image_height - 1.0,
                    ]
                ),
                pts2d.new_tensor([view.Fx * 2.0 / view.image_width, view.Fy * 2.0 / view.image_height]),
                pts2d,
            )
            sampled_mask = torch.nn.functional.grid_sample(view.gt_mask[None].cuda(), pts2d[None, None], align_corners=True)
            return (sampled_mask.squeeze() > 0.5) & inside
        
        @torch.no_grad()
        def sdf_function(points: torch.Tensor) -> torch.Tensor:
            final_sdf = []
            any_valid = []
            chunk_size = 20_000_000
            for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
                final_weight_chunk = torch.ones(point_chunk.shape[0], dtype=torch.float32, device="cuda")
                any_valid_chunk = torch.zeros(point_chunk.shape[0], dtype=torch.bool, device="cuda")
                for view in tqdm(views, desc="Rendering progress"):
                    ret = integrate_surflo(point_chunk, view, gaussians)
                    valid_points = evaluation_validation(view, point_chunk, ret["inside"])
                    any_valid_chunk = torch.logical_or(any_valid_chunk, valid_points)
                    final_weight_chunk = torch.where(
                        valid_points,
                        torch.min(ret["alpha_integrated"], final_weight_chunk),
                        final_weight_chunk,
                    )
                final_weight_chunk[torch.logical_not(any_valid_chunk)] = 0
                final_sdf_chunk = 0.5 - final_weight_chunk
                final_sdf.append(final_sdf_chunk)
                any_valid.append(any_valid_chunk)
            # return torch.cat(final_sdf), torch.cat(any_valid)
            return torch.cat(final_sdf).view(-1)  # (N,)  
    else:
        raise ValueError(f"Invalid sdf mode: {sdf_mode}")
    
    # Adjust isosurface value
    def sdf_function_wrapper(points, **kwargs):
        return sdf_function(points, **kwargs) - sdf_isosurface_value
    
    # Batchify the sdf function if necessary
    def batchified_sdf_function(points, **kwargs):
        all_sdf = []
        n_points = points.shape[0]
        
        if n_points > n_points_per_sdf_evaluation:
            n_batches = (n_points + n_points_per_sdf_evaluation - 1) // n_points_per_sdf_evaluation
        else:
            n_batches = 1
            
        for i_batch in range(n_batches):
            start_idx = i_batch * n_points_per_sdf_evaluation
            end_idx = min(start_idx + n_points_per_sdf_evaluation, n_points)
            batch_points = points[start_idx:end_idx]
            batch_sdf = sdf_function_wrapper(batch_points, **kwargs)
            all_sdf.append(batch_sdf)
        
        return torch.cat(all_sdf, dim=0)
    
    # Get pivots
    if use_regular_pivots:
        n_pivots = 9
        pivots, pivot_scales = get_regular_gaussian_pivots(
            gaussians=gaussians, 
            std_factor=std_factor,
            opacity_threshold=None,
            override_opacity=None,
        )
        pivots = pivots.view(-1, 3)
        pivot_scales = pivot_scales.view(-1, 1)
    else:
        pivot_results = get_gaussian_pivots(
            n_pivots=n_pivots,
            gaussians=gaussians,
            normals=None,
            std_factor=std_factor,
            use_smallest_axis_as_normal = use_smallest_axis_as_normal,
        )
        pivots, pivot_scales = pivot_results
        pivots = pivots.view(-1, 3)
        pivot_scales = pivot_scales.view(-1, 1)
    
    pivot_sdfs = batchified_sdf_function(pivots)
    
    # Compute valid mask
    if use_valid_mask:
        valid_mask = compute_valid_mask(points=pivots, views=views)
        pivot_sdfs[torch.logical_not(valid_mask)] = 0.5
    else:
        valid_mask = None
    
    # Compute Delaunay triangulation
    t0 = time.time()
    tets = return_delaunay_tets(pivots, method=delaunay_method).cpu()
    t1 = time.time()
    _log.info(
        f"Computed {tets.shape[0]} tets with Delaunay triangulation "
        f"({delaunay_method}): {t1 - t0}s"
    )
    
    # Extract mesh
    mesh, details = extract_mesh(
        delaunay_tets=tets.cuda(),
        pivots=pivots,
        pivots_sdf=pivot_sdfs,
        pivots_colors=None,
        pivots_scale=pivot_scales,
        filter_large_edges=filter_large_edges,
        collapse_large_edges=collapse_large_edges,
        return_details=True,
        mtet_on_cpu=mtet_on_cpu,
        valid=valid_mask,
    )
    torch.cuda.empty_cache()
    
    # Binary search
    if n_binary_steps > 0:        
        end_points = details['end_points']
        end_sdf = details['end_sdf']
        verts = refine_intersections_with_binary_search(
            end_points=end_points,
            end_sdf=end_sdf,
            sdf_function=batchified_sdf_function,
            n_binary_steps=n_binary_steps,
        )
        mesh.verts = verts

    return mesh
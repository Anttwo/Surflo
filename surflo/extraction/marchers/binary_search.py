import torch
from typing import Callable
import gc


def refine_intersections_with_binary_search(
    end_points:torch.Tensor,
    end_sdf:torch.Tensor,
    sdf_function:Callable,
    n_binary_steps:int,
) -> torch.Tensor:
    """
    Refine the intersected isosurface points with binary search.
    
    Args:
        end_points (torch.Tensor): The end points. (N_verts, 2, 3)
        end_sdf (torch.Tensor): The SDF values at the end points. (N_verts, 2, 1)
        sdf_function (Callable): The SDF function. Takes a tensor of points and returns the SDF values.
        n_binary_steps (int): The number of binary steps.
        
    Returns:
        refined_points (torch.Tensor): The refined points. (N_verts, 3)
    """
    
    left_points = end_points[:, 0, :].clone()  # (N_verts, 3)
    right_points = end_points[:, 1, :].clone()  # (N_verts, 3)
    left_sdf = end_sdf[:, 0, :].clone()  # (N_verts, 1)
    right_sdf = end_sdf[:, 1, :].clone()  # (N_verts, 1)
    points = (left_points + right_points) / 2  # (N_verts, 3)

    for step in range(n_binary_steps):
        mid_points = (left_points + right_points) / 2
        
        mid_sdf = sdf_function(mid_points)
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        
        points = (left_points + right_points) / 2  # (N_verts, 3)

        torch.cuda.empty_cache()
        gc.collect()
        
    return points
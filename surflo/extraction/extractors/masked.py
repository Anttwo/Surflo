import torch
from typing import Optional
import numpy as np
from surflo.extraction.marchers.tetrahedra import marching_tetrahedra
from surflo.extraction.marchers.masked_tetrahedra import marching_tetrahedra as marching_tetrahedra_masked
from surflo.structures.mesh import Meshes
from surflo.rendering.sh_utils import eval_sh


def extract_mesh(
    delaunay_tets:torch.Tensor,
    pivots:torch.Tensor,
    pivots_sdf:torch.Tensor,
    pivots_colors:Optional[torch.Tensor]=None,
    pivots_scale:Optional[torch.Tensor]=None,
    filter_large_edges:bool=False,
    collapse_large_edges:bool=False,
    return_details:bool=False,
    sdf_sh:Optional[torch.Tensor]=None,
    mtet_on_cpu:bool=False,
    valid:Optional[torch.Tensor]=None,
) -> Meshes:
    """
    Extract a mesh from a set of pivots, their SDF values and the Delaunay triangulation.

    Args:
        delaunay_tets (torch.Tensor): The Delaunay tetrahedra. Shape: (N_tets, 4).
        pivots (torch.Tensor): The pivots. Shape: (N_pivots, 3).
        pivots_sdf (torch.Tensor): The SDF values for the pivots. Shape: (N_pivots,).
        pivots_colors (torch.Tensor): The colors for the pivots. Shape: (N_pivots, 3).
        pivots_scale (torch.Tensor): The scales for the pivots. Shape: (N_pivots,).
        filter_large_edges (bool, optional): If True, filter out large edges. Defaults to True.
        collapse_large_edges (bool, optional): If True, collapse large edges onto the pivot with the smallest SDF value. Defaults to False.

    Returns:
        Meshes: The extracted mesh.
    """
    # If filtering or collapsing large edges, the pivot scales must be provided
    if filter_large_edges or collapse_large_edges:
        assert pivots_scale is not None
        
    if pivots_scale is None:
        pivots_scale = torch.ones_like(pivots_sdf)

    # Applying Marching Tetrahedra
    if valid is None:
        verts_list, scale_list, faces_list, verts_idx_list = marching_tetrahedra(
            vertices=pivots[None].cpu() if mtet_on_cpu else pivots[None],
            tets=delaunay_tets.cpu() if mtet_on_cpu else delaunay_tets,
            sdf=pivots_sdf.reshape(1, -1).cpu() if mtet_on_cpu else pivots_sdf.reshape(1, -1),
            scales=pivots_scale[None].cpu() if mtet_on_cpu else pivots_scale[None],
        )
    else:
        verts_list, scale_list, faces_list, verts_idx_list = marching_tetrahedra_masked(
            vertices=pivots[None].cpu() if mtet_on_cpu else pivots[None],
            tets=delaunay_tets.cpu() if mtet_on_cpu else delaunay_tets,
            sdf=pivots_sdf.reshape(1, -1).cpu() if mtet_on_cpu else pivots_sdf.reshape(1, -1),
            scales=pivots_scale[None].cpu() if mtet_on_cpu else pivots_scale[None],
            valids=valid[None].cpu() if mtet_on_cpu else valid[None],
        )
        
    end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
    if not mtet_on_cpu:
        end_points = end_points.cuda()
        end_sdf = end_sdf.cuda()
    end_scales = scale_list[0].cuda()  # (N_verts, 2, 1)
    if pivots_colors is not None:
        verts_idx = verts_idx_list[0].cuda()  # (N_verts, 2)
        verts_colors = pivots_colors[verts_idx]  # (N_verts, 2, 3)
    else:
        verts_colors = None
    
    # If spherical harmonics are provided, we use them to interpolate the SDF values along edges
    if sdf_sh is not None:
        # Add zero as the 0-degree SH component
        sdf_harmonics = torch.cat(
            [
                torch.zeros(sdf_sh.shape[0], 1, device=sdf_sh.device),  # (N_voronoi_points, 1)
                sdf_sh,  # (N_voronoi_points, N_sh-1)
            ],
            dim=-1,
        )  # (N_voronoi_points, N_sh)
        
        # Compute edge directions
        edge_dir = end_points[:, 1, :] - end_points[:, 0, :]  # (N_verts, 3)
        edge_dir_normalized = edge_dir / edge_dir.norm(dim=-1, keepdim=True)  # (N_verts, 3)
        edge_dir_normalized = torch.cat(
            [
                edge_dir_normalized.unsqueeze(1),  # (N_verts, 1, 3)
                -edge_dir_normalized.unsqueeze(1),  # (N_verts, 1, 3)
            ], 
            dim=1,
        )  # (N_verts, 2, 3)
        
        sh_deg = int(np.sqrt(sdf_harmonics.shape[-1])) - 1
        
        # We apply exponential to the SDF factors computed from spherical harmonics.
        # This ensures that:
        # - The SDF factors are always positive, so they do not change the sign of the SDF values
        # - The SDF factors are initialized to 1, as the SH coefficients are initialized to 0
        # - It is easier to encode high variance accross edge directions in the SDF values
        sdf_factors = torch.exp(
            eval_sh(
                deg=sh_deg,
                sh=sdf_harmonics[verts_idx_list[0].cuda()].unsqueeze(-2),  # (N_verts, 2, 1, N_sh)
                dirs=edge_dir_normalized,  # (N_verts, 2, 3)
            )  # (N_verts, 2, 1)
        )  # (N_verts, 2, 1)
        
        # Linear interpolation along edges, adjusted with SDF factors
        end_sdf = end_sdf * sdf_factors  # (N_verts, 2, 1)
    
    # Normalizing the SDF values to get the weights for the interpolation
    norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)  # (N_verts, 2, 1)
    verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]        
    faces = faces_list[0].cuda()  # (N_faces, 3)
    
    # If colors are provided, we interpolate them along edges
    if pivots_colors is not None:
        verts_colors = verts_colors[:, 0, :] * norm_sdf[:, 1, :] + verts_colors[:, 1, :] * norm_sdf[:, 0, :]

    # Filtering and collapsing edges based on the distance between the pivots
    if filter_large_edges or collapse_large_edges:
        dmtet_distance = torch.norm(end_points[:, 0, :] - end_points[:, 1, :], dim=-1)
        dmtet_scale = end_scales[:, 0, 0] + end_scales[:, 1, 0]
        dmtet_vertex_mask = (dmtet_distance <= dmtet_scale)
    
    #    > Filtering for large edges, inspired by GOF.
    #      If the edge between two pivots is larger than the sum 
    #      of the scales of the two corresponding Gaussians,
    #      The pivots should probably not be connected.
    if filter_large_edges:
        dmtet_face_mask = dmtet_vertex_mask[faces].all(axis=1)
        faces = faces[dmtet_face_mask]
    
    #    > The following option collapses big edges
    #      onto the pivot with the smallest SDF value.
    if collapse_large_edges:
        min_end_points = end_points[
            np.arange(end_points.shape[0]), 
            end_sdf.argmin(dim=1).flatten().cpu().numpy()
        ]  # TODO: Do the computation only for filtered vertices
        verts = torch.where(dmtet_vertex_mask[:, None], verts, min_end_points)

    # Build Mesh and return details if requested
    output_mesh = Meshes(verts=verts, faces=faces, verts_colors=verts_colors)
    if return_details:
        details = {
            'end_points': end_points,
            'end_sdf': end_sdf,
            'end_scales': end_scales,
            'end_idx': verts_idx_list[0].cuda(),
        }
        return output_mesh, details
    else:
        return output_mesh
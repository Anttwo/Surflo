import logging
from typing import List
import torch
from surflo.structures.cameras import Camera, is_in_view_frustum
from surflo.structures.mesh import Meshes

_log = logging.getLogger(__name__)


def frustum_cull_mesh(
    mesh:Meshes,
    viewpoint_cam:Camera,
) -> Meshes:
    """Cull the mesh based on the view frustum of the given viewpoint.

    Args:
        mesh (Meshes): The mesh to cull.
        viewpoint_cam (Camera): The viewpoint camera.

    Returns:
        Meshes: The culled mesh.
    """
    faces_mask = is_in_view_frustum(mesh.verts, viewpoint_cam)[mesh.faces].any(axis=1)
    return Meshes(
        verts=mesh.verts, 
        faces=mesh.faces[faces_mask], 
        verts_colors=mesh.verts_colors
    )


def compute_face_to_camera_minimum_distance(
    mesh: Meshes, 
    cameras: List[Camera], 
    batch_size: int=100_000,
    znear=None,
) -> torch.Tensor:
    """Compute the minimum distance to any camera for each face of the mesh.

    Args:
        mesh (Meshes): The mesh.
        cameras (List[Camera]): The cameras to compute the minimum distance to.
        batch_size (int, optional): The batch size to use for the computation. Defaults to 100_000.

    Returns:
        torch.Tensor: The minimum distance to any camera for each face of the mesh.
            Has shape (N_faces,).
    """
    # Compute camera centers
    all_cam_centers = torch.cat([camera_i.camera_center[None] for camera_i in cameras], dim=0)

    # Initialize minimum face to camera distance
    min_face_to_cam_distance = 1_000_000. * torch.ones(mesh.faces.shape[0], device=mesh.faces.device)  # (N_faces,)

    # Do it per batch of faces to avoid OOM
    for i in range(0, mesh.faces.shape[0], batch_size):
        batch_faces = mesh.faces[i:i+batch_size]  # (batch_size, 3)
        batch_face_centers = mesh.verts[batch_faces].mean(dim=1)  # (batch_size, 3)
        
        # For each camera, check if the face is in the view frustum
        in_view_mask = torch.zeros(
            batch_faces.shape[0], len(cameras), 
            device=mesh.faces.device, 
            dtype=torch.bool
        )  # (batch_size, N_cams)
        for j, camera in enumerate(cameras):
            in_view_mask[:, j] = is_in_view_frustum(
                points=batch_face_centers,
                camera=camera,
                znear=znear,
            )
        
        # Compute distance to all cameras
        batch_face_dist = (batch_face_centers[:, None] - all_cam_centers[None]).norm(dim=2)  # (batch_size, N_cams)
        batch_face_dist[~in_view_mask] = 1_000_000.
        
        # For each face, get the minimum distance to any camera
        batch_face_min_dist = batch_face_dist.min(dim=1).values  # (batch_size,)
        min_face_to_cam_distance[i:i+batch_size] = batch_face_min_dist
        
    return min_face_to_cam_distance  # (N_faces,)


def sample_points_on_mesh(
    mesh:Meshes, 
    n_points:int,
    adjust_with_distance_to_cameras:bool=False,
    cameras:List[Camera]=None,
    batch_size_for_face_to_camera_distance:int=100_000,
    znear=None,
    n_face_max:int=2**24,
    return_normals:bool=False,
) -> torch.Tensor:
    """Sample Gaussians on the surface of the mesh.
        If adjust_with_distance_to_cameras is True, the Gaussians are sampled so that 
        1. they look uniformly distributed once projected on the camera plane,
        2. they have similar size once projected on the camera plane.

    Args:
        mesh (Meshes): The mesh to sample on.
        n_points (int): The number of Gaussians to sample.
        adjust_with_distance_to_cameras (bool, optional): Whether to adjust the sampling and the size 
            with the distance to the cameras. Defaults to False.
        cameras (List[Camera], optional): The cameras to compute the minimum distance to. Defaults to None.
        batch_size_for_face_to_camera_distance (int, optional): The batch size to use for the computation of 
            the minimum distance to the cameras. Defaults to 100_000.
        znear (float, optional): The near plane of the view frustum. Defaults to None.
        n_face_max (int, optional): The maximum number of faces to sample. Defaults to 2**24.
        return_normals (bool, optional): Whether to return the normals of the sampled points. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The means, scales and rotations of the Gaussians.
            The means have shape (n_points, 3), the scales have shape (n_points, 3) and the rotations have shape (n_points, 4).
    """
    
    # If adjusting with the distance to the cameras, 
    # compute the minimum distance to the cameras for each face.
    if adjust_with_distance_to_cameras:
        assert cameras is not None
        min_face_to_cam_distance = compute_face_to_camera_minimum_distance(
            mesh=mesh,
            cameras=cameras,
            batch_size=batch_size_for_face_to_camera_distance,
            znear=znear,
        )  # (N_faces,)
        
    # Get triangle verts
    face_verts = mesh.verts[mesh.faces]  # N_faces, 3, 3
    
    # Get triangle normals
    if return_normals:
        face_normals = mesh.face_normals  # (N_faces, 3)
    
    # Get triangle areas
    face_areas = torch.linalg.norm(
        torch.cross(
            face_verts[:, 1] - face_verts[:, 0],  # (N_faces, 3)
            face_verts[:, 2] - face_verts[:, 0],  # (N_faces, 3)
            dim=-1,
        ),  # (N_faces, 3)
        dim=-1,
    ) / 2.0  # (N_faces,)
    
    # Get triangle probabilities.
    # Probas should be proportional to the area of the triangle.
    # If adjusting with the distance to the cameras, the proba is divided by the square of the distance
    # so that the proba is proportional to the area of the triangle ONCE projected on the camera plane.
    face_probs = face_areas.clone()
    if adjust_with_distance_to_cameras:
        prob_adjustment = 1. / (min_face_to_cam_distance ** 2)
        face_probs = face_probs * prob_adjustment
    face_probs = face_probs / torch.sum(face_probs)  # (N_faces,)
    
    # Sample triangles following the probabilities
    if face_probs.shape[0] > n_face_max:
        _log.info(f"More faces than the maximum number of faces to sample ({n_face_max}).")
        _log.info(f"       Sampling {n_face_max} faces randomly.")
        filtered_face_idx = torch.randperm(face_probs.shape[0], device=mesh.faces.device)[:n_face_max]
        filtered_face_probs = face_probs[filtered_face_idx]
        sampled_filtered_face_idx = torch.multinomial(input=filtered_face_probs, num_samples=n_points, replacement=True)  # (n_points,)
        sampled_face_idx = filtered_face_idx[sampled_filtered_face_idx]
    else:
        sampled_face_idx = torch.multinomial(input=face_probs, num_samples=n_points, replacement=True)  # (n_points,)
    sampled_face_verts = face_verts[sampled_face_idx]  # (n_points, 3, 3)
    if return_normals:
        sampled_face_normals = face_normals[sampled_face_idx]  # (n_points, 3)

    # Sample barycentric coordinates (summing to 1) in each sampled triangle
    sampled_barycentric_coords_1 = torch.rand(n_points, 1, device=mesh.faces.device)  # (n_points, 1)
    sampled_barycentric_coords_2 = (
        torch.rand(n_points, 1, device=mesh.faces.device) * (1. - sampled_barycentric_coords_1)
    )  # (n_points, 1)
    sampled_barycentric_coords_3 = 1. - sampled_barycentric_coords_1 - sampled_barycentric_coords_2  # (n_points, 1)
    sampled_barycentric_coords = torch.cat([sampled_barycentric_coords_1, sampled_barycentric_coords_2, sampled_barycentric_coords_3], dim=1)  # (n_points, 3)
    
    # Compute points
    points = (
        sampled_face_verts  # (n_points, 3, 3)
        * sampled_barycentric_coords[..., None]  # (n_points, 3, 1)
    ).sum(dim=1)  # (n_points, 3)
    
    # Return results
    if return_normals:
        return points, sampled_face_normals
    else:
        return points


def is_out_of_bound(
    points:torch.Tensor,
    views:List[Camera],
    znear=None,
) -> torch.Tensor:
    """Check if the points are out of the view frustum of all cameras.

    Args:
        points (torch.Tensor): The points to check. Shape (N, 3).
        views (List[Camera]): The cameras to check the points against.

    Returns:
        torch.Tensor: The mask of the points that are out of the view frustum. Shape (N,).
    """
    
    out_of_bound_mask = torch.ones_like(points[:, 0], dtype=torch.bool, device=points.device)
    
    for view in views:
        frustum_mask = is_in_view_frustum(points=points, camera=view, znear=znear)
        out_of_bound_mask[frustum_mask] = False
    
    return out_of_bound_mask

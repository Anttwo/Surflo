import logging
import torch
import numpy as np
from typing import List, Optional
from torch import nn
from surflo.structures.struct_utils import getWorld2ViewTorch, getProjectionMatrixTorch, focal2fov
from surflo.utils.geometry import inverse_se3
from surflo.structures.cameras import Camera
from pathlib import Path
from einops import einsum

_log = logging.getLogger(__name__)


def _select_nested_list(lst, dim: int, index: int):
    """Select an element from a nested list along a given depth level.
    
    dim=0: lst[index]
    dim=1: [row[index] for row in lst]
    dim=2: [[row[index] for row in sublst] for sublst in lst]
    ...
    """
    if dim == 0:
        return lst[index]
    return [_select_nested_list(sub, dim - 1, index) for sub in lst]


class MultiCameras(nn.Module):
    """
    A class to store one or multiple cameras.

    Args:
        colmap_id: The id of the camera in the colmap dataset. Recursive list of ints
        R (torch.Tensor): The rotation matrix of the camera. Shape: (..., 3, 3)
        T (torch.Tensor): The translation vector of the camera. Shape: (..., 3)
        FoVx (torch.Tensor): The horizontal field of view of the camera. Shape (...,)
        FoVy (torch.Tensor): The vertical field of view of the camera. Shape (...,)
        image (torch.Tensor): The image of the camera. Shape: (..., 3, H, W)
        image_name: The names of the images. Recursive list of strings
        uid: The uid of the camera. Recursive list of ints
        gt_alpha_mask: The ground truth alpha mask of the camera. Shape: (..., H, W)
        data_device: The device to store the camera data.
    """
    def __init__(
        self, 
        colmap_id: List[List[int]], 
        R: torch.Tensor, 
        T: torch.Tensor, 
        FoVx: torch.Tensor, 
        FoVy: torch.Tensor, 
        image: torch.Tensor, 
        gt_alpha_mask: torch.Tensor,
        image_name: List[List[str]], 
        uid: List[List[int]],
        trans: Optional[torch.Tensor] = None, 
        scale: Optional[float] = None, 
        data_device: str = "cuda",
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ):
        super(MultiCameras, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            _log.warning(f"Custom device {data_device} failed ({e}); falling back to the default cuda device.")
            self.data_device = torch.device("cuda")

        assert (image is not None) or (image_height is not None and image_width is not None)
        if image is not None:
            if image.device == self.data_device:
                self.original_image = image
            else:
                self.original_image = image.to(self.data_device).clamp(0.0, 1.0)
            self.image_width = self.original_image.shape[-1]
            self.image_height = self.original_image.shape[-2]
        else:
            self.original_image = None
            self.image_width = image_width
            self.image_height = image_height

        if gt_alpha_mask is not None:
            self.gt_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.gt_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        world_view_transform = getWorld2ViewTorch(R, T, trans, scale)
        view_world_transform = inverse_se3(world_view_transform)

        self.world_view_transform = world_view_transform.transpose(-1, -2).to(self.data_device)
        
        self.projection_matrix = getProjectionMatrixTorch(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(-1, -2).to(self.data_device)
        
        self.full_proj_transform = torch.bmm(
            self.world_view_transform.view(-1, 4, 4),
            self.projection_matrix.view(-1, 4, 4),
        ).view_as(self.world_view_transform)
        
        self.view_world_transform = view_world_transform.transpose(-1, -2)
        self.camera_center = self.view_world_transform[..., 3, :3]
        
        self._p3d_factors = torch.tensor([-1, -1, 1], device=self.data_device)
        if isinstance(self.image_width, torch.Tensor) and isinstance(self.image_height, torch.Tensor):
            self._image_size = torch.stack([self.image_width.squeeze(), self.image_height.squeeze()], dim=-1)
        else:
            self._image_size = torch.tensor([self.image_width, self.image_height], device=self.data_device)
    
    @property
    def batch_size(self) -> int:
        return self.R.shape[0]
    
    @property
    def n_cameras(self) -> int:
        return self.R.shape[1]
    
    def get_camera(self, scene_id: int, camera_id: int) -> Camera:
        assert scene_id < self.batch_size and camera_id < self.n_cameras
        R = self.R[scene_id, camera_id].detach().cpu().numpy()
        T = self.T[scene_id, camera_id].detach().cpu().numpy()
        FoVx = self.FoVx[scene_id, camera_id].item()
        FoVy = self.FoVy[scene_id, camera_id].item()
        if self.original_image is not None:
            image = self.original_image[scene_id, camera_id]
        else:
            image = torch.empty((3, self.image_height.item(), self.image_width.item()), device=self.data_device)
        gt_mask = self.gt_mask[scene_id, camera_id] if self.gt_mask is not None else None
        trans = self.trans[scene_id, camera_id].detach().cpu().numpy() if self.trans is not None else np.array([0.0, 0.0, 0.0])
        scale = self.scale if self.scale is not None else 1.0
        return Camera(
            colmap_id=self.colmap_id[scene_id][camera_id],
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=image,
            gt_alpha_mask=gt_mask,
            image_name=self.image_name[scene_id][camera_id],
            uid=self.uid[scene_id][camera_id],
            trans=trans,
            scale=scale,
            data_device=str(self.data_device),
        )

    def get_sub_cameras(self, dim: int, index: int) -> 'MultiCameras':
        """Create a new MultiCameras by selecting along a batch dimension.
        
        Reuses all pre-computed matrices (world_view_transform, etc.)
        by slicing them directly instead of recomputing.

        Args:
            dim (int): The batch dimension to select from
                (e.g. 0 for scene, 1 for camera).
            index (int): The index to select along that dimension.

        Returns:
            MultiCameras: A new MultiCameras with the selected sub-cameras.
        """
        obj = MultiCameras.__new__(MultiCameras)
        nn.Module.__init__(obj)

        obj.data_device = self.data_device
        obj.image_width = self.image_width
        obj.image_height = self.image_height
        obj.zfar = self.zfar
        obj.znear = self.znear
        obj.scale = self.scale

        obj.R = self.R.select(dim, index)
        obj.T = self.T.select(dim, index)
        obj.FoVx = self.FoVx.select(dim, index)
        obj.FoVy = self.FoVy.select(dim, index)
        if self.original_image is not None:
            obj.original_image = self.original_image.select(dim, index)
        else:
            obj.original_image = None
        obj.gt_mask = self.gt_mask.select(dim, index) if self.gt_mask is not None else None
        obj.trans = self.trans.select(dim, index) if self.trans is not None else None

        obj.world_view_transform = self.world_view_transform.select(dim, index)
        obj.projection_matrix = self.projection_matrix.select(dim, index)
        obj.full_proj_transform = self.full_proj_transform.select(dim, index)
        obj.view_world_transform = self.view_world_transform.select(dim, index)
        obj.camera_center = self.camera_center.select(dim, index)

        obj._p3d_factors = self._p3d_factors
        obj._image_size = self._image_size

        obj.colmap_id = _select_nested_list(self.colmap_id, dim, index)
        obj.image_name = _select_nested_list(self.image_name, dim, index)
        obj.uid = _select_nested_list(self.uid, dim, index)

        return obj
    

def get_cameras_spatial_extent(cameras:MultiCameras):
    """
    Get the spatial extent of the cameras.
    Given MultiCameras has a transform matrix with shape (..., N, 4, 4),
    the spatial extent is computed over the N cameras along the 3rd axis starting from the last dimension.

    Args:
        cameras (MultiCameras): 

    Returns:
        dict: A dictionary containing the translate, radius, and average camera center.
    """
    
    if cameras.world_view_transform.ndim < 3:
        raise ValueError("World view transform must have at least 3 dimensions")
    
    cam_centers = cameras.camera_center  # (..., N, 3)

    avg_cam_center = torch.mean(cam_centers, dim=-2, keepdim=True)  # (..., 1, 3)
    dist = torch.norm(cam_centers - avg_cam_center, dim=-1)  # (..., N)

    half_diagonal = torch.max(dist, dim=-1).values  # (...,)
    radius = half_diagonal * 1.1  # (...,)

    translate = -avg_cam_center  # (..., 1, 3)

    return {"translate": translate, "radius": radius, "avg_cam_center": avg_cam_center}


def transform_points_world_to_view(
    points:torch.Tensor,
    cameras:MultiCameras,
    use_p3d_convention:bool=False,
):
    """
    Transform points from world space to view space.

    Args:
        points (torch.Tensor): Should have shape (..., P, 3).
        cameras (MultiCameras): MultiCameras object, with world_view_transform of shape (..., 4, 4).
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (..., P, 3).
    """
    # Move 3D points to homogeneous coordinates
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (..., P, 4)
    
    # Transform points to view space
    view_points = einsum(
        points_h,  # (..., P, 4)
        cameras.world_view_transform,  # (..., 4, 4)
        "... i j, ... j p -> ... i p",
    )  # (..., P, 4)
    
    # Convert back to 3D coordinates
    view_points = view_points[..., :3]  # (..., P, 3)
    
    if use_p3d_convention:
        view_points = cameras._p3d_factors * view_points
    return view_points


def transform_points_view_to_world(
    points:torch.Tensor,
    cameras:MultiCameras,
    use_p3d_convention:bool=False,
):
    """Transform points from view space to world space.

    Args:
        points (torch.Tensor): Should have shape (..., P, 3).
        cameras (MultiCameras): MultiCameras object.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (..., P, 3).
    """
    view_world_transforms = cameras.view_world_transform  # (..., 4, 4)
    
    if use_p3d_convention:
        _points = cameras._p3d_factors * points
    else:
        _points = points
    
    # Convert points to homogeneous coordinates
    points_h = torch.cat([_points, torch.ones_like(_points[..., :1])], dim=-1)  # (..., P, 4)
    
    # Transform points to world space
    world_points = einsum(
        points_h,  # (..., P, 4)
        view_world_transforms,  # (..., 4, 4), 
        "... i j, ... j p -> ... i p",
    )  # (..., P, 4)
    
    # Convert points to 3D coordinates
    world_points = world_points[..., :3]  # (..., P, 3)
    
    return world_points


def transform_points_to_pixel_space(
    points:torch.Tensor,
    cameras:MultiCameras,
    points_are_already_in_view_space:bool=False,
    use_p3d_convention:bool=False,
    znear:float=1e-6,
    keep_float:bool=False,
):
    """Transform points from world space (3 coordinates) to pixel space (2 coordinates).

    Args:
        points (torch.Tensor): Should have shape (..., P, 3).
        cameras (MultiCameras): MultiCameras object.
        points_are_already_in_view_space (bool, optional): Defaults to False.
        use_p3d_convention (bool, optional): Defaults to False.
        znear (float, optional): Defaults to 1e-6.

    Returns:
        torch.Tensor: Has shape (..., P, 2). 
            In pixel space, (0, 0) is the center of the left-top pixel,
            and (W-1, H-1) is the center of the right-bottom pixel.
    """
    if points_are_already_in_view_space:
        full_proj_transforms = cameras.projection_matrix  # (..., 4, 4)
        if use_p3d_convention:
            points = cameras._p3d_factors * points
    else:
        full_proj_transforms = cameras.full_proj_transform  # (..., 4, 4)
    
    # Convert points to homogeneous coordinates
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (..., P, 4)
    
    # Move points to NDC space
    proj_points = einsum(
        points_h,  # (..., P, 4)
        full_proj_transforms,  # (..., 4, 4), 
        "... i j, ... j p -> ... i p",
    )  # (..., P, 4)
    proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (..., P, 2)
    # proj_points is currently in a normalized space where 
    # (-1, -1) is the left-top corner of the left-top pixel,
    # and (1, 1) is the right-bottom corner of the right-bottom pixel.

    # For converting to pixel space, we need to scale and shift the normalized coordinates
    # such that (-1/2, -1/2) is the left-top corner of the left-top pixel, 
    # and (H-1/2, W-1/2) is the right-bottom corner of the right-bottom pixel.
    image_size = cameras._image_size  # (2,)
    
    # proj_points = (1. + proj_points) * image_size / 2
    proj_points = (1. + proj_points) / 2 * image_size - 1./2.

    if keep_float:
        return proj_points        
    else:
        return torch.round(proj_points).long()
        
        
def get_multi_cameras_from_intrinsics_and_extrinsics(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_filepaths: Optional[list[Path]] = None,
    images: Optional[torch.Tensor] = None,
    data_device: str = "cuda",
) -> MultiCameras:
    """
    Convert COLMAP intrinsics and extrinsics to MultiCameras format. 
    Can store cameras for multiple scenes at once.

    Args:
        intrinsics (Tensor): COLMAP intrinsics of shape (B, N, 3, 3).
        extrinsics (Tensor): COLMAP extrinsics of shape (B, N, 3, 4).
        image_filepaths (list of Path): List of List of image file paths.
        data_device (str): Device to store the camera data.

    Returns:
        MultiCameras: A MultiCameras object.
    """
    B, N = intrinsics.shape[:2]
    assert intrinsics.shape == (B, N, 3, 3)
    assert extrinsics.shape == (B, N, 3, 4)
    
    # Get image names
    if image_filepaths is not None:
        assert len(image_filepaths) == B
        assert len(image_filepaths[0]) == N
    
        image_names = [[str(p.name) for p in image_filepaths[i]] for i in range(B)]
    else:
        image_names = [[f"image_{j}.png" for j in range(N)] for _ in range(B)]
        
    # IDs
    ids = [[j for j in range(N)] for _ in range(B)]
    
    # Get individual camera intrinsics
    cx, cy = intrinsics[:, :, 0, 2], intrinsics[:, :, 1, 2]  # (B, N)
    height, width = (2 * cy).int(), (2 * cx).int()  # (B, N)
    fx, fy = intrinsics[:, :, 0, 0], intrinsics[:, :, 1, 1]  # (B, N)
    FoVx = focal2fov(fx, width)  # (B, N)
    FoVy = focal2fov(fy, height)  # (B, N)
    
    # Get individual camera extrinsics
    R = extrinsics[:, :, :, :3].transpose(-1, -2)  # (B, N, 3, 3)
    T = extrinsics[:, :, :, 3]  # (B, N, 3)
    
    # Get images
    if images is not None:
        assert images.shape[:3] == (B, N, 3)
        images = images.to(data_device).clamp(0.0, 1.0)
        image_height = None
        image_width = None
    else:
        image_height, image_width = height[0, 0], width[0, 0]
        images = None
        
    return MultiCameras(
        colmap_id=ids,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        gt_alpha_mask=None,
        image=images,
        image_name=image_names,
        uid=ids,
        trans=None,
        scale=None,
        data_device=data_device,
        image_height=image_height,
        image_width=image_width,
    )

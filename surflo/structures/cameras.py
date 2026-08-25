#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import logging
import torch
from typing import List, Optional
from torch import nn
import numpy as np
from surflo.structures.struct_utils import getWorld2View2, getProjectionMatrix, focal2fov
from pathlib import Path
import math

_log = logging.getLogger(__name__)


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

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

        if image.device == self.data_device:
            self.original_image = image
        else:
            self.original_image = image.to(self.data_device).clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        # self.gt_mask = None
    
    @property
    def Fx(self):
        return self.image_width / (2 * math.tan(self.FoVx / 2.))

    @property
    def Fy(self):
        return self.image_height / (2 * math.tan(self.FoVy / 2.))

    @property
    def Cx(self):
        return float(self.image_width - 1) / 2

    @property
    def Cy(self):
        return float(self.image_height - 1) / 2


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def get_cameras_spatial_extent(cameras:List[Camera]):
    cam_centers = torch.cat([camera.camera_center.view(1, 3) for camera in cameras], dim=0)

    avg_cam_center = torch.mean(cam_centers, dim=0, keepdim=True)
    dist = torch.norm(cam_centers - avg_cam_center, dim=1, keepdim=True)

    half_diagonal = torch.max(dist)
    radius = half_diagonal * 1.1

    translate = -avg_cam_center

    return {"translate": translate, "radius": radius, "avg_cam_center": avg_cam_center}


def transform_points_world_to_view(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from world space to view space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    world_view_transforms = torch.stack([camera.world_view_transform for camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        view_points = factors * view_points  # (n_cameras, N, 3)
    return view_points


def transform_points_view_to_world(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from view space to world space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    view_world_transforms = torch.stack([camera.world_view_transform.inverse() for camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        _points = factors * points  # (n_cameras, N, 3)
    else:
        _points = points    
    points_h = torch.cat([_points, torch.ones_like(_points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    world_points = (points_h @ view_world_transforms)[..., :3]  # (n_cameras, N, 3)
    
    return world_points


def transform_points_to_pixel_space(
        points:torch.Tensor,
        cameras:List[Camera],
        points_are_already_in_view_space:bool=False,
        use_p3d_convention:bool=False,
        znear:float=1e-6,
        keep_float:bool=False,
):
    """Transform points from world space (3 coordinates) to pixel space (2 coordinates).

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        points_are_already_in_view_space (bool, optional): Defaults to False.
        use_p3d_convention (bool, optional): Defaults to False.
        znear (float, optional): Defaults to 1e-6.

    Returns:
        torch.Tensor: Has shape (n_cameras, N, 2). 
            In pixel space, (0, 0) is the center of the left-top pixel,
            and (W-1, H-1) is the center of the right-bottom pixel.
    """
    if points_are_already_in_view_space:
        full_proj_transforms = torch.stack([camera.projection_matrix for camera in cameras])  # (n_depth, 4, 4)
        if use_p3d_convention:
            points = torch.tensor([[[-1, -1, 1]]], device=points.device) * points
    else:
        full_proj_transforms = torch.stack([camera.full_proj_transform for camera in cameras])  # (n_cameras, 4, 4)
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    proj_points = points_h @ full_proj_transforms  # (n_cameras, N, 4)
    proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (n_cameras, N, 2)
    # proj_points is currently in a normalized space where 
    # (-1, -1) is the left-top corner of the left-top pixel,
    # and (1, 1) is the right-bottom corner of the right-bottom pixel.

    # For converting to pixel space, we need to scale and shift the normalized coordinates
    # such that (-1/2, -1/2) is the left-top corner of the left-top pixel, 
    # and (H-1/2, W-1/2) is the right-bottom corner of the right-bottom pixel.
    
    height, width = cameras[0].image_height, cameras[0].image_width
    image_size = torch.tensor([[width, height]], device=points.device)
    
    # proj_points = (1. + proj_points) * image_size / 2
    proj_points = (1. + proj_points) / 2 * image_size - 1./2.

    if keep_float:
        return proj_points        
    else:
        return torch.round(proj_points).long()
        
        
def get_cameras_from_intrinsics_and_extrinsics(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_filepaths: Optional[list[Path]] = None,
    images: Optional[torch.Tensor] = None,
    data_device: str = "cuda",
) -> list[Camera]:
    """
    Convert COLMAP intrinsics and extrinsics to GS cameras format.

    Args:
        intrinsics (Tensor): COLMAP intrinsics of shape (N, 3, 3).
        extrinsics (Tensor): COLMAP extrinsics of shape (N, 3, 4).
        image_filepaths (list of Path): List of image file paths.
        data_device (str): Device to store the camera data.

    Returns:
        tuple[Tensor, Tensor]: GS intrinsics and extrinsics.
    """
    N = intrinsics.shape[0]
    assert intrinsics.shape == (N, 3, 3)
    assert extrinsics.shape == (N, 3, 4)
    if image_filepaths is not None:
        assert len(image_filepaths) == N
        image_names = [str(p.name) for p in image_filepaths]
    else:
        image_names = [f"image_{i}.png" for i in range(N)]
    if images is not None:
        assert images.shape[:2] == (N, 3)
        images = images.to(data_device).clamp(0.0, 1.0)

    intrinsics_cpu = intrinsics.detach().cpu()
    extrinsics_cpu = extrinsics.detach().cpu()

    cameras = []
    for i in range(N):
        intrinsic = intrinsics_cpu[i]  # (3, 3)
        extrinsic = extrinsics_cpu[i]  # (3, 4)
        cx, cy = intrinsic[0, 2].item(), intrinsic[1, 2].item()
        height, width = int(2 * cy), int(2 * cx)
        fx, fy = intrinsic[0, 0].item(), intrinsic[1, 1].item()

        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)
        
        R = extrinsic[:, :3].transpose(1, 0).numpy()
        T = extrinsic[:, 3].numpy()
        
        if images is not None:
            image = images[i]
        else:
            image = torch.empty((3, height, width), device=data_device)
        
        cameras.append(
            Camera(
                colmap_id=i,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                image=image,
                gt_alpha_mask=None,
                image_name=image_names[i],
                uid=i,
                data_device=data_device,
            )
        )

    return cameras


def is_in_view_frustum(
    points:torch.Tensor,
    camera:Camera,
    znear:float=None,
) -> torch.Tensor:
    """_summary_

    Args:
        points (torch.Tensor): Tensor with shape (N, 3)
        camera (Camera): _description_
    """
    H, W = camera.image_height, camera.image_width
    
    _znear = camera.znear if znear is None else znear
    
    view_points = transform_points_world_to_view(
        points.view(1, -1, 3),
        cameras=[camera],
    )[0]  # (N, 3)
    
    pix_pts = transform_points_to_pixel_space(
        view_points.view(1, -1, 3),
        points_are_already_in_view_space=True,
        cameras=[camera],
    )[0]  # (N, 2)
    
    pix_x, pix_y, pix_z = pix_pts[..., 0], pix_pts[..., 1], view_points[..., 2]
    
    valid_mask = (
        (pix_x >= 0) & (pix_x <= W-1) 
        & (pix_y >= 0) & (pix_y <= H-1) 
        & (pix_z > _znear) & (pix_z < camera.zfar)
    )  # (N,)
    
    return valid_mask
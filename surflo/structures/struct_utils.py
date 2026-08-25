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

import torch
import math
import numpy as np
from typing import NamedTuple, Optional

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getWorld2ViewTorch(
    R: torch.Tensor, t: torch.Tensor, 
    translate: Optional[torch.Tensor] = None, scale: Optional[float] = None
):
    """
    Convert a rotation matrix and a translation vector to a world-to-view transformation matrix.

    Args:
        R (torch.Tensor): The rotation matrix of shape (..., 3, 3).
        t (torch.Tensor): The translation vector of shape (..., 3).
        translate (torch.Tensor, optional): The translation vector to apply to the camera center. Defaults to None.
        scale (float, optional): The scale factor to apply to the camera center. Defaults to None.

    Returns:
        torch.Tensor: The world-to-view transformation matrix of shape (..., 4, 4).
    """
    
    Rt = torch.zeros(*R.shape[:-2], 4, 4, device=R.device)
    Rt[..., :3, :3] = R.transpose(-2, -1)
    Rt[..., :3, 3] = t
    Rt[..., 3, 3] = 1.0
    
    if (translate is not None) or (scale is not None):
        C2W = torch.linalg.inv(Rt)
        cam_center = C2W[..., :3, 3]
        if translate is not None:
            cam_center = cam_center + translate
        if scale is not None:
            cam_center = cam_center * scale
        C2W[..., :3, 3] = cam_center
        Rt = torch.linalg.inv(C2W)
    
    return Rt


def getProjectionMatrixTorch(
    znear: float, zfar: float, 
    fovX: torch.Tensor, fovY: torch.Tensor
):
    """
    Get the projection matrix for a camera.

    Args:
        znear (float): The near clipping plane.
        zfar (float): The far clipping plane.
        fovX (torch.Tensor): The horizontal field of view. Has shape (...,).
        fovY (torch.Tensor): The vertical field of view. Has shape (...,).

    Returns:
        torch.Tensor: The projection matrix of shape (..., 4, 4).
    """
    tanHalfFovY = torch.tan((fovY / 2))  # (..., )
    tanHalfFovX = torch.tan((fovX / 2))  # (..., )

    top = tanHalfFovY * znear  # (..., )
    bottom = -top  # (..., )
    right = tanHalfFovX * znear  # (..., )
    left = -right  # (..., )

    P = torch.zeros(*fovX.shape, 4, 4, device=fovX.device)  # (..., 4, 4)

    z_sign = 1.0

    P[..., 0, 0] = 2.0 * znear / (right - left)
    P[..., 1, 1] = 2.0 * znear / (top - bottom)
    P[..., 0, 2] = (right + left) / (right - left)
    P[..., 1, 2] = (top + bottom) / (top - bottom)
    P[..., 3, 2] = z_sign
    P[..., 2, 2] = z_sign * zfar / (zfar - znear)
    P[..., 2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    if not isinstance(fov, torch.Tensor):
        return pixels / (2. * math.tan(fov / 2.))
    else:
        return pixels / (2. * torch.tan(fov / 2.))

def focal2fov(focal, pixels):
    if not isinstance(focal, torch.Tensor):
        return 2. * math.atan(pixels / ( 2. * focal))
    else:
        return 2. * torch.atan(pixels / (2. * focal))
